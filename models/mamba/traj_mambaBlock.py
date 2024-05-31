# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from .selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
'''
b: 批量大小(batch size), 对应Mamba论文中algorithm 2中的B
l: 序列长度，对应Mamba论文中algorithm 2中的L
d / d_model: 隐藏层的维度大小
n / d_state: 状态维度，对应Mamba论文中algorithm 2中的N
expand: 扩张系数，Mamba论文3.4节的E
d_in / d_inner: d*expand, 对应Mamba论文中algorithm 2中的D
A,B,C,D对应的是状态空间模型的参数。其中B,C是依赖于输入的，A,D并不是。
Δ / delta: 依赖于输入的时间步长。
dt_rank: Δ的秩，对应Mamba论文中3.6节的“parameterization of Δ”
'''

class Mamba(nn.Module): # 论文中的Figure 2（Mamba block） + Algorithm 2
    '''
    Mamba Block

    Mamba模型中的核心模块，负责执行序列数据的处理和状态空间模型的更新
    '''
    def __init__(
        self,
        d_model, # 模型的隐藏层维度 D（768,1024,1536,2048,2560,...）
        d_state=16, # 状态空间的维度 N
        d_conv=4, # 1D卷积的卷积核大小
        expand=2, # 扩展因子 E (the controllable expansion factor)
        dt_rank="auto", # 定义输入依赖的参数Δ的秩，'auto'表示自动设置
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True, # 卷积层是否使用偏置项
        bias=False, # 其他层（如线性层）是否使用偏置项
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        aux_feature_size=0,
        # num_roads=None,
        # road_embedding_size=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        # 计算内部维度，即扩展后的维度 (expanding the model dimension D by the controllable expansion factor E)
        self.d_inner = int(self.expand * self.d_model)
        # dt_rank="auto"时，根据隐藏层维度自动计算Δ的秩
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank # dt_rank in "auto" mode = 48,64,96,128,160,...
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        
        self.aux_feature_size = aux_feature_size
        # self.road_embedding_size = road_embedding_size
        
        # 输入线性变换层
            # 把论文Mamba Block结构的两个分支中的输入线性层合并，用一个线性层实现！！
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 一维卷积层，执行深度卷积（Mamba模型的特色之一，用于处理序列数据）
            # 沿着序列长度L的方向应用卷积核
            # 每个输入通道被单独卷积到每个输出通道，意味着每个输出通道的结果是通过仅与一个输入通道卷积得到的
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, # 意味着卷积操作是在d_model维的特征空间内独立进行的————啊？
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner, # groups=in_channels: 输入的通道分成in_channels组(每一组就一个通道)，此时每一个输出通道只需要在其中一个输入通道上做卷积。
            padding=d_conv - 1,
            **factory_kwargs,
        ) # B*in_channels*L → B*out_channels*(L + d_conv-1)     in_channels=out_channels=d_inner

        # 激活函数固定为SiLU
        self.activation = "silu"
        self.act = nn.SiLU()

        # if self.road_embedding_size is not None:
        #     self.road_embedding = nn.Embedding(num_roads, self.road_embedding_size)

        # 线性变换层（2个），用于将输入映射到状态空间模型的参数
        # self.x_proj 对输入x做映射，生成依赖于输入的SSM参数Δ、B和C
        # self.x_proj = nn.Linear(
        #     self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        # )
        self.x_proj = nn.Linear(self.aux_feature_size if self.aux_feature_size else self.d_inner, # self.aux_feature_size - 1 + self.road_embedding_size
                                self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        # self.dt_proj 将Δ从dt_rank维度映射到d_inner维度
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        '''ssm参数中的 A D 是与输入无关的'''
        # 创建一个重复的序列，用于初始化SSM的矩阵A
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous() # (d_inner, d_state)
        A_log = torch.log(A)  # Keep A_log in fp32
        # 矩阵A的对数值，作为一个可训练参数
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        # 矩阵D初始化为全1，也是一个可训练参数 shape(d_inner)
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # 输出线性变换层，用于输出的投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    # Input: x(𝙱, 𝙻, 𝙳) → Output: y(𝙱, 𝙻, 𝙳)
    def forward(self, hidden_states, aux_features=None, inference_params=None):
        """
        论文中 Mamba block 结构的整体流程

        包含Algorithm 2————可参考run_SSM(A, B, C, u) in The Annotated S4，深入理解ssm的运行流程并对比S6和S4

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape # 获取输入的维度

        # src_params = []
        # for key, value in aux_features.items():
        #     if key in ["road", "weekday", "seq_i"]: continue
        #     src_params.append(value)
        # src_params = torch.stack(src_params, dim=-1)
        # if aux_features.get("road") is not None:
        #     road_embedding = self.road_embedding(aux_features["road"].int())
        #     src_params = torch.cat([src_params, road_embedding],dim=-1)

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, aux_features, conv_state, ssm_state)
                return out

        '''使用self.in_proj对输入进行线性变换     且将权重和偏置分离使用，因为可能不用偏置'''
        # We do matmul and transpose BLH -> HBL at the same time
            # 此形状调整是为了适配后续的一维卷积层self.conv1d，其期望输入的形状为(B, channels, L)
        xz = rearrange(
            # 与权重进行矩阵乘法前，先将形状变为[D, (B L)]
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", # shape [d_inner * 2, (B L)] -> (B, d_inner * 2, L)
            l=seqlen,
        ) 
        if self.in_proj.bias is not None: # 使用偏置的情况，额外加偏置
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        src_params = None
        if self.aux_feature_size:
            src_params = rearrange(aux_features, "b l d -> b d l")
        '''
        这里的负号-是因为在ssm中，矩阵A通常表示的是一个离散时间系统的转换矩阵，它描述了系统状态随时间的演变
        在许多情况下，A矩阵的元素应该是负的，以确保系统的稳定性
        这是因为在离散时间系统中，我们希望系统的状态随着时间的推移而衰减或稳定下来，而不是增长，从而避免系统变得不稳定或发散
        '''
        A = -torch.exp(self.A_log.float())  # shape (d_inner, d_state)
        
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            # 定义于ops/selective_scan_interface.py
            # 梳理注释见下【或同.py文件中的def mamba_inner_ref()】
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(), # 取D的值
                src_params=src_params,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            # 将输入线性层self.in_proj的输出xz分为两部分：x和z  shape(B, d_inner, L)
            x, z = xz.chunk(2, dim=1) 
            '''
            x是论文Mamba Block结构第一个分支所用的数据，用于后续变换，生成ssm所需要的参数
            z是论文Mamba Block结构第二个分支所用的数据
            '''
            
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            
            if causal_conv1d_fn is None:
                # 卷积操作的输出shape (B, d_inner, L + d_conv-1)
                # 通过切片操作[..., :seqlen]只保留了序列长度为L的输出，因为我们只对序列中的前L个元素感兴趣
                # 应用SiLU激活函数——1st activation in Mamba Block
                x = self.act(self.conv1d(x)[..., :seqlen]) # shape (B, d_inner, L)
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            
            # 应用x的投影变换 [(B L), d_inner] -> [(B L), dt_rank+2*d_state]
                # 变换前要调整卷积后的x形状     确保数据在后续层中的流动是连贯的，特别是当数据传递给后续的Mamba块或其他层时
            x_dbl = self.x_proj(rearrange(src_params if self.aux_feature_size else x, "b d l -> (b l) d"))  # (bl d)
            
            # 分割出Δ, B, C
            '''ssm参数中的 Δ, B, C 是与输入有关的'''
            # Δ: [(B L), dt_rank]   B, C: [(B L), d_state]
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            
            # 使用self.dt_proj计算Δ
                # Δ在SSM中的作用，类似于RNN中的门控机制
            dt = self.dt_proj.weight @ dt.t() # shape [d_inner, (B L)]
            '''注：对Δ应用softplus激活函数的操作在选择性扫描算法函数def selective_scan_ref中'''
            
            # 调整Δ, B, C的形状
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen) # shape (B, d_inner, L)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous() # shape (B, d_state, L)
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous() # shape (B, d_state, L)
            
            assert self.activation in ["silu", "swish"] # 这里为啥要判断一下self.activation？后面也没再用了————好的，这是句废代码
            
            '''
            选择性扫描算法  
            定义于ops/selective_scan_interface.py，梳理注释见同.py文件中的def selective_scan_ref()
            '''
            y = selective_scan_fn(
                x, # (B, d_inner, L)
                dt, # (B, d_inner, L)
                A, # (d_inner, d_state)
                B, # (B, d_state, L)
                C, # (B, d_state, L)
                self.D.float(), # 取D的值 shape (d_inner)
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ) #  shape (B, d_inner, L)
            
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            
            # 应用输出线性变换
            y = rearrange(y, "b d l -> b l d") # 调整y的形状
            out = self.out_proj(y) # (B, L, d_inner) -> (B, L, D)

        return out

    def step(self, hidden_states, aux_features, conv_state, ssm_state): # hidden_states: (B 1 d_model)
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        if self.aux_feature_size:
            assert aux_features.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)   D: d_inner, the expanded model dimension
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(aux_features.squeeze(1) if self.aux_feature_size else x)  # (B dt_rank+2*d_state)  为dt + B + C
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A)) # (B d_inner d_state)   b:B, d:d_inner(D), n:d_state
            dB = torch.einsum("bd,bn->bdn", dt, B) # (B d_inner d_state)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB) # (B d_inner d_state)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C) # (B d_inner)
            y = y + self.D.to(dtype) * x # (B d_inner)   self.D:(D), x:(B D)
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y) # (B d_model)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            # 初始化conv_state，ssm_state为0 tensor
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state) # 加入到inference_params中
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # Block的核心组件：Mamba block
        self.mixer = mixer_cls(dim) # dim即d_model
        # 归一化模块，用于在数据送入Mamba block前进行归一化操作
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, aux_features=None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required). shape (B, L, D)
            # residual: hidden_states = Mixer(LN(residual))
            residual: shape (B, L, D)
        
        Returns: 
            hidden_states, residual: updated params with same shape
                residual = hidden_states (+ residual)
                hidden_states = Mixer(LN(residual))
        """
        # if-else中的操作同class MixerModel forward函数中的相应部分(self.fused_add_norm)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states # add
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype)) # LayerNorm
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True, # need to return the residual
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )

        hidden_states = self.mixer(hidden_states, aux_features=aux_features, inference_params=inference_params) # 获得class Manmba的forward函数的输出
        
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)