# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from .ssd_combined import mamba_chunk_scan_combined
from .ssd_combined import mamba_split_conv1d_scan_combined


class Mamba2(nn.Module):
    def __init__(
        self,
        d_model, # 模型输入输出维度 D
        d_inner=0, # 模型内部维度
        d_state=128, # 状态空间的维度 N [从Mamba-1的16扩大到128]
        d_conv=4, # 1D卷积的卷积核大小
        conv_init=None,
        expand=2, # 扩展因子 E (the controllable expansion factor)
        headdim=64, # head 的维度 P, 即一个单头有P个通道 [Mamba-1的P=1(SISO)]
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1, # 固定值不可改！参数 𝐵 和 𝐶 投影在 𝑋 头部之间只存在 1 个单头进行共享，类似于多值注意力(Multi-value Attn. 论文式20)
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True, # 是否在最后的输出投影层前添加一个额外的规范化层
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False, # 其他层（如线性层）是否使用偏置项
        conv_bias=True, # 卷积层是否使用偏置项
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True, # 是否应用序列并行策略
        device=None,
        dtype=None,
        aux_feature_size=0,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        # self.d_inner: 内部维度，即扩展后的维度
        self.d_inner = d_inner // self.world_size if d_inner else (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == d_inner if d_inner else self.expand * self.d_model # 确保整除
        self.headdim = headdim
        # self.d_ssm: ssm的（总）维度
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        # self.nheads: 多头SSM的haed个数    Mamba2中使用的是多头SSM，由ssm的总维度self.d_ssm 和 单个head的维度self.headdim 计算得出
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.aux_feature_size = aux_feature_size

        # Order: [z, x, B, C, dt]   z,x: self.d_inner;  B,C: self.ngroups * self.d_state; dt: self.nheads
        d_in_proj = 2 * self.d_inner if self.aux_feature_size else 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        # 输入线性变换层
            # 把Mamba Block结构的两个分支中的输入线性层合并，用一个线性层实现！！
        """ 改动1：输入线性变换层生成 x, z 的同时也生成了 SSM 参数 B,C,Δ
                      此时，B,C,Δ 是层输入的函数（并行投影），而不是作为 SSM 输入 x 的函数 """
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm if self.aux_feature_size else self.d_ssm + 2 * self.ngroups * self.d_state # Order: [x, B, C]
        # 一维卷积层，执行深度卷积（Mamba模型的特色之一，用于处理序列数据）
            # 沿着序列长度L的方向应用卷积核
            # 每个输入通道被单独卷积到每个输出通道，意味着每个输出通道的结果是通过仅与一个输入通道卷积得到的
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim, # groups=in_channels: 输入的通道分成in_channels组(每一组就一个通道)，此时每一个输出通道只需要在其中一个输入通道上做卷积。
            padding=d_conv - 1,
            **factory_kwargs,
        ) # B*in_channels*L → B*out_channels*(L + d_conv-1)     in_channels=out_channels=conv_dim
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU() # 激活函数固定为SiLU

        """ 改动1.5：删除了将输入映射为SSM参数(B,C,Δ)的两个线性变换层————B,C,Δ在块的开头由输入线性变换层self.in_proj生成 """

        # Initialize log dt bias    so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        ## ssm参数 A、D 与输入无关
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        # 初始化SSM的矩阵A
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range) # (nheads)
        A_log = torch.log(A).to(dtype=dtype) # also Keep A_log in fp32 in update version: delete ".to(dtype=dtype)"
        # 矩阵A的对数值，作为一个可训练参数
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        # 矩阵D初始化为全1，也是一个可训练参数 shape:(self.d_ssm,) / (self.nheads,)  [self.d_ssm=self.nheads*self.headdim]
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        """  改动2：在最后的输出投影层前添加了一个额外的norm层，就像在NormFormer中一样，以提高稳定性 """
        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        # 输出线性变换层，用于输出的投影
        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, inference_params=None, B=None, C=None, dt=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        # 要么不使用高阶特征，ssm参数由输入构造；否则需确保传入参数B, C, dt不为None
        assert self.aux_feature_size==0 or B is not None
        # assert self.aux_feature_size==0 or (self.aux_feature_size and B is not None)

        # 获取输入的维度：batch, seqlen, dim
        seqlen_og = seqlen
        if seqlen is None: # 输入u是三维
            batch, seqlen, dim = u.shape
        else: # 输入u是二维
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state, (B, C, dt))
                return out

        # Order: [z, x, B, C, dt]
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None: # 二维转三维
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        
        # 这里的负号-是因为在ssm中，矩阵A通常表示的是一个离散时间系统的转换矩阵，它描述了系统状态随时间的演变
        # 在许多情况下，A矩阵的元素应该是负的，以确保系统的稳定性
        # 这是因为在离散时间系统中，我们希望系统的状态随着时间的推移而衰减或稳定下来，而不是增长，从而避免系统变得不稳定或发散
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state) [Mamba-2:(nheads), Mamba-1:(d_inner, d_state)]
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit) # ？
        
        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                B=rearrange(B, "b l (g n) -> b l g n", g=self.ngroups) if self.aux_feature_size else None, # (B, L, self.ngroups, self.d_state)
                C=rearrange(C, "b l (g n) -> b l g n", g=self.ngroups) if self.aux_feature_size else None, # (B, L, self.ngroups, self.d_state)
                dt=dt,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            # 三维转二维——变回去
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            # 使用某个并行策略对out进行处理：序列并行（reduce_scatter） or 张量并行（all_reduce）
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            # zxbcdt.shape[-1] = d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
            # self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
            """ d_mlp = self.d_inner - self.d_ssm """
            if self.aux_feature_size:
                d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm) // 2
                z0, x0, z, xBC = torch.split(
                    zxbcdt,
                    [d_mlp, d_mlp, self.d_ssm, self.d_ssm],
                    dim=-1
                )
            else:
                d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
                """ 将输入线性层self.in_proj的输出zxbcdt分成几部分：(z0, x0,) z, xBC, dt
                        注：当d_mlp > 0时，才可能split出 z0, x0 """
                z0, x0, z, xBC, dt = torch.split(
                    zxbcdt,
                    [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                    dim=-1
                )
                """
                xBC还要输入到Conv+激活函数，然后再进行分割生成x, B, C
                dt，即SSM参数Δ，在此已生成完毕，无需再进行任何操作！
                """

            # Compute short convolution
            if conv_state is not None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l") # transpose xBC
                conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                # self.conv1d输出的序列长度L不变，无需切片？不应该是L + d_conv-1嘛……
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2) # b l d -> b d l -> b l d
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            
            if self.aux_feature_size: # 变量xBC即为x，B,C 用传入参数
                x = xBC
            else:
                # 从Conv的输出直接分割出x, B, C   [删除了将SSM输入x映射为SSM参数(B,C,Δ)的线性投影层]
                x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            
            # 新加速算法
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim), # (B, L, self.nheads, self.headdim)
                dt, # (B, L, self.nheads)
                A, # (nheads)
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups), # (B, L, self.ngroups, self.d_state)
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups), # (B, L, self.ngroups, self.d_state)
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D, # (self.nheads, self.headdim) / (self.nheads,)
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None, # (B, L, self.nheads, self.headdim) / None
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b l h p -> b l (h p)") # 输出y形状要reshape回去
            
            if self.rmsnorm: # 过新加的norm层：Mamba block的一二分支输出相乘+norm
                y = self.norm(y, z)
            
            if d_mlp > 0: # （norm后的）SSM输出y需额外cat上F.silu(z0) * x0
                y = torch.cat([F.silu(z0) * x0, y], dim=-1) # (B, L, d_ssm) -> (B, L, d_inner)
            if seqlen_og is not None: # 三维转二维
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y) # 输出线性变换: (B, L, d_inner) -> (B, L, D) 
        
        return out
    """ NO DEBUG """
    def step(self, hidden_states, conv_state, ssm_state, bcdt=(None, None, None)): # hidden_states only have 1 token, seqlen=1
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        if self.aux_feature_size:
            assert bcdt[0] is not None and bcdt[0].shape[1] == 1, "Only support decoding with 1 token at a time for now"
        
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)   2D=d_in_proj
        
        # d_mlp = self.d_inner - self.d_ssm
        if self.aux_feature_size:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm) // 2
            z0, x0, z, xBC = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm],
                dim=-1
            )
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            # 将zxbcdt分成几部分：(z0, x0,) z, xBC, dt
                # 当d_mlp > 0时，才可能split出 z0, x0
                # xBC还要输入到Conv+激活函数，然后再进行分割生成x, B, C
                # dt，即SSM参数Δ，在此已生成完毕，无需再进行任何操作！
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )

        # Conv step
        """ 相比于Mamba-1，仅将变量x替换为xBC，其他不变 """
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        if self.aux_feature_size: # 变量xBC即为x，B,C 用传入参数
            x = xBC
            B, C, dt = bcdt[0].squeeze(1), bcdt[1].squeeze(1), bcdt[2].squeeze(1)
        else:
            # 从Conv的输出直接分割出x, B, C   [删除了将SSM输入x映射为SSM参数(B,C,Δ)的线性投影层]
                # x: (B, self.d_ssm)  B,C: (B, self.ngroups*self.d_state)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            # 对A，使用ZOH离散化
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim) # (B, self.d_ssm) -> (B, self.nheads, self.headdim)
            # 对B，使用一个简化的Euler discretization
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x) # (B, self.nheads, self.headdim, self.d_state)
            # SSM式1: h_t = Ah_{t-1} + Bx_t
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx) # (B, self.nheads, self.headdim, self.d_state)
            # SSM式2: y_t = Ch_t
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x # +Dx, 残差
            y = rearrange(y, "b h p -> b (h p)") # 输出y形状再reshape回去
            if not self.rmsnorm: # Mamba block的一二分支输出相乘
                y = y * self.act(z)  # (B D)
        else:
            # 对存储的原始参数 A,dt,dt_bias,D 沿单个head的内部维度P（以及状态空间的维度N——参数A）创建重复的序列
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            # 调整B, C的形状
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim) # reshape输入x
            if not self.rmsnorm: # 没有额外加norm层，计算SSM输出y要用到z，则要和x一样对z做reshape
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)") # 输出y形状再reshape回去
        
        if self.rmsnorm: # 过新加的norm层：Mamba block的一二分支输出相乘+norm
            y = self.norm(y, z)
        
        if d_mlp > 0: # （norm后的）SSM输出y需额外cat上F.silu(z0) * x0
            y = torch.cat([F.silu(z0) * x0, y], dim=-1) # (B, d_ssm) -> (B, d_inner)
        out = self.out_proj(y) # (B d_model)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.conv1d.weight.shape[0], self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


""" mixer（即Mamba block/MHA）后可以再加 norm+MLP（新增），其他不变 """
class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
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
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                residual = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)