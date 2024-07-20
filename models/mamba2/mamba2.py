# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os
import copy

from collections import namedtuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# from mamba_ssm.models.config_mamba import MambaConfig
# from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.modules.mamba2 import Mamba2
# from models.mamba.traj_mambaBlock import Mamba
from .traj_mambaBlock2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
# from mamba_ssm.utils.generation import GenerationMixin
# from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.models.mixer_seq_simple import _init_weights

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

""" 更灵活地构建模型组件block：
   mixer可以选用Mamba1(默认，与v1版本一致)、Mamba2 和 multihead attention
   block中可加入MLP（新增），hidden dim 由参数 d_intermediate 给定

最近新出的混合模型（Jamba、Zamba）增加了一些注意力层来提高模型质量。
基于这些研究，作者将 4-6 个注意力层与 Mamba-2 层混合，发现其表现优于 Transformer++ 和纯 Mamba-2，
因而得出：注意力和 SSM 是互补的 """
def create_block(
    d_model,
    d_intermediate,
    aux_feature_size = 0,
    d_state = 128,
    headdim = 64,
    d_inner = 0,
    ssm_cfg=None,
    attn_layer_idx=None, # 指定某些层的mixer选用multihead attention
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # 修改：该层不为注意力层时，mixer_cls只能使用Mamba2
        mixer_cls = partial(Mamba2,
                            d_inner=d_inner,
                            d_state=d_state,
                            headdim=headdim,
                            aux_feature_size=aux_feature_size,
                            layer_idx=layer_idx,
                            **ssm_cfg,
                            **factory_kwargs
                            )
        # # Create a copy of the config to modify
        # ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        # # 改成了默认用Mamba2！！！！
        # ssm_layer = ssm_cfg.pop("layer", "Mamba2")
        # if ssm_layer not in ["Mamba1", "Mamba2"]:
        #     raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        # mixer_cls = partial(
        #     Mamba2 if ssm_layer == "Mamba2" else Mamba,
        #     aux_feature_size=aux_feature_size,
        #     layer_idx=layer_idx,
        #     **ssm_cfg,
        #     **factory_kwargs
        # )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


""" 无大的改动，仅因block构造的新增部分而增加了相应的传入参数 """
class TrajMixerModel2(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        # vocab_size: int,
        aux_feature_size: int,
        d_state: int = 128, 
        headdim: int = 64,
        d_inner: int = 0, 
        ssm_cfg=None,
        # attn_layer_idx=None,
        # attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        # 与class Mamba2的相应参数的默认值保持一致↓
        bias=False, # 其他层（如线性层）是否使用偏置项
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    aux_feature_size=aux_feature_size,
                    d_state=d_state,
                    headdim=headdim,
                    d_inner=d_inner,
                    ssm_cfg=ssm_cfg,
                    # attn_layer_idx=attn_layer_idx,
                    # attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        # 构造SSM输入相关参数B,C,Δ的线性投影层
        self.bcdt_proj_outdim = 0
        self.bcdt_dim_list = []
        for i in range(n_layer):
            b_dim = c_dim = self.layers[i].mixer.ngroups * self.layers[i].mixer.d_state
            dt_dim = self.layers[i].mixer.nheads
            self.bcdt_proj_outdim += (b_dim + c_dim + dt_dim)
            self.bcdt_dim_list.extend([b_dim, c_dim, dt_dim])
        self.bcdt_proj = nn.Linear(aux_feature_size, self.bcdt_proj_outdim, bias=bias, **factory_kwargs)

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, aux_features=None, inference_params=None, **mixer_kwargs):
        # hidden_states = self.embedding(input_ids)
        if aux_features is not None:
            all_bcdt = self.bcdt_proj(aux_features) # 一次性生成所有block的SSM输入相关参数B,C,Δ
            all_bcdt_tuple = torch.split(all_bcdt, self.bcdt_dim_list, dim=-1)
        else:
            all_bcdt_tuple = (None,) * len(self.bcdt_dim_list)

        residual = None
        for i, layer in enumerate(self.layers):
            B, C, dt =all_bcdt_tuple[3*i], all_bcdt_tuple[3*i+1], all_bcdt_tuple[3*i+2]
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, B=B, C=C, dt=dt 
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


@dataclass
class MambaConfig:

    d_model: int = 128
    d_intermediate: int = 0
    n_layer: int = 4
    feature_size: int = 2, # vocab_size: int = 50277
    aux_feature_size: int = 0,
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    # pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

# """ 无大的改动，仅因block构造的新增部分而增加了相应的传入参数 """
# class MambaLMHeadModel(nn.Module, GenerationMixin):

#     def __init__(
#         self,
#         config: MambaConfig,
#         initializer_cfg=None,
#         device=None,
#         dtype=None,
#     ) -> None:
#         self.config = config
#         d_model = config.d_model
#         n_layer = config.n_layer
#         d_intermediate = config.d_intermediate
#         vocab_size = config.vocab_size
#         ssm_cfg = config.ssm_cfg
#         attn_layer_idx = config.attn_layer_idx
#         attn_cfg = config.attn_cfg
#         rms_norm = config.rms_norm
#         residual_in_fp32 = config.residual_in_fp32
#         fused_add_norm = config.fused_add_norm
#         pad_vocab_size_multiple = config.pad_vocab_size_multiple
#         factory_kwargs = {"device": device, "dtype": dtype}

#         super().__init__()
#         if vocab_size % pad_vocab_size_multiple != 0:
#             vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
#         self.backbone = MixerModel(
#             d_model=d_model,
#             n_layer=n_layer,
#             d_intermediate=d_intermediate,
#             vocab_size=vocab_size,
#             ssm_cfg=ssm_cfg,
#             attn_layer_idx=attn_layer_idx,
#             attn_cfg=attn_cfg,
#             rms_norm=rms_norm,
#             initializer_cfg=initializer_cfg,
#             fused_add_norm=fused_add_norm,
#             residual_in_fp32=residual_in_fp32,
#             **factory_kwargs,
#         )
#         self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

#         # Initialize weights and apply final processing
#         self.apply(
#             partial(
#                 _init_weights,
#                 n_layer=n_layer,
#                 **(initializer_cfg if initializer_cfg is not None else {}),
#             )
#         )
#         self.tie_weights()

#     def tie_weights(self):
#         if self.config.tie_embeddings:
#             self.lm_head.weight = self.backbone.embedding.weight

#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

#     def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
#         """
#         "position_ids" is just to be compatible with Transformer generation. We don't use it.
#         num_last_tokens: if > 0, only return the logits for the last n tokens
#         """
#         hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
#         if num_last_tokens > 0:
#             hidden_states = hidden_states[:, -num_last_tokens:]
#         lm_logits = self.lm_head(hidden_states)
#         CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
#         return CausalLMOutput(logits=lm_logits)

#     @classmethod
#     def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
#         config_data = load_config_hf(pretrained_model_name)
#         config = MambaConfig(**config_data)
#         model = cls(config, device=device, dtype=dtype, **kwargs)
#         model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
#         return model

#     def save_pretrained(self, save_directory):
#         """
#         Minimal implementation of save_pretrained for MambaLMHeadModel.
#         Save the model and its configuration file to a directory.
#         """
#         # Ensure save_directory exists
#         os.makedirs(save_directory, exist_ok=True)

#         # Save the model's state_dict
#         model_path = os.path.join(save_directory, 'pytorch_model.bin')
#         torch.save(self.state_dict(), model_path)

#         # Save the configuration of the model
#         config_path = os.path.join(save_directory, 'config.json')
#         with open(config_path, 'w') as f:
#             json.dump(self.config.__dict__, f, indent=4)
