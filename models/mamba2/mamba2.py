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

from .traj_mambaBlock2 import TrajMamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.models.mixer_seq_simple import _init_weights

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

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
        mixer_cls = partial(TrajMamba2,
                            d_inner=d_inner,
                            d_state=d_state,
                            headdim=headdim,
                            aux_feature_size=aux_feature_size,
                            layer_idx=layer_idx,
                            **ssm_cfg,
                            **factory_kwargs
                            )
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


class TrajMixerModel2(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        aux_feature_size: int,
        d_state: int = 128, 
        headdim: int = 64,
        d_inner: int = 0, 
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        bias=False, # 其他层（如线性层）是否使用偏置项（与class Mamba2的相应参数的默认值保持一致）
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

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
        print(f"param check:")
        print(f"d_model={self.layers[0].mixer.d_model}, d_inner={self.layers[0].mixer.d_inner}")
        print(f"d_state={self.layers[0].mixer.d_state}, headdim={self.layers[0].mixer.headdim}")

        # 构造SSM输入相关参数B,C,Δ的线性投影层
        if aux_feature_size:
            self.bcdt_proj_outdim = 0
            self.bcdt_dim_list = []
            for i in range(n_layer):
                b_dim = c_dim = self.layers[i].mixer.ngroups * self.layers[i].mixer.d_state
                dt_dim = self.layers[i].mixer.nheads
                self.bcdt_proj_outdim += (b_dim + c_dim + dt_dim)
                self.bcdt_dim_list.extend([b_dim, c_dim, dt_dim])
            self.bcdt_proj = nn.Linear(aux_feature_size, self.bcdt_proj_outdim, bias=bias, **factory_kwargs)
        else:
            self.bcdt_proj = None
            self.num_bcdt = n_layer * 3
            print("Don't use trajectory's higher-order features")

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

    def forward(self, hidden_states, aux_features, inference_params=None, **mixer_kwargs):
        if self.bcdt_proj is not None:
            all_bcdt = self.bcdt_proj(aux_features) # 一次性生成所有block的SSM输入相关参数B,C,Δ
            all_bcdt_tuple = torch.split(all_bcdt, self.bcdt_dim_list, dim=-1)
        else:
            all_bcdt_tuple = (None,) * self.num_bcdt

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