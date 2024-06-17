# import sys
# sys.path.append('..')

import math
from functools import partial
import json
import os
import gc
import random
import numpy as np
from collections import namedtuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from utils import DotDict
# from data import TARGET_SAMPLE_RATE,TRIP_COLS
from .traj_mambaBlock import Mamba, Block
# from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.utils.generation import InferenceParams, DecodingCGCache
# from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class MambaConfig(DotDict):
    def __init__(self,
            d_model: int = 128,
            n_layer: int = 4, 
            feature_size: int = 2, # vocab_size: int = 50277
            aux_feature_size: int = 0,
            # num_roads: int = None,
            # road_embedding_size: int = None,
            ssm_cfg: dict = {},
            rms_norm: bool = True,
            residual_in_fp32: bool = True,
            fused_add_norm: bool = True,
            # pad_vocab_size_multiple: int = 1 # 8
        ):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.feature_size = feature_size
        self.aux_feature_size = aux_feature_size
        # self.num_roads = num_roads
        # self.road_embedding_size = road_embedding_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # self.pad_vocab_size_multiple = pad_vocab_size_multiple


def create_block(
    d_model, # 模型的隐藏层维度
    aux_feature_size=0,
    # num_roads=None,
    # road_embedding_size=None,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,  # 是否使用RMSNorm
    residual_in_fp32=False,
    fused_add_norm=False, # 是否融合add + layer_norm
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    
    '''partial 的功能：把一个函数/类的某些参数给固定住，返回一个新的函数/类'''
    # Block的核心组件
    mixer_cls = partial(Mamba, aux_feature_size=aux_feature_size, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs) # num_roads=num_roads, road_embedding_size=road_embedding_size, 
    # 归一化模块，用于归一化操作
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    
    # Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class TrajMixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        # feature_size: int,
        aux_feature_size: int,
        # num_roads: int,
        # road_embedding_size: int,
        ssm_cfg={},
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
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
            if layer_norm_fn is None or rms_norm_fn is None: # 两个都要 not None !
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    aux_feature_size = aux_feature_size,
                    # num_roads = num_roads,
                    # road_embedding_size = road_embedding_size,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i, # 0 ~ n_layer-1 顺次
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) # class Block的allocate_inference_cache函数 
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, aux_features=None, inference_params=None): # input_ids -> hidden_states
        residual = None # initialize
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, aux_features=aux_features, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states # add
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype)) # layer_norm
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


# class TrajMambaModel(nn.Module):

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
#         feature_size = config.feature_size
#         aux_feature_size = config.aux_feature_size
#         # num_roads = config.num_roads
#         # road_embedding_size = config.road_embedding_size
#         ssm_cfg = config.ssm_cfg
#         rms_norm = config.rms_norm
#         residual_in_fp32 = config.residual_in_fp32
#         fused_add_norm = config.fused_add_norm
#         # pad_vocab_size_multiple = config.pad_vocab_size_multiple
#         self.factory_kwargs = {"device": device, "dtype": dtype}

#         super().__init__()
#         # if feature_size % pad_vocab_size_multiple != 0:
#         #     feature_size += pad_vocab_size_multiple - (feature_size % pad_vocab_size_multiple) # 确保vocab_size可以整除pad_vocab_size_multiple
#         self.backbone = TrajMixerModel(
#             d_model=d_model,
#             n_layer=n_layer,
#             feature_size=feature_size,
#             aux_feature_size = aux_feature_size,
#             # num_roads = num_roads,
#             # road_embedding_size = road_embedding_size,
#             ssm_cfg=ssm_cfg,
#             rms_norm=rms_norm,
#             initializer_cfg=initializer_cfg,
#             fused_add_norm=fused_add_norm,
#             residual_in_fp32=residual_in_fp32,
#             **self.factory_kwargs,
#         )
#         self.output_linear = nn.Linear(d_model, feature_size, **self.factory_kwargs) # output layer # 取消了bias=False
#         self.final_act = nn.ReLU() # nn.LeakyReLU()

#         # Initialize weights and apply final processing
#         self.apply(
#             partial(
#                 _init_weights,
#                 n_layer=n_layer,
#                 **(initializer_cfg if initializer_cfg is not None else {}),
#             )
#         )
#         # self.tie_weights()

#     def tie_weights(self):
#         self.output_linear.weight = nn.Parameter(self.backbone.linear_embedding.weight.t())

#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

#     def forward(self, input_ids, position_ids=None, aux_features=None, inference_params=None, num_last_tokens=0, **kwargs):
#         """
#         "position_ids" is just to be compatible with Transformer generation. We don't use it.
#         num_last_tokens: if > 0, only return the logits for the last n tokens
#         """
#         hidden_states = self.backbone(input_ids, aux_features=aux_features, inference_params=inference_params)
#         if num_last_tokens > 0:
#             hidden_states = hidden_states[:, -num_last_tokens:] # 只取最后n个tokens的hidden_states
#         traj_logits = self.output_linear(self.final_act(hidden_states))
#         # CausalTrajOutput = namedtuple("CausalTrajOutput", ["logits"])
#         return traj_logits
    
#     @torch.inference_mode()
#     def decode(
#         self,
#         input_ids,
#         predict_len,
#         feat_names,
#         aux_feature_names=[],
#         aux_features=None,
#         scaler_list=[],
#         teacher_outputs=None,
#         teacher_aux_features=None,
#         teacher_forcing_ratio = 0.3,
#         cg=True, # whether using graph_cache——有问题！
#         **kwargs
#         ):
#         """
#         We assume that all sequences in the same batch have the same length.

#         Arguments:
#             input_ids: (batch, seq_len, feature_dim)
#             predict_len: int
#             teacher_outputs (optional): (batch, max_length-seq_len)
#         Returns: 
#             predicts: (batch, max_length-seq_len, feature_dim)
#         """

#         batch_size, seqlen_og, feature_size = input_ids.shape
#         aux_feature_size = len(aux_feature_names)
#         lng_col = feat_names.index("lng") # 0
#         lat_col = feat_names.index("lat") # 1
#         if aux_feature_names:
#             speed_col = aux_feature_names.index("speed") # 0
        
#         teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
        
#         # inference_params = InferenceParams(max_seqlen=seqlen_og+predict_len, max_batch_size=batch_size)
#         max_length = seqlen_og + predict_len
#         if cg:
#             if not hasattr(self, "_decoding_cache"):
#                 self._decoding_cache = None
#             self._decoding_cache = update_graph_cache(
#                 self,
#                 self._decoding_cache,
#                 batch_size,
#                 seqlen_og,
#                 max_length,
#                 feature_size,
#                 aux_feature_size,
#             )
#             inference_params = self._decoding_cache.inference_params
#             inference_params.reset(max_length, batch_size)
#         else:
#             inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

#         def get_logits(input_ids, inference_params, aux_features=None):
#             position_ids = None
#             decoding = inference_params.seqlen_offset > 0
#             if decoding:
#                 position_ids = torch.full(
#                     (batch_size, 1),
#                     inference_params.seqlen_offset,
#                     dtype=torch.long,
#                     device=input_ids.device,
#                 )
#             if not cg or not decoding:
#                 logits = self.forward(
#                     input_ids,
#                     position_ids=position_ids,
#                     aux_features=aux_features,
#                     inference_params=inference_params,
#                     num_last_tokens=1,
#                 )
#             else:
#                 logits = self._decoding_cache.run(
#                     input_ids, position_ids, aux_features, inference_params.seqlen_offset
#                 )
#             return logits
        
#         def get_aux_features(predict, last_trajpoint, last_speeds):
#             Predict = scaler_list[0](feat_names, predict).squeeze(1)#.cpu().detach().numpy() # (B, feature_dim) 当前推理出的轨迹点预测特征
#             Last_trajpoint = scaler_list[0](feat_names, last_trajpoint)#.cpu().detach().numpy() # (B, feature_dim) 上一个轨迹点预测特征
            
#             dists = cal_tensor_geo_distance(Last_trajpoint[:, lng_col], Last_trajpoint[:, lat_col], Predict[:, lng_col], Predict[:, lat_col])
#             speeds = dists / TARGET_SAMPLE_RATE # (B)
            
#             Last_speeds = scaler_list[1].inverse_transform(last_speeds)#.cpu().detach().numpy() # (B)
#             speed_diff = speeds - Last_speeds
#             accs = speed_diff / TARGET_SAMPLE_RATE # (B)
            
#             courseAngles = cal_tensor_courseAngle(Last_trajpoint[:, lng_col], Last_trajpoint[:, lat_col], Predict[:, lng_col], Predict[:, lat_col])
            
#             speeds = scaler_list[1].transform(speeds).reshape(-1,1) # (B,1)
#             accs = scaler_list[2].transform(accs).reshape(-1,1)
#             courseAngles = scaler_list[3].transform(courseAngles).reshape(-1,1)
            
#             predict_aux_features = torch.stack([speeds,accs,courseAngles],axis=-1) # (B,1,axu_feature_dim)
#             return predict_aux_features
        
#         sequences = [input_ids]
#         aux_features_seq = [aux_features]
#         # 模型输入为L步历史轨迹序列，预测第1个未来轨迹点
#         predict = get_logits(sequences[-1], inference_params, aux_features_seq[-1])
#         inference_params.seqlen_offset += sequences[-1].shape[1]
#         # 模型输入为1步未来轨迹，预测第i+1个未来轨迹点
#         for i in range(1, predict_len):
#             if i-1 < teacher_output_len and teacher_forcing_ratio > random.random():
#                 predict = teacher_outputs[:,i-1]
#                 predict_aux_features = teacher_aux_features[:,i-1] if aux_feature_size else None
#             elif aux_feature_size:
#                 # 用推理出的预测值计算高阶特征
#                 predict_aux_features = get_aux_features(predict, last_trajpoint=sequences[-1][:, -1], last_speeds=aux_features_seq[-1][:, -1, speed_col])
#             else:
#                 predict_aux_features = None
#             sequences.append(predict)
#             aux_features_seq.append(predict_aux_features)

#             predict = get_logits(sequences[-1], inference_params, aux_features_seq[-1])
#             inference_params.seqlen_offset += sequences[-1].shape[1]
#         sequences.append(predict)  
        
#         return torch.cat(sequences[1:], dim=1)
#     '''
#     @torch.inference_mode()
#     def decode(
#         self,
#         input_ids,
#         predict_len,
#         teacher_outputs=None,
#         teacher_forcing_ratio = 0.3,
#         cg=False, # whether using graph_cache——有问题，不能用
#         **kwargs
#         ):
#         """
#         We assume that all sequences in the same batch have the same length.

#         Arguments:
#             input_ids: (batch, seq_len, feature_dim)
#             predict_len: int
#             teacher_outputs (optional): (batch, predict_len, feature_dim)
#         Returns: 
#             predicts: (batch, predict_len, feature_dim)
#         """

#         batch_size, seqlen_og, feature_size = input_ids.shape
#         max_length = seqlen_og + predict_len
#         teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
#         if cg:
#             if not hasattr(self, "_decoding_cache"):
#                 self._decoding_cache = None
#             self._decoding_cache = update_graph_cache(
#                 self,
#                 self._decoding_cache,
#                 batch_size,
#                 seqlen_og,
#                 max_length,
#                 feature_size,
#             )
#             inference_params = self._decoding_cache.inference_params
#             inference_params.reset(max_length, batch_size)
#         else:
#             inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

#         def get_logits(input_ids, inference_params):
#             decoding = inference_params.seqlen_offset > 0
#             if decoding:
#                 position_ids = torch.full(
#                     (batch_size, 1),
#                     inference_params.seqlen_offset,
#                     dtype=torch.long,
#                     device=input_ids.device,
#                 )
#             else:
#                 position_ids = None
#             if not cg or not decoding:
#                 logits = self.forward(
#                     input_ids,
#                     position_ids=position_ids,
#                     inference_params=inference_params,
#                     num_last_tokens=1,
#                 )
#             else:
#                 logits = self._decoding_cache.run(
#                     input_ids, position_ids, inference_params.seqlen_offset
#                 )
#             return logits

#         sequences = [input_ids]
#         for i in range(predict_len):
#             predict = get_logits(sequences[-1], inference_params)
#             inference_params.seqlen_offset += sequences[-1].shape[1]
#             if teacher_forcing_ratio > random.random() and i < teacher_output_len:
#                 predict = teacher_outputs[:,i]
#             sequences.append(predict)

#         return torch.cat(sequences[1:], dim=1)
#     '''

#     @classmethod
#     def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs): # 由预训练模型名实例化出相应的与训练模型并返回
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
#         if not os.path.exists(save_directory):
#             os.makedirs(save_directory)

#         # Save the model's state_dict
#         model_path = os.path.join(save_directory, 'pytorch_model.bin')
#         torch.save(self.state_dict(), model_path)

#         # Save the configuration of the model
#         config_path = os.path.join(save_directory, 'config.json')
#         with open(config_path, 'w') as f:
#             json.dump(self.config.__dict__, f)



@torch.inference_mode()
def update_graph_cache(
    model,
    cache,
    batch_size,
    seqlen_og,
    max_seqlen,
    feature_size,
    aux_feature_size,
    decoding_seqlens=(1,),
    dtype=None,
    n_warmups=2,
):
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if (
        (device, dtype) != (cache.device, cache.dtype)
        or batch_size > cache.max_batch_size
        or max_seqlen > cache.max_seqlen
    ):  # Invalidate the cache
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        assert hasattr(model, "allocate_inference_cache"), "CUDA graph decoding requires that the model has a method allocate_inference_cache"
        inf_cache = model.allocate_inference_cache(batch_size, max_seqlen, dtype)
        lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
        cache.inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            seqlen_offset=seqlen_og,
            key_value_memory_dict=inf_cache,
            lengths_per_sample=lengths_per_sample,
        )
        cache.mempool = torch.cuda.graphs.graph_pool_handle() # (0,1)
    for decoding_seqlen in decoding_seqlens:
        if (batch_size, decoding_seqlen) not in cache.callables:
            cache.callables[batch_size, decoding_seqlen] = capture_graph(
                model,
                cache.inference_params,
                batch_size,
                max_seqlen,
                decoding_feanum=feature_size,
                decoding_auxfeanum=aux_feature_size,
                decoding_seqlen=decoding_seqlen,
                mempool=cache.mempool,
                n_warmups=n_warmups,
            )

    def dispatch(input_ids, position_ids, aux_features, seqlen):
        batch_size, decoding_seqlen = input_ids.shape[:2]
        return cache.callables[batch_size, decoding_seqlen](input_ids, position_ids, aux_features, seqlen)

    cache.run = dispatch
    cache.inference_params.seqlen_offset = 0  # Reset so it's not confusing
    return cache


def capture_graph(
    model, inference_params, batch_size, max_seqlen, decoding_feanum=1, decoding_auxfeanum=1, decoding_seqlen=1, mempool=None, n_warmups=2
):
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, decoding_seqlen, decoding_feanum), 0, dtype=torch.float32, device=device)
    position_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    if decoding_auxfeanum:
        aux_features = torch.full((batch_size, decoding_seqlen, decoding_auxfeanum), 0, dtype=torch.float32, device=device)
    seqlen_offset_og = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    inference_params.lengths_per_sample[:] = inference_params.seqlen_offset

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model(
                input_ids,
                position_ids=position_ids,
                aux_features=aux_features if decoding_auxfeanum else None,
                inference_params=inference_params,
                num_last_tokens=decoding_seqlen,
            )
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        # 无法看到input_ids, position_ids=position_ids, aux_features=aux_features, inference_params=inference_params,
        logits = model(
            input_ids,
            position_ids=position_ids,
            aux_features=aux_features if decoding_auxfeanum else None,
            inference_params=inference_params,
            num_last_tokens=decoding_seqlen,
        )

    def run(new_input_ids, new_position_ids, new_aux_features, seqlen):
        inference_params.lengths_per_sample[:] = seqlen
        input_ids.copy_(new_input_ids)
        position_ids.copy_(new_position_ids)
        if new_aux_features is not None:
            aux_features.copy_(new_aux_features)
        graph.replay()
        return logits.clone()

    inference_params.seqlen_offset = seqlen_offset_og
    return run