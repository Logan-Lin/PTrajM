import numpy as np
import torch
from torch import nn
from einops import repeat
import time

from .encode import PositionalEmbedding, FourierEncode
from data import COL_I
from utils import cal_tensor_geo_distance, cal_tensor_courseAngle
from .mamba.mamba import TrajMixerModel
from .mamba2.mamba2 import TrajMixerModel2


class TrajClip(nn.Module):
    def __init__(self, embed_size, d_model, road_embed, poi_embed, poi_coors, spatial_border,
                 device, road_weight=1, poi_weight=1, use_higher_features=True, use_mamba2=True, n_layer=4, d_state=128, headdim=64, d_inner=0):
        """The core model of Trajectory CLIP.

        Args:
            embed_size (int): dimension of learnable embedding modules.
            d_model (int): dimension of the sequential models.
            road_embed (np.array): pre-defined embedding matrix of roads, with shape (n_roads, E).
            poi_embed (np.array): pre-defined embedding matrix of POIs, with shape (n_pois, E).
            poi_coors (np.array): coordiantes of all POIs, with shape (n_pois, 2).
            spatial_border (list): coordinates indicating the spatial border: [[x_min, y_min], [x_max, y_max]].
            road_weight (int, optional): loss weight of road view. Defaults to 1.
            poi_weight (int, optional): loss weight of poi view. Defaults to 1.
            use_higher_features (bool, optional): whether to use trajectory's higher-order features. Defaults to True.
            use_mamba2 (bool, optional): whether to use Mamba2 architecture. Defaults to True.
            n_layer (int, optional): number of stacked Traj-Mamba Blocks. Defaults to 4.
            d_state (int, optional): state size of Traj-SSM in Traj-Mamba Block. Defaults to 128.
            headdim (int, optional): head dimension of Traj-SSM in Traj-Mamba Block. Defaults to 64.
            d_inner (int, optional): inner model dimension of Traj-Mamba Blocks. If setting to 0 means d_inner=2*d_model.
        """

        super().__init__()

        self.poi_coors = nn.Parameter(torch.from_numpy(poi_coors).float(), requires_grad=False) # n_pois：12439[chengdu]
        self.spatial_border = nn.Parameter(torch.tensor(spatial_border), requires_grad=False)
        self.road_weight = road_weight
        self.poi_weight = poi_weight
        self.use_higher_features = use_higher_features

        self.pos_encode_layer = PositionalEmbedding(d_model)

        self.traj_view = nn.ModuleDict({
            'spatial_embed_layer': nn.Sequential(nn.Linear(2, embed_size), nn.LeakyReLU(), nn.Linear(embed_size, d_model)),
            'temporal_embed_modules': nn.ModuleList([FourierEncode(embed_size) for _ in range(4)]),
            'temporal_embed_layer': nn.Sequential(nn.LeakyReLU(), nn.Linear(embed_size * 4, d_model)),
            'seq_encoder': 
            TrajMixerModel2(d_model=d_model, n_layer=n_layer, d_intermediate=0, aux_feature_size=3 if self.use_higher_features else 0,
                            d_state=d_state, headdim=headdim, d_inner=d_inner, device=device, dtype=torch.float32) if use_mamba2 else \
            TrajMixerModel(d_model=d_model, n_layer=n_layer, aux_feature_size=3 if self.use_higher_features else 0,
                                          device=device, dtype=torch.float32)
            # nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, batch_first=True),
            #                                      num_layers=2)
        })
        print("model use Mamba2 block?",isinstance(self.traj_view['seq_encoder'], TrajMixerModel2))

        road_embed_mat = nn.Embedding(*road_embed.shape)
        road_embed_mat.weight = nn.Parameter(torch.from_numpy(road_embed).float(), requires_grad=False)
        self.road_view = nn.ModuleDict({
            'text_embed_mat': road_embed_mat,
            'text_embed_layer': nn.Sequential(nn.LayerNorm(road_embed.shape[1]),
                                              nn.Linear(road_embed.shape[1], d_model)),
            'index_embed_layer': nn.Sequential(nn.Embedding(road_embed.shape[0], embed_size),
                                               nn.LayerNorm(embed_size),
                                               nn.Linear(embed_size, d_model)),
            'seq_encoder': nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, batch_first=True),
                                                 num_layers=2)
        })

        poi_embed_mat = nn.Embedding(*poi_embed.shape)
        poi_embed_mat.weight = nn.Parameter(torch.from_numpy(poi_embed).float(), requires_grad=False)
        self.poi_view = nn.ModuleDict({
            'text_embed_mat': poi_embed_mat,
            'text_embed_layer': nn.Sequential(nn.LayerNorm(poi_embed.shape[1]),
                                              nn.Linear(poi_embed.shape[1], d_model)),
            'index_embed_layer': nn.Sequential(nn.Embedding(poi_embed.shape[0], embed_size),
                                               nn.LayerNorm(embed_size),
                                               nn.Linear(embed_size, d_model)),
            'seq_encoder': nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, batch_first=True),
                                                 num_layers=2)
        })

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_entropy = nn.CrossEntropyLoss()

    def cal_traj_h(self, norm_spatial, temporal, aux_features, valid_lens):
        """Calculate trajectories' embedding vectors given their spatio-temporal features.
        The detailed definition of trajectories' spatial and temporal features can be refered to `data.py`.

        Args:
            norm_patial (FloatTensor): trajectories' spatial features, with shape (B, L, F_s).
            temporal (FloatTensor): trajectories' temporal features, with shape (B, L, F_t).
            valid_lens (LongTensor): valid lengths of trajectories in this batch.
            aux_features (FloatTensor): trajectories' high-order features, with shape (B, L, F_h=3) or None.

        Returns:
            FloatTensor: the embedding vectors of this batch of trajectories, with shape (B, E).
        """
        B, L = norm_spatial.size(0), norm_spatial.size(1)
        positions = repeat(torch.arange(L), 'L -> B L', B=B).to(valid_lens.device)
        batch_mask = get_batch_mask(B, L, valid_lens)

        spatial_e = self.traj_view['spatial_embed_layer'](norm_spatial)  # (B, L, E)

        temporal_e = self.traj_view['temporal_embed_layer'](
            torch.cat([self.traj_view['temporal_embed_modules'][i](temp_token)
                       for i, temp_token in enumerate(tokenize_timestamp(temporal))], -1)
        )  # (B, L, E)
        # Temporal values lower than 0 stands for feature mask.
        temporal_e = temporal_e.masked_fill(temporal[..., :1] < 0, 0)

        pos_encoding = self.pos_encode_layer(positions)
        traj_h = spatial_e + temporal_e + pos_encoding
        # traj_h = self.traj_view['seq_encoder'](traj_h, src_key_padding_mask=batch_mask)
        traj_h = self.traj_view['seq_encoder'](traj_h, aux_features)
        traj_h = traj_h.masked_fill(batch_mask.unsqueeze(-1), 0).sum(1) / repeat(valid_lens, 'B -> B 1')

        return traj_h

    def forward(self, input_seq, valid_lens):
        """
        Args:
            input_seq (FloatTensor): batch of trajectory features, with shape (B, L, F).
            valid_lens (LongTensor): valid lengths of trajectories in this batch.

        Returns:
            Tensor: embedding vectors for this batch of trajectories, with shape (B, E).
        """
        spatial = input_seq[:, :, COL_I['spatial']]  # (B, L, 2)
        temporal = input_seq[:, :, COL_I['temporal']]  # (B, L, 2)
        aux_features = self.cal_high_order_features(spatial, temporal, valid_lens)
        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        traj_h = self.cal_traj_h(norm_spatial, temporal, aux_features, valid_lens)

        return traj_h

    def loss(self, input_seq, valid_lens):
        """Calcualte the pre-training loss.

        Args:
            Same as the forward function.

        Returns:
            FloatTensor: the pre-training loss value of this batch.
        """
        B, L, _ = input_seq.shape
        positions = repeat(torch.arange(L), 'L -> B L', B=B).to(input_seq.device)
        batch_mask = get_batch_mask(B, L, valid_lens)
        pos_encoding = self.pos_encode_layer(positions) # (B,L,D)

        # Trajectory (spatio-temporal) view.
        spatial = input_seq[:, :, COL_I['spatial']]  # (B, L, 2)
        temporal = input_seq[:, :, COL_I['temporal']]  # (B, L, 2)
        aux_features = self.cal_high_order_features(spatial, temporal, valid_lens)
        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        traj_h = self.cal_traj_h(norm_spatial, temporal, aux_features, valid_lens)

        # Road view.
        road = input_seq[:, :, COL_I['road']].long() # (B,L)
        road_index_e = self.road_view['index_embed_layer'](road) # (B,L,D)
        road_text_e = self.road_view['text_embed_layer'](self.road_view['text_embed_mat'](road)) # (B,L,D)
        road_h = road_index_e + road_text_e + pos_encoding
        road_h = self.road_view['seq_encoder'](road_h, src_key_padding_mask=batch_mask)
        road_h = road_h.masked_fill(batch_mask.unsqueeze(-1), 0).sum(1) / valid_lens.unsqueeze(-1)

        # POI view.
        poi = ((self.poi_coors.unsqueeze(0).unsqueeze(0) -
                spatial.unsqueeze(2)) ** 2).sum(-1).argmin(dim=-1) # (B,L)
        poi_index_e = self.poi_view['index_embed_layer'](poi)
        poi_text_e = self.poi_view['text_embed_layer'](self.poi_view['text_embed_mat'](poi))
        poi_h = poi_index_e + poi_text_e + pos_encoding
        poi_h = self.poi_view['seq_encoder'](poi_h, src_key_padding_mask=batch_mask)
        poi_h = poi_h.masked_fill(batch_mask.unsqueeze(-1), 0).sum(1) / valid_lens.unsqueeze(-1)

        # CLIP loss.
        traj_h = traj_h / traj_h.norm(dim=1, keepdim=True)
        road_h = road_h / road_h.norm(dim=1, keepdim=True)
        poi_h = poi_h / poi_h.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logit_road = logit_scale * traj_h @ road_h.t()
        logit_poi = logit_scale * traj_h @ poi_h.t()

        label = torch.arange(B).long().to(input_seq.device)
        loss_road = (self.cross_entropy(logit_road, label) + self.cross_entropy(logit_road.t(), label)) / 2
        loss_poi = (self.cross_entropy(logit_poi, label) + self.cross_entropy(logit_poi.t(), label)) / 2
        loss = self.road_weight * loss_road + self.poi_weight * loss_poi
        return loss
    
    def cal_high_order_features(self, spatial, temporal, valid_lens):
        '''Calculate trajectories' high-order features, including speed, acceleration, and movement angle.

        Args:
            spatial (FloatTensor): trajectories' spatial features, with shape (B, L, F_s).
            temporal (FloatTensor): trajectories' temporal features, with shape (B, L, F_t).
            valid_lens (LongTensor): valid lengths of trajectories in this batch.
        
        Returns:
            FloatTensor: the three high-order features of this batch of trajectories, with shape (B, L, 3).
        '''
        
        B, L = spatial.size(0), spatial.size(1)
        batch_mask = get_batch_mask(B, L, valid_lens)

        dists = cal_tensor_geo_distance(spatial[:, :-1, 0], spatial[:, :-1, 1], spatial[:, 1:, 0], spatial[:, 1:, 1])
        time_diff = temporal[:, 1:, 1] - temporal[:, :-1, 1]
        time_diff = time_diff.masked_fill(time_diff == 0, 1)
        speeds = torch.div(dists, time_diff) # (B,L-1)
        speeds = torch.concatenate([speeds[:,:1], speeds],dim=-1) # 将轨迹起始速度v0置为求出的第一个速度
        speeds = speeds.masked_fill(batch_mask, 0) # need to do masked_fill for padding trajectories!

        speed_diff = speeds[:, 1:] - speeds[:, :-1]
        accs = torch.div(speed_diff, time_diff) # (B,L-1)
        accs = torch.concatenate([accs[:,:1], accs],dim=-1)

        courseAngles = cal_tensor_courseAngle(spatial[:, :-1, 0], spatial[:, :-1, 1], spatial[:, 1:, 0], spatial[:, 1:, 1])
        courseAngles = torch.concatenate([courseAngles[:,:1], courseAngles],dim=-1)
        courseAngles = courseAngles.masked_fill(batch_mask, 0) # need to do masked_fill for padding trajectories!

        norm_speeds = (speeds - speeds.min()) / (speeds.max() - speeds.min())
        norm_accs = (accs - accs.min()) / (accs.max() - accs.min())
        norm_courseAngles = (courseAngles - courseAngles.min()) / (courseAngles.max() - courseAngles.min())
        aux_features = torch.stack([norm_speeds,norm_accs,norm_courseAngles],axis=-1)
        
        return aux_features
    
    def forward_on_search_mode(self, input_seq, valid_lens):
        '''Calculate trajectories' embedding vectors given their spatio-temporal features and get embedding time of model.
        '''
        s_time = time.time()
        spatial = input_seq[:, :, COL_I['spatial']]  # (B, L, 2)
        temporal = input_seq[:, :, COL_I['temporal']]  # (B, L, 2)
        aux_features = self.cal_high_order_features(spatial, temporal, valid_lens)
        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        e_time = time.time()
        
        start_time = time.time()
        traj_h = self.cal_traj_h(norm_spatial, temporal, aux_features, valid_lens)
        end_time = time.time()

        return traj_h, (end_time - start_time), (e_time-s_time)


def gen_causal_mask(seq_len, include_self=True):
    """
    Generate a casual mask which prevents i-th output element from
    depending on any input elements from "the future".
    Note that for PyTorch Transformer model, sequence mask should be
    filled with -inf for the masked positions, and 0.0 else.

    :param seq_len: length of sequence.
    :return: a casual mask, shape (seq_len, seq_len)
    """
    if include_self:
        mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
    else:
        mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
    return mask.bool()


def get_batch_mask(B, L, valid_len):
    mask = repeat(torch.arange(end=L, device=valid_len.device),
                  'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
    return mask


def tokenize_timestamp(t):
    week = t[..., 0] % (7 * 24 * 60 * 60) / (24 * 60 * 60)
    hour = t[..., 0] % (24 * 60 * 60) / (60 * 60)
    minute = t[..., 0] % (60 * 60) / 60
    d_minute = t[..., 1] / 60
    return week, hour, minute, d_minute


def geo_distance(a_coor, b_coor):
    a_coor, b_coor = torch.deg2rad(a_coor), torch.deg2rad(b_coor)
    a_x, a_y = a_coor[..., 0], a_coor[..., 1]
    b_x, b_y = b_coor[..., 0], b_coor[..., 1]
    d_x = a_x - b_x
    d_y = a_y - b_y

    a = torch.sin(d_y / 2) ** 2 + torch.cos(a_y) * torch.cos(b_y) * torch.sin(d_x / 2) ** 2
    distance = 2 * torch.arcsin(torch.sqrt(a)) * 6371 * 1000
    return distance
