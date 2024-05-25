import numpy as np
import torch
from torch import nn
from einops import repeat

from .encode import PositionalEmbedding, FourierEncode
from data import COL_I


class TrajClip(nn.Module):
    def __init__(self, embed_size, d_model, road_embed, poi_embed, poi_coors, spatial_border,
                 road_weight=1, poi_weight=1):
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
        """

        super().__init__()

        self.poi_coors = nn.Parameter(torch.from_numpy(poi_coors).float(), requires_grad=False)
        self.spatial_border = nn.Parameter(torch.tensor(spatial_border), requires_grad=False)
        self.road_weight = road_weight
        self.poi_weight = poi_weight

        self.pos_encode_layer = PositionalEmbedding(d_model)

        self.traj_view = nn.ModuleDict({
            'spatial_embed_layer': nn.Sequential(nn.Linear(2, embed_size), nn.LeakyReLU(), nn.Linear(embed_size, d_model)),
            'temporal_embed_modules': nn.ModuleList([FourierEncode(embed_size) for _ in range(4)]),
            'temporal_embed_layer': nn.Sequential(nn.LeakyReLU(), nn.Linear(embed_size * 4, d_model)),
            'seq_encoder': nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, batch_first=True),
                                                 num_layers=2)
        })

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

    def cal_traj_h(self, spatial, temporal, valid_lens):
        """Calculate trajectories' embedding vectors given their spatio-temporal features.
        The detailed definition of trajectories' spatial and temporal features can be refered to `data.py`.

        Args:
            spatial (FloatTensor): trajectories' spatial features, with shape (B, L, F_s).
            temporal (FloatTensor): trajectories' temporal features, with shape (B, L, F_t).
            valid_lens (LongTensor): valid lengths of trajectories in this batch.

        Returns:
            FloatTensor: the embedding vectors of this batch of trajectories, with shape (B, E).
        """
        B, L = spatial.size(0), spatial.size(1)
        positions = repeat(torch.arange(L), 'L -> B L', B=B)

        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        spatial_e = self.traj_view['spatial_embed_layer'](norm_spatial)  # (B, L, E)

        temporal_e = self.traj_view['temporal_embed_layer'](
            torch.cat([self.traj_view['temporal_embed_modules'][i](temp_token)
                       for i, temp_token in enumerate(tokenize_timestamp(temporal))], -1)
        )  # (B, L, E)
        # Temporal values lower than 0 stands for feature mask.
        temporal_e = temporal_e.masked_fill(temporal[..., :1] < 0, 0)

        pos_encoding = self.pos_encode_layer(positions)
        traj_h = spatial_e + temporal_e + pos_encoding
        batch_mask = get_batch_mask(B, L, valid_lens)
        traj_h = self.traj_view['seq_encoder'](traj_h, src_key_padding_mask=batch_mask)
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
        traj_h = self.cal_traj_h(spatial, temporal, valid_lens)

        return traj_h

    def loss(self, input_seq, valid_lens):
        """Calcualte the pre-training loss.

        Args:
            Same as the forward function.

        Returns:
            FloatTensor: the pre-training loss value of this batch.
        """
        B, L, _ = input_seq.shape
        positions = repeat(torch.arange(L), 'L -> B L', B=B)
        batch_mask = get_batch_mask(B, L, valid_lens)
        pos_encoding = self.pos_encode_layer(positions)

        # Trajectory (spatio-temporal) view.
        spatial = input_seq[:, :, COL_I['spatial']]  # (B, L, 2)
        temporal = input_seq[:, :, COL_I['temporal']]  # (B, L, 2)
        traj_h = self.cal_traj_h(spatial, temporal, valid_lens)

        # Road view.
        road = input_seq[:, :, COL_I['road']].long()
        road_index_e = self.road_view['index_embed_layer'](road)
        road_text_e = self.road_view['text_embed_layer'](self.road_view['text_embed_mat'](road))
        road_h = road_index_e + road_text_e + pos_encoding
        road_h = self.poi_view['seq_encoder'](road_h, src_key_padding_mask=batch_mask)
        road_h = road_h.masked_fill(batch_mask.unsqueeze(-1), 0).sum(1) / valid_lens.unsqueeze(-1)

        # POI view.
        poi = ((self.poi_coors.unsqueeze(0).unsqueeze(0) -
                spatial.unsqueeze(2)) ** 2).sum(-1).argmin(dim=-1)
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