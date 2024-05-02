import math

import torch
from torch import nn
import numpy as np
from einops import rearrange


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super(PositionalEmbedding, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        B, L = pos_seq.shape
        sinusoid_inp = torch.ger(rearrange(pos_seq, 'B L -> (B L)'), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = rearrange(pos_emb, '(B L) E -> B L E', B=B, L=L)

        return pos_emb


class FourierEncode(nn.Module):
    """
    A type of trigonometric encoding for encode continuous values into distance-sensitive vectors.
    """

    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(),
                                  requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x):
        """
        :param x: input sequence for encoding, (batch_size, seq_len, 1)
        :return: encoded sequence, shape (batch_size, seq_len, embed_size)
        """
        if len(x.shape) < 3:
            x = x.unsqueeze(-1)

        encode = x * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        encode = torch.cos(encode)
        return self.div_term * encode
