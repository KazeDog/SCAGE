from typing import List

import torch
from echo_logger import *
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F
import math

from models.layers.ffn import PositionWiseFeedForward
from utils.global_var_util import GlobalVar, DEFAULTS
from utils.public_util import qkv_reset_parameters


# noinspection DuplicatedCode
class MultiScaleAttention(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout=0.1, attn_dropout=0.1, temperature=1,
                 use_super_node=True):
        super(MultiScaleAttention, self).__init__()
        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads  # number of heads
        self.temperature = temperature
        self.use_super_node = use_super_node
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.a_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(len(GlobalVar.dist_bar))])

        self.scale_linear = nn.Sequential(nn.Linear(len(GlobalVar.dist_bar) * hidden_dim, hidden_dim),
                                          nn.SiLU(),
                                          nn.Dropout(p=dropout),
                                          nn.Linear(hidden_dim, hidden_dim))
        self.cnn = nn.Sequential(nn.Conv2d(1, num_heads, kernel_size=1),
                                 nn.SiLU(),
                                 nn.Conv2d(num_heads, num_heads, kernel_size=1))
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        # qkv_reset_parameters(self)

    def forward(self, x, dist, dist_bar: List[int], mask=None, attn_bias=None):
        residual = x
        batch_size = x.size(0)
        query = self.q_proj(x)  # (batch_size, atom_num, hidden_dim)
        key = self.k_proj(x)  # (batch_size, atom_num, hidden_dim)
        value = self.v_proj(x)  # (batch_size, atom_num, hidden_dim)

        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # ScaledDotProductAttention
        if mask is not None and len(mask.shape) == 3:
            mask = mask.unsqueeze(1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # bacth * len * len
            scores = scores + attn_bias

        if mask is not None:
            if scores.shape == mask.shape:  # different heads have different mask
                scores = scores * mask
                scores = scores.masked_fill(scores == 0, -1e12)
            else:
                scores = scores.masked_fill(mask == 0, -1e12)

        dist = dist.unsqueeze(1)
        n_node = dist.size(-1) + int(self.use_super_node)
        new_dist = dist.new_ones([batch_size, 1, n_node, n_node])
        new_dist[:, :, int(self.use_super_node):, int(self.use_super_node):] = dist

        multi_out = []
        attn_list = []  # Used only for debug

        dist_bar = dist_bar.unsqueeze(1)
        for i in range(dist_bar.shape[-1]):
            # for i in self.dist_bar:
            cut = dist_bar[:, :, i]
            cut = cut.repeat(1, new_dist.shape[-1]).unsqueeze(1)
            dist_mask = new_dist < cut
            indices = (torch.arange(new_dist.shape[0], device="cuda").view(new_dist.shape[0], 1, 1, 1)
                       .expand(new_dist.shape[0], new_dist.shape[0], new_dist.shape[-2], new_dist.shape[-1]))
            mask_ = torch.eq(indices, torch.arange(new_dist.shape[0], device="cuda").view(1, new_dist.shape[0], 1, 1))
            dist_mask = torch.where(mask_, dist_mask, torch.zeros_like(dist_mask))
            dist_mask = dist_mask.sum(dim=1, keepdim=True)

            dist_mask[:, :, 0, :] = 1
            dist_mask[:, :, :, 0] = 1

            scores_dist = scores.masked_fill(dist_mask == 0, -1e12)
            attn = self.attn_dropout(F.softmax(scores_dist, dim=-1))  # Used only for debug
            attn_list.append(attn)
            out = torch.matmul(attn, value)
            multi_out.append(out)
        x_list = [x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k) for x in multi_out]
        x = torch.cat([self.a_proj[i](x_list[i]) for i in range(len(GlobalVar.dist_bar))], dim=-1)
        x = self.scale_linear(x)
        x += residual
        x = self.layer_norm(x)

        return x, attn_list


class MultiScaleTransformer(nn.Module):
    def __init__(self, num_heads, hidden_dim, ffn_hidden_dim, dropout=0.1, attn_dropout=0.1, temperature=1,
                 activation_fn='GELU'):
        super(MultiScaleTransformer, self).__init__()
        assert hidden_dim % num_heads == 0
        self.self_attention = MultiScaleAttention(num_heads, hidden_dim, dropout, attn_dropout, temperature)
        self.self_ffn_layer = PositionWiseFeedForward(hidden_dim, ffn_hidden_dim, activation_fn=activation_fn)


    def forward(self, x, dist, dist_bar, attn_mask, attn_bias=None):
        x, attn = self.self_attention(x, dist, dist_bar, mask=attn_mask, attn_bias=attn_bias)
        x = self.self_ffn_layer(x)


        return x, attn


class EncoderAtomLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_hidden_dim, num_heads, dropout=DEFAULTS.DROP_RATE,
                 attn_dropout=DEFAULTS.DROP_RATE, temperature=1, activation_fn='GELU'):
        super(EncoderAtomLayer, self).__init__()
        self.transformer = MultiScaleTransformer(num_heads, hidden_dim, ffn_hidden_dim, dropout, attn_dropout,
                                                 temperature, activation_fn)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer_norm.weight)

    def forward(self, x: Tensor, attn_mask: Tensor,
                attn_bias: Tensor = None, dist=None, dist_bar=None):

        x, attn = self.transformer(x, dist, dist_bar, attn_mask, attn_bias)

        return x


