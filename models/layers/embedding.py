import torch
import torch.nn as nn
from torch.nn import Embedding
from torch import Tensor

from data_process.edge_types import EDGE_TYPES_DICT, EDGE_TYPES_DICT_OLD
from utils.global_var_util import GlobalVar
from data_process.compound_tools import CompoundKit
from echo_logger import print_debug
from typing import Dict


# noinspection PyUnresolvedReferences
class AtomEmbedding(nn.Module):

    def __init__(self, atom_names, embed_dim, num_kernel):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names

        self.embed_list = nn.ModuleList()
        for name in self.atom_names:
            embed = nn.Embedding(CompoundKit.get_atom_feature_size(name) + 5, embed_dim, padding_idx=0)
            self.embed_list.append(embed)

        self.graph_embedding = nn.Embedding(1, embed_dim)

        self.graph_finger_print = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mass_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim)
        )

        self.van_der_waals_radis_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim)
        )

        self.partial_charge_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim)
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, node_features: Dict[str, Tensor]):
        out_embed = 0
        for i, name in enumerate(self.atom_names):
            out_embed += self.embed_list[i](node_features[name])
        if "mass" in node_features:
            mass_embed = self.mass_embedding(node_features["mass"])
            out_embed += mass_embed
        if "van_der_waals_radius" in node_features:
            van_der_waals_radis_embed = self.van_der_waals_radis_embedding(node_features["van_der_waals_radius"])
            out_embed += van_der_waals_radis_embed
        if "partial_charge" in node_features:
            partial_charge_embed = self.partial_charge_embedding(node_features["partial_charge"])
            out_embed += partial_charge_embed

        graph_token_embed = self.graph_embedding.weight.unsqueeze(0).repeat(out_embed.size()[0], 1, 1)

        out_embed = torch.cat([graph_token_embed, out_embed], dim=1)
        # normalize
        # out_embed = out_embed / (out_embed.norm(dim=-1, keepdim=True) + 1e-5)
        out_embed = self.final_layer_norm(out_embed)
        return out_embed

@torch.jit.script
def gaussian(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    # noinspection PyTypeChecker
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (torch.sqrt(2 * torch.pi) * std)


# noinspection PyPep8Naming
class GaussianKernel(nn.Module):
    def __init__(self, K: int = 128, std_width: float = 1.0, start: float = 0.0, stop: float = 9.0):
        super().__init__()
        self.K = K
        mean = torch.linspace(start, stop, K)
        std = std_width * (mean[1] - mean[0])
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.mul = Embedding(1, 1, padding_idx=0)
        self.bias = Embedding(1, 1, padding_idx=0)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1.0)
    def forward(self, x: Tensor) -> Tensor:
        mul = self.mul.weight
        bias = self.bias.weight
        x = (mul * x.unsqueeze(-1)) + bias
        expand_shape = [-1] * len(x.shape)
        expand_shape[-1] = self.K
        x = x.expand(expand_shape)
        mean = self.mean.float()
        return gaussian(x.float(), mean, self.std)
