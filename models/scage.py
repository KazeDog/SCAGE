import loguru
import torch
import torch.nn as nn
from echo_logger import *
from torch import Tensor
import sys
from models.layers.encoder import EncoderAtomLayer
from models.layers.embedding import AtomEmbedding
from models.layers.projection import (AtomProjection, AtomFGProjection,
                                      BondFGProjection, AtomPairProjection, AtomAngleProjection, AtomPairProjectionBin,
                                      AtomAngleProjectionBin)
from naive_fg import all_possible_fg_nums
from utils.global_var_util import GlobalVar


# noinspection PyPep8Naming,SpellCheckingInspection
class Scage(nn.Module):
    def __init__(self, mode, atom_names, atom_embed_dim, num_kernel,
                 layer_num, num_heads, atom_FG_class: int, hidden_size, num_tasks):
        super(Scage, self).__init__()

        self.mode = mode
        self.atom_names = atom_names
        self.atom_embed_dim = atom_embed_dim
        self.num_kernel = num_kernel
        self.layer_num = layer_num
        self.num_heads = num_heads
        self.atom_FG_class = atom_FG_class
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks

        self.register_buffer('dist_bar', torch.tensor(GlobalVar.dist_bar))

        self.atom_feature = AtomEmbedding(self.atom_names, self.atom_embed_dim, self.num_kernel)

        self.EncoderAtomList = nn.ModuleList()
        for j in range(self.layer_num):
            self.EncoderAtomList.append(
                EncoderAtomLayer(self.atom_embed_dim, self.hidden_size, self.num_heads)
            )

        if self.mode == "pretrain":
            if 'fg' in GlobalVar.pretrain_task:
                self.head_FG_atom = AtomFGProjection(self.atom_embed_dim, self.atom_embed_dim // 2, self.atom_FG_class)
            if 'finger' in GlobalVar.pretrain_task:
                self.head_finger_keeping_atom = AtomProjection(self.atom_embed_dim, self.atom_embed_dim * 2, 2048)
                self.head_finger_keeping_bond = AtomProjection(self.atom_embed_dim, self.atom_embed_dim * 2, 2048)

            if 'sp' in GlobalVar.pretrain_task or 'pair_distance' in GlobalVar.pretrain_task:
                self.head_pair_distances = AtomPairProjection(self.atom_embed_dim, self.atom_embed_dim // 8, 1)
            if 'angle' in GlobalVar.pretrain_task:
                self.head_angle = AtomAngleProjection(self.atom_embed_dim, self.atom_embed_dim // 2, 1)

        elif self.mode == "pretrain_bin":
            if 'fg' in GlobalVar.pretrain_task:
                self.head_FG_atom = AtomFGProjection(self.atom_embed_dim, self.atom_embed_dim // 2, self.atom_FG_class)
            if 'finger' in GlobalVar.pretrain_task:
                self.head_finger_keeping_atom = AtomProjection(self.atom_embed_dim, self.atom_embed_dim * 2, 2048)
            if 'sp' in GlobalVar.pretrain_task or 'pair_distance' in GlobalVar.pretrain_task:
                self.head_pair_distances = AtomPairProjectionBin(self.atom_embed_dim, self.atom_embed_dim // 8, 1)
            if 'angle' in GlobalVar.pretrain_task:
                self.head_angle = AtomAngleProjectionBin(self.atom_embed_dim, self.atom_embed_dim // 2, 20)

        elif self.mode == "finetune":
            self.head_Graph = AtomProjection(self.atom_embed_dim, self.atom_embed_dim // 2, self.num_tasks)
            self.head_finger_keeping_atom = AtomProjection(self.atom_embed_dim, self.atom_embed_dim * 2, 2048)
            self.head_finger_keeping_bond = AtomProjection(self.atom_embed_dim, self.atom_embed_dim * 2, 2048)
            self.head_FG_atom = AtomFGProjection(self.atom_embed_dim, self.atom_embed_dim // 2, self.atom_FG_class)
        elif self.mode == "cliff":
            self.head_Graph = AtomProjection(self.atom_embed_dim, self.atom_embed_dim // 2, 1)
            self.head_cliff = AtomProjection(self.atom_embed_dim, self.atom_embed_dim // 2, 1)
            self.head_finger_keeping_atom = AtomProjection(self.atom_embed_dim, self.atom_embed_dim * 2, 2048)
            self.head_finger_keeping_bond = AtomProjection(self.atom_embed_dim, self.atom_embed_dim * 2, 2048)
            self.head_FG_atom = AtomFGProjection(self.atom_embed_dim, self.atom_embed_dim // 2, self.atom_FG_class)

    def forward(self, batched_data: Dict[str, Tensor]):

        atom: Tensor = self.atom_feature(batched_data)
        batch, atom_num, _ = atom.shape

        atom_attention_mask: Tensor = batched_data["atom_attention_mask"]
        for i in range(self.layer_num):
            atom = \
                self.EncoderAtomList[i](x=atom, attn_mask=atom_attention_mask,
                                        dist=batched_data['pair_distances'],
                                        dist_bar=batched_data['atom_dist_bar']
                                        )

        if self.mode == "pretrain":
            predictions = {}
            if 'fg' in GlobalVar.pretrain_task:
                atom_fg = self.head_FG_atom(atom[:, 1:, :])
                predictions['atom_fg'] = atom_fg
            if 'finger' in GlobalVar.pretrain_task:
                atom_finger_feature = self.head_finger_keeping_atom(atom[:, 0, :])
                predictions['atom_finger_feature'] = atom_finger_feature
            if 'sp' in GlobalVar.pretrain_task or 'pair_distance' in GlobalVar.pretrain_task:
                spatial_pos_pred, pair_distences_pred = self.head_pair_distances(atom[:, 1:, :])
                predictions['spatial_pos_pred'] = spatial_pos_pred
                predictions['pair_distences_pred'] = pair_distences_pred
            if 'angle' in GlobalVar.pretrain_task:
                angle_pred = self.head_angle(atom[:, 1:, :], batched_data['angles_atom_index'])
                predictions['angle_pred'] = angle_pred

        elif self.mode == "pretrain_bin":
            predictions = {}
            if 'fg' in GlobalVar.pretrain_task:
                atom_fg = self.head_FG_atom(atom[:, 1:, :])
                predictions['atom_fg'] = atom_fg
            if 'finger' in GlobalVar.pretrain_task:
                atom_finger_feature = self.head_finger_keeping_atom(atom[:, 0, :])
                predictions['atom_finger_feature'] = atom_finger_feature

            if 'sp' in GlobalVar.pretrain_task or 'pair_distance' in GlobalVar.pretrain_task:
                spatial_pos_pred, pair_distences_pred = self.head_pair_distances(atom[:, 1:, :])
                predictions['spatial_pos_pred'] = spatial_pos_pred
                predictions['pair_distences_pred'] = pair_distences_pred
            if 'angle' in GlobalVar.pretrain_task:
                angle_pred = self.head_angle(atom[:, 1:, :], batched_data['angles_atom_index'])
                predictions['angle_pred'] = angle_pred
        elif self.mode == "finetune":
            graph_feature = self.head_Graph(atom[:, 0, :])
            finger_feature = self.head_finger_keeping_atom(atom[:, 0, :])

            atom_fg = self.head_FG_atom(atom[:, 1:, :])
            predictions = {
                "graph_feature": graph_feature,
                "finger_feature": finger_feature,
                "atom_fg": atom_fg
            }
        elif self.mode == "cliff":
            graph_feature = self.head_Graph(atom[:, 0, :])
            cliff_feature = self.head_cliff(atom[:, 0, :])
            finger_feature = self.head_finger_keeping_atom(atom[:, 0, :])

            atom_fg = self.head_FG_atom(atom[:, 1:, :])
            predictions = {
                "graph_feature": graph_feature,
                "cliff_feature": cliff_feature,
                "finger_feature": finger_feature,
                "atom_fg": atom_fg,
            }
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")

        return predictions
