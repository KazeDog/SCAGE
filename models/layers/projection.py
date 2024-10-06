import torch
import torch.nn as nn
import torch.nn.functional as F
from echo_logger import print_debug

from utils.global_var_util import DEFAULTS


class AtomProjection(nn.Module):
    def __init__(self, d_atom, d_hid, d_output):
        # print d_atom, d_hid, d_output
        # print_debug('AtomProjection: d_atom={}, d_hid={}, d_output={}'.format(d_atom, d_hid, d_output))
        super(AtomProjection, self).__init__()

        self.linear_seq = nn.Sequential(
            nn.Linear(d_atom, d_hid),
            # nn.BatchNorm1d(d_hid),
            nn.SiLU(),
            nn.Dropout(p=DEFAULTS.DROP_RATE),
            nn.Linear(d_hid, d_output)
        )

    def forward(self, z, mask=None):
        x = self.linear_seq(z)
        return x


class AtomFGProjection(nn.Module):
    def __init__(self, d_atom, d_hid, d_output):
        # print d_atom, d_hid, d_output
        # print_debug('AtomProjection: d_atom={}, d_hid={}, d_output={}'.format(d_atom, d_hid, d_output))
        super(AtomFGProjection, self).__init__()

        self.linear_seq = nn.Sequential(
            nn.Linear(d_atom, d_hid),
            nn.LayerNorm(d_hid),
            nn.ReLU(),
            nn.Dropout(p=DEFAULTS.DROP_RATE),
            nn.Linear(d_hid, d_output)
        )

    def forward(self, z, mask=None):
        x = self.linear_seq(z)
        return x


class BondFGProjection(nn.Module):
    def __init__(self, d_atom, d_hid, d_output):
        # print d_atom, d_hid, d_output
        # print_debug('AtomProjection: d_atom={}, d_hid={}, d_output={}'.format(d_atom, d_hid, d_output))
        super(BondFGProjection, self).__init__()

        self.linear_seq = nn.Sequential(
            nn.Linear(d_atom, d_hid),
            nn.LayerNorm(d_hid),
            nn.ReLU(),
            nn.Dropout(p=DEFAULTS.DROP_RATE),
            nn.Linear(d_hid, d_output)
        )

    def forward(self, z, mask=None):
        x = self.linear_seq(z)
        return x


class AtomPairProjection(nn.Module):
    def __init__(self, d_atom, d_hid, d_output):
        # print d_atom, d_hid, d_output
        # print_debug('AtomProjection: d_atom={}, d_hid={}, d_output={}'.format(d_atom, d_hid, d_output))
        super(AtomPairProjection, self).__init__()

        self.linear_in = nn.Sequential(
            nn.Linear(d_atom, d_hid * 2),
            nn.GELU(),
            nn.LayerNorm(d_hid * 2)
        )

        # self.linear_out = nn.Linear(d_hid ** 2, d_output)

        self.linear_out1 = nn.Linear(d_hid ** 2, 11)
        self.linear_out2 = nn.Linear(d_hid ** 2, d_output)

    def forward(self, z, mask=None):
        ab = self.linear_in(z)
        a, b = ab.chunk(2, dim=-1)
        x = torch.einsum("...bc,...de->...bdce", a, b)
        # bsz, n, d = a.shape
        # a = a.view(bsz, n, 1, d, 1)
        # b = b.view(bsz, 1, n, 1, d)
        # x = a * b
        del a, b
        x = x.view(x.shape[:-2] + (-1,))
        # x1 = self.linear_out1(x)
        # x2 = self.linear_out2(x)
        # return x1, x2
        x = self.linear_out(x)
        return x


class AtomAngleProjection(nn.Module):
    def __init__(self, d_atom, d_hid, d_output):
        # print d_atom, d_hid, d_output
        # print_debug('AtomProjection: d_atom={}, d_hid={}, d_output={}'.format(d_atom, d_hid, d_output))
        super(AtomAngleProjection, self).__init__()

        self.linear_seq = nn.Sequential(
            nn.Linear(d_atom, d_hid),
            nn.BatchNorm1d(d_hid),
            nn.ReLU(),
            nn.Dropout(p=DEFAULTS.DROP_RATE),
            nn.Linear(d_hid, d_output)
        )

    def forward(self, z, angel_atom_table, mask=None):
        valid_entries = angel_atom_table[:, :, 0] != -1
        indices = torch.nonzero(valid_entries)
        indices_i, indices_j = indices[:, 0], indices[:, 1]

        x = z[indices_i, angel_atom_table[indices_i, indices_j, 0]] + z[
            indices_i, angel_atom_table[indices_i, indices_j, 1]] + z[
                indices_i, angel_atom_table[indices_i, indices_j, 2]]

        x = self.linear_seq(x)
        return x


class AtomPairProjectionBin(nn.Module):
    def __init__(self, d_atom, d_hid, d_output):
        # print d_atom, d_hid, d_output
        # print_debug('AtomProjection: d_atom={}, d_hid={}, d_output={}'.format(d_atom, d_hid, d_output))
        super(AtomPairProjectionBin, self).__init__()

        self.linear_in = nn.Sequential(
            nn.Linear(d_atom, d_hid * 2),
            nn.GELU(),
            nn.LayerNorm(d_hid * 2)
        )

        # self.linear_out = nn.Linear(d_hid ** 2, d_output)

        self.linear_out1 = nn.Linear(d_hid ** 2, 21)
        self.linear_out2 = nn.Linear(d_hid ** 2, 30)

    def forward(self, z, mask=None):
        ab = self.linear_in(z)
        a, b = ab.chunk(2, dim=-1)
        x = torch.einsum("...bc,...de->...bdce", a, b)
        del a, b
        x = x.view(x.shape[:-2] + (-1,))
        x1 = self.linear_out1(x)
        x2 = self.linear_out2(x)
        return x1, x2
        # x = self.linear_out(x)
        # return x


class AtomAngleProjectionBin(nn.Module):
    def __init__(self, d_atom, d_hid, d_output):
        # print d_atom, d_hid, d_output
        # print_debug('AtomProjection: d_atom={}, d_hid={}, d_output={}'.format(d_atom, d_hid, d_output))
        super(AtomAngleProjectionBin, self).__init__()

        self.linear_seq = nn.Sequential(
            nn.Linear(d_atom, d_hid),
            nn.BatchNorm1d(d_hid),
            nn.ReLU(),
            nn.Dropout(p=DEFAULTS.DROP_RATE),
            nn.Linear(d_hid, d_output)
        )

    def forward(self, z, angel_atom_table, mask=None):
        valid_entries = angel_atom_table[:, :, 0] != -1
        indices = torch.nonzero(valid_entries)
        indices_i, indices_j = indices[:, 0], indices[:, 1]

        x = z[indices_i, angel_atom_table[indices_i, indices_j, 0]] + z[
            indices_i, angel_atom_table[indices_i, indices_j, 1]] + z[
                indices_i, angel_atom_table[indices_i, indices_j, 2]]

        x = self.linear_seq(x)
        return x