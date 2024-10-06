from typing import Callable

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor, tensor

from _config import get_audit_atom_fg_frequency_arr
from utils.global_var_util import LossStyle, GlobalVar


# noinspection SpellCheckingInspection,PyPep8Naming
class bce_loss(nn.Module):
    def __init__(self, weights=None):
        super(bce_loss, self).__init__()
        self.weights = weights

    def forward(self, pred, label):
        if self.weights is not None:
            fore_weights = torch.tensor(self.weights[0])
            back_weights = self.weights[1]

            fore_weights = fore_weights.to('cuda')
            back_weights = back_weights.to('cuda')
            weights = label * back_weights + (1.0 - label) * fore_weights
        else:
            weights = torch.ones(label.shape, device=label.device)

        loss = F.binary_cross_entropy_with_logits(pred, label, weights, reduction='none')
        return loss


# noinspection SpellCheckingInspection,PyPep8Naming
class NTXentLoss_atom(nn.Module):
    def __init__(self, t=0.1):
        super(NTXentLoss_atom, self).__init__()
        self.T = t
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss(ignore_index=-1)

    def forward(self, out, out_mask, labels):
        out = nn.functional.normalize(out, dim=-1)
        out_mask = nn.functional.normalize(out_mask, dim=-1)

        logits = torch.matmul(out_mask, out.permute(0, 2, 1))
        logits /= self.T

        softmaxs = self.softmax(logits)
        loss = self.criterion(softmaxs.transpose(1, 2), labels)

        return loss, logits


# noinspection SpellCheckingInspection
class NTXentLoss(torch.nn.Module):

    def __init__(self, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_similarity
        else:
            return self._dot_similarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to('cuda')

    @staticmethod
    def _dot_similarity(x: Tensor, y: Tensor):
        v: Tensor = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_similarity(self, x: Tensor, y: Tensor):
        # x shape: (N, 1, C)
        # y shape: (1, N, C)
        # v shape: (N, N)
        v: Tensor = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to('cuda').long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


def get_balanced_atom_fg_loss(pred__, label__, loss_f_atom_fg: Callable, advanced=True):
    if not advanced:
        label_positive_index = (label__ == 1)
        label_zero_index = (label__ == 0)
        label_pos = label__[label_positive_index]
        label_zero = label__[label_zero_index]
        index_zero = torch.randperm(len(label_zero), device='cuda')
        label_zero_random = label_zero[index_zero[:len(label_pos)]]
        label_all = torch.cat([label_pos, label_zero_random], dim=0)
        pred_positive = pred__[label_positive_index]
        pred_negative = pred__[label_zero_index][index_zero[:len(label_pos)]]
        pred_all = torch.cat([pred_positive, pred_negative], dim=0)
        return loss_f_atom_fg(pred_all, label_all + 0.0)
    else:
        label_positive_index = (label__ == 1)
        label_zero_index = (label__ == 0)
        label_pos = label__[label_positive_index]
        label_zero = label__[label_zero_index]
        index_zero = torch.randperm(len(label_zero), device='cuda')
        label_zero_random = label_zero[index_zero[:len(label_pos)]]
        label_all = torch.cat([label_pos, label_zero_random], dim=0)
        pred_positive = pred__[label_positive_index]
        pred_negative = pred__[label_zero_index][index_zero[:len(label_pos)]]
        indices: Tensor = torch.nonzero(label_positive_index)
        # table__ = torch.ones(label__.shape[1], device='cuda')
        table__ = tensor(get_audit_atom_fg_frequency_arr(), device='cuda')
        indices_i, indices_j, indices_k = indices[:, 0], indices[:, 1], indices[:, 2]
        # print('indices_j=', indices_j)
        # print('indices_k=', indices_k)
        w_positive = table__[indices_k]
        w_negative = torch.ones_like(table__[indices_k])
        w_united = torch.cat([w_positive, w_negative], dim=0)
        pred_all = torch.cat([pred_positive, pred_negative], dim=0)
        return loss_f_atom_fg(pred_all, label_all + 0.0, weight=w_united)


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=1, epsilon=1.e-9, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, np.ndarray):
            self.alpha = torch.tensor(alpha, device=device)
        else:
            self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        num_labels = input.size(-1)
        idx = target.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key[:, 0] = 0  # ignore 0 index.
        logits = torch.softmax(input, dim=-1)
        loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss = loss.sum(1)
        return loss.mean()


def get_focal_loss(pred, label, alpha, device):
    loss = FocalLoss(alpha=alpha, device=device)
    return loss(pred, label)


def use_balanced_atom_fg_loss(loss_style_):
    return loss_style_ == LossStyle.loong_and_finger_and_balanced_fg_loss or \
        loss_style_ == LossStyle.loong_and_finger_and_yw_balanced_fg_loss
