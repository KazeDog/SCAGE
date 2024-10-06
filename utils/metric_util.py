import loguru
import numpy as np
import torch
import torchmetrics
from sklearn.metrics import (auc, precision_recall_curve, roc_auc_score)
from torch import Tensor
from torchmetrics import Accuracy

from data_process.function_group_constant import nfg
from utils.userconfig_util import get_dataset_form


def compute_cls_metric(y_true, y_pred):
    # print('y_pred=', y_pred)
    y_pred = np.array(y_pred)
    # print('y_true=', y_true)
    # y_true = y_true[:, 1::2]
    # y_true = [item[0] for item in y_true]
    y_true = np.array(y_true).reshape(y_pred.shape)
    dataset_form: str = get_dataset_form()
    if dataset_form == 'pyg':
        is_valid = y_true ** 2 > 0
    elif dataset_form == 'pkl':
        is_valid = y_true >= 0
    else:
        raise ValueError(f"dataset_form {dataset_form} not supported")
    roc_list = []
    for i in range(y_true.shape[1]):
        valid, label, pred = is_valid[:, i], y_true[:, i], y_pred[:, i]
        if dataset_form == 'pyg':
            label = (label[valid] + 1) / 2
        elif dataset_form == 'pkl':
            label = (label[valid] + 0.0)
        else:
            raise ValueError(f"dataset_form {dataset_form} not supported")
        # AUC is only defined when there is at least one positive pretrain_data.
        if len(np.unique(label)) == 2:
            roc_list.append(roc_auc_score(label, pred[valid]))

    roc_auc = np.mean(roc_list)
    # print('Valid ratio: %s' % (np.mean(is_valid)))
    if len(roc_list) == 0:
        raise RuntimeError("No positively labeled pretrain_data available. Cannot compute ROC-AUC.")
    return roc_auc


def compute_cls_metric_tensor(y_true: Tensor, y_pred: Tensor):
    y_true = y_true.view(y_pred.shape)
    dataset_form: str = get_dataset_form()
    if dataset_form == 'pyg':
        is_valid = y_true ** 2 > 0
    elif dataset_form == 'pkl':
        is_valid = y_true >= 0
    else:
        raise ValueError(f"dataset_form {dataset_form} not supported")
    roc_list = torch.zeros(y_true.shape[1], device='cpu')
    for i in range(y_true.shape[1]):
        valid, label, pred = is_valid[:, i], y_true[:, i], y_pred[:, i]
        if dataset_form == 'pyg':
            label = (label[valid] + 1) / 2
        elif dataset_form == 'pkl':
            label = (label[valid] + 0.0)
        else:
            raise ValueError(f"dataset_form {dataset_form} not supported")
        # AUC is only defined when there is at least one positive pretrain_data.
        if len(torch.unique(label)) == 2:
            # roc_list[i] = roc_auc_score(label, pred[valid])
            roc_list[i] = roc_auc_score(label.detach().cpu().numpy(), pred[valid].detach().cpu().numpy())
        else:
            loguru.logger.warning(f"No positively labeled pretrain_data available for label {i}. Cannot compute ROC-AUC.")

    roc_auc: Tensor = torch.mean(roc_list).cuda()
    if torch.isnan(roc_auc):
        raise RuntimeError("No positively labeled pretrain_data available. Cannot compute ROC-AUC.")
    return roc_auc.item()


atom_acc = Accuracy(num_labels=nfg() + 1, average="micro", task='multilabel').cuda()


def compute_atom_fg_metric(y_score, y_true):
    global atom_acc
    acc = atom_acc(y_score, y_true)
    # roc_auc = torchmetrics.functional.auroc(y_score, y_true, task='multilabel', num_labels=nfg())

    y_true = np.array(y_true.detach().cpu().numpy())
    y_score = np.array(y_score.detach().cpu().numpy())
    roc_list = []
    # roc_auc = roc_auc_score(y_true, y_score, average='micro', multi_class='ovr')
    for i in range(y_true.shape[1]):
        label, socre = y_true[:, i], y_score[:, i]
        if len(np.unique(label)) == 2:
            roc_list.append(roc_auc_score(label, socre))

    roc_auc = np.mean(roc_list)
    return acc, roc_auc


torchmetrics_auroc = torchmetrics.AUROC(num_classes=2, task='binary')


def compute_atom_fg_metric_cuda_tensor(y_score: Tensor, y_true: Tensor):
    """
    Compute the accuracy and ROC AUC for atom functional group classification.
    Args:
        y_score: dim of (atom_total_num_in_batch, kind_of_functional_group)
        y_true: dim of (atom_total_num_in_batch, kind_of_functional_group)
    For example, if there are 3 kinds of functional groups, the y_score and y_true should be like:
    y_score = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
    y_true = [[0, 1, 0], [1, 0, 0], [1, 0, 1]] (each atom can be labeled with multiple functional groups)
    Returns:

    """
    with torch.no_grad():
        global torchmetrics_auroc, atom_acc

        # Calculate accuracy using the provided Accuracy metric from torchmetrics
        acc = atom_acc(y_score, y_true)

        # List to store auroc values
        roc_list = []

        # Loop through each column (assuming each column is an independent binary classification)
        for i in range(y_true.shape[1]):
            label, score = y_true[:, i], y_score[:, i]
            # Only compute AUROC if the label contains two unique classes
            if len(torch.unique(label)) == 2:
                torchmetrics_auroc.update(score, label.int())
                roc_list.append(torchmetrics_auroc.compute().item())  # Compute and collect the AUROC
                torchmetrics_auroc.reset()  # Reset after each computation

        # Calculate the mean of ROC AUC values, handle case with no valid ROC AUC computations
        roc_auc = np.mean(roc_list) if roc_list else 0

        return acc, roc_auc


def compute_bond_fg_metric(y_score, y_true):
    bond_acc = Accuracy(num_classes=8, average="micro", task='multiclass').cuda()
    acc = bond_acc(y_score, y_true)

    roc_auc = torchmetrics.functional.auroc(y_score, y_true, task='multiclass', num_classes=8)
    # y_true = np.array(y_true.detach().cpu().numpy())
    # y_score = np.array(y_score.detach().cpu().numpy())
    # y_score = F.softmax(torch.Tensor(y_score).to('cuda'), dim=-1).detach().cpu().numpy()
    # roc_auc = roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr')
    return acc, roc_auc


sp_metric_acc = Accuracy(num_classes=21, average="weighted", task='multiclass').cuda()


def compute_sp_metric(y_score, y_true):
    acc = sp_metric_acc(y_score, y_true)
    roc_auc = torchmetrics.functional.auroc(y_score, y_true, task='multiclass', num_classes=21, average='weighted')
    return acc, roc_auc

finger_metric_acc = Accuracy(num_classes=2, average="weighted", task='binary').cuda()
def compute_finger_metric(y_score, y_true):
    acc = finger_metric_acc(y_score, y_true)
    roc_auc = torchmetrics.functional.auroc(y_score, y_true, task='binary', num_classes=2, average='weighted')
    return acc, roc_auc


pair_distances_metric_acc = Accuracy(num_classes=30, average="weighted", task='multiclass').cuda()


def compute_pair_distances_metric(y_score, y_true):
    acc = pair_distances_metric_acc(y_score, y_true)
    roc_auc = torchmetrics.functional.auroc(y_score, y_true, task='multiclass', num_classes=30, average='weighted')
    return acc, roc_auc


angles_metric_acc = Accuracy(num_classes=20, average="weighted", task='multiclass').cuda()


def compute_angles_metric(y_score, y_true):
    acc = angles_metric_acc(y_score, y_true)
    roc_auc = torchmetrics.functional.auroc(y_score, y_true, task='multiclass', num_classes=20, average='weighted')
    return acc, roc_auc


def compute_cls_metric_relabel(y_true, y_pred):
    dataset_form: str = get_dataset_form()
    # print('y_pred=', y_pred)
    y_pred = np.array(y_pred)
    # print('y_true=', y_true)
    # y_true = y_true[:, 1::2]
    # y_true = [item[0] for item in y_true]
    y_true = np.array(y_true).reshape(y_pred.shape)

    if dataset_form == 'pyg':
        is_valid = y_true ** 2 > 0
    elif dataset_form == 'pkl':
        is_valid = y_true >= 0
    else:
        raise ValueError(f"dataset_form {dataset_form} not supported")
    roc_list = []
    for i in range(y_true.shape[1]):
        valid, label, pred = is_valid[:, i], y_true[:, i], y_pred[:, i]
        if dataset_form == 'pyg':
            label = (label[valid] + 1) / 2
        elif dataset_form == 'pkl':
            label = (label[valid] + 0.0)
        else:
            raise ValueError(f"dataset_form {dataset_form} not supported")
        # AUC is only defined when there is at least one positive pretrain_data.
        if len(np.unique(label)) == 2:
            roc_list.append(roc_auc_score(label, pred[valid]))

    roc_auc = np.mean(roc_list)
    # print('Valid ratio: %s' % (np.mean(is_valid)))
    if len(roc_list) == 0:
        raise RuntimeError("No positively labeled pretrain_data available. Cannot compute ROC-AUC.")
    return roc_auc


def compute_cliff_metric(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    roc_list = []
    for i in range(y_true.shape[1]):
        label, pred = y_true[:, i], y_pred[:, i]
        # AUC is only defined when there is at least one positive pretrain_data.
        if len(np.unique(label)) == 2:
            roc_list.append(roc_auc_score(label, pred))

    roc_auc = np.mean(roc_list)
    # print('Valid ratio: %s' % (np.mean(is_valid)))
    if len(roc_list) == 0:
        raise RuntimeError("No positively labeled pretrain_data available. Cannot compute ROC-AUC.")
    return roc_auc


def compute_reg_metric(y_true: Tensor, y_pred: Tensor):
    mae_list = []
    rmse_list = []
    for i in range(y_true.shape[1]):
        label, pred = y_true[:, i], y_pred[:, i]
        # mae = mean_absolute_error(label, pred)
        mae = torch.mean(torch.abs(label - pred))
        # rmse = torch.sqrt(mean_squared_error(label, pred, squared=True))
        rmse = torch.sqrt(torch.mean((label - pred) ** 2))
        mae_list.append(mae)
        rmse_list.append(rmse)

    mae, rmse = torch.mean(torch.tensor(mae_list)), torch.mean(torch.tensor(rmse_list))
    # return mae, rmse
    return mae.item(), rmse.item()


# noinspection SpellCheckingInspection
def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
    """
    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred).reshape(-1)
    if type(true) is not np.array:
        true = np.array(true)

    return np.sqrt(np.mean(np.square(true - pred)))


# noinspection SpellCheckingInspection
def calc_cliff_rmse(y_test_pred, y_test, cliff_mols_test=None, smiles_test=None,
                    y_train=None, smiles_train=None, **kwargs):
    """ Calculate the RMSE of activity cliff compounds

    :param y_test_pred: (lst/array) predicted test values
    :param y_test: (lst/array) true test values
    :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
    :param smiles_test: (lst) list of SMILES strings of the test molecules
    :param y_train: (lst/array) train labels
    :param smiles_train: (lst) list of SMILES strings of the train molecules
    :param kwargs: arguments for ActivityCliffs()
    :return: float RMSE on activity cliff compounds
    """

    # Check if we can compute activity cliffs when pre-computed ones are not provided.
    if cliff_mols_test is None:
        if smiles_test is None or y_train is None or smiles_train is None:
            raise ValueError('if cliff_mols_test is None, smiles_test, y_train, and smiles_train should be provided '
                             'to compute activity cliffs')

    # Convert to numpy array if it is none
    y_test_pred = np.array(y_test_pred).reshape(-1) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    if cliff_mols_test is None:
        y_train = np.array(y_train) if type(y_train) is not np.array else y_train
        # Calculate cliffs and
        # noinspection PyUnresolvedReferences
        cliffs = ActivityCliffs(smiles_train + smiles_test, np.append(y_train, y_test))
        cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, **kwargs)
        # Take only the test cliffs
        cliff_mols_test = cliff_mols[len(smiles_train):]

    # Get the index of the activity cliff molecules
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]

    return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)
