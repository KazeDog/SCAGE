from functools import cache

import loguru
from echo_logger import deprecated

from data_process.edge_types import EDGE_TYPES_DICT, EDGE_TYPES_DICT_OLD
from data_process.function_group_constant import nfg


class GlobalVar:
    max_epochs = None
    hyper_node_init_with_morgan = False
    use_finger_re_construction_loss = False
    loss_style: str = ""
    use_fg_loss = False
    use_comprehensive_loss = True
    use_calc_mt_loss = True
    use_cliff_pred_loss = False
    balanced_atom_fg_loss = False
    transformer_dim = 0
    ffn_dim = 0
    embedding_style = "more"
    num_heads = 0
    patience = -1
    fg_number = nfg() + 1
    edge_type_number = len(EDGE_TYPES_DICT)
    fg_edge_type_num = 8
    max_fn_group_edge_type_additional_importance_score = 5  # real max num is this num + 2
    debug_dataloader_collator = False
    data_process_style = 'qjb'  # or 'qjb'
    distance_bar_num = 3
    parallel_train = True
    use_ckpt = False
    freeze_layers = 0
    use_cliff_pred = False
    is_mol_net_tasks = False
    pretrain_task = ['finger', 'fg']  # can be fg, sp, pair_distance, angle
    finetune_task = ['finger', 'fg']
    pretrain_dataset_is_from_pkl_file_directly = False
    use_testing_pretrain_dataset = False
    fg_loss_type = 'advanced'  # possible: 'advanced', 'raw', 'new'
    dist_bar = None
    dist_bar_type = '3d'

    @staticmethod
    @cache
    def get_embedding_dict():
        if GlobalVar.embedding_style == 'more':
            return EDGE_TYPES_DICT
        elif GlobalVar.embedding_style == 'less':
            return EDGE_TYPES_DICT_OLD
        else:
            raise ValueError("Unknown embedding style, expect 'more' or 'less'")

    @staticmethod
    @cache
    def get_loss_num():
        tasks = ['finger', 'fg', 'sp', 'angle']
        return sum(1 for task in tasks if task in GlobalVar.pretrain_task)

class MaskingConfig:
    mask_style = 'only'


class FunctionGroupConfig:
    fg_style = 'new'


class HyperParamTransport:
    """Used for transport hyperparameters from command line to override the default value in params_nni or GlobalVar."""
    init_lr = None
    batch_size = None


class LossStyle:
    loong_loss = "loong_loss"
    complex_loss = "complex_loss"


def set_dist_bar_two(dist_bar_start, dist_bar_end):
    if dist_bar_start >= dist_bar_end:
        print(f"dist_bar_start: {dist_bar_start} should be less than dist_bar_end: {dist_bar_end}")
        GlobalVar.dist_bar = None
    else:
        GlobalVar.dist_bar = [dist_bar_start, dist_bar_end]


def set_dist_bar_three(dist_bar_start, dist_bar_middle, dist_bar_end):
    if dist_bar_start >= dist_bar_middle or dist_bar_middle >= dist_bar_end:
        print(
            f"dist_bar_start: {dist_bar_start} should be less than dist_bar_middle: {dist_bar_middle} and dist_bar_middle: {dist_bar_middle} should be less than dist_bar_end: {dist_bar_end}")
        GlobalVar.dist_bar = None
    else:
        GlobalVar.dist_bar = [dist_bar_start, dist_bar_middle, dist_bar_end]


class RoutineControl:
    save_best_ckpt = True
    debug_train_loss_and_auc = False


class DEFAULTS:
    DROP_RATE = 0.1


REGRESSION_TASK_NAMES = ['esol', 'freesolv', 'lipophilicity']
