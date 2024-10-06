import os
import pickle
from typing import Iterable, Union

import numpy as np
from echo_logger import print_info
from matplotlib import pyplot as plt

from _config import task_configs
from data_process.function_group_constant import reference_fn_group
from utils.userconfig_util import get_dataset_dir
from torch.utils.data import Dataset, DataLoader, Subset

rf = reference_fn_group
rf.update({"0": "None"})
# convert into int keys
rf = {int(k): v for k, v in rf.items()}
all_possible_fg_nums = len(rf)


def convert_atom_to_fg_nums__to__fg_counts(one_item) -> np.ndarray:
    # print("all_possible_fg_nums", all_possible_fg_nums)
    fg_counts = np.zeros(all_possible_fg_nums)
    for fg_ids in one_item.values():
        for fg_id in fg_ids:
            fg_counts[fg_id] += 1
    return fg_counts
