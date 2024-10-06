from echo_logger import *

from data_process.function_group_constant import reference_fn_group

from utils.loss_util import bce_loss
from utils.userconfig_util import get_dataset_dir

rf = reference_fn_group
rf.update({"0": "None"})
# convert into int keys
rf = {int(k): v for k, v in rf.items()}
all_possible_fg_nums = len(rf)
