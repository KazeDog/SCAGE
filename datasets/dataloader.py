import os
import pickle
import random

import lmdb
import loguru
from torch.utils.data import Dataset, Subset

from data_process.compound_tools import mol_to_data_pkl
from utils.global_var_util import GlobalVar
from utils.userconfig_util import get_current_user
from utils.viz.naive_fg_model import convert_atom_to_fg_nums__to__fg_counts


class PretrainDataset(Dataset):
    def __init__(self, root):
        self.root = root

        # if '200k+finetune.lmdb' exists, use it, otherwise use the first lmdb file in the dir, and WARNING
        fallout_filename = '200k+finetune.lmdb'
        if os.path.exists(os.path.join(self.root, fallout_filename)):
            lmdb_file = os.path.join(self.root, fallout_filename)
        else:
            lmdb_files = [f for f in os.listdir(self.root) if f.endswith('.lmdb')]
            if len(lmdb_files) == 0:
                raise ValueError('No lmdb file found in the dir')
            lmdb_file = os.path.join(self.root, lmdb_files[0])
            loguru.logger.warning(f'You did not specify the source of the dataset, and the file {fallout_filename} '
                                  f'does not exist, using the first lmdb file in the dir: {lmdb_file}')
        self.env = lmdb.open(
            lmdb_file,
            subdir=False,
            readonly=True, lock=False,
            readahead=False, meminit=False
        )
        self.txn = self.env.begin()
        self.is_raw = False
        self.length = int(self.txn.get(b'__len__').decode())


        self.safe_mol_indices = []
        self.safe_mol_num = 100
        self.safe_mol_indices_init_flag = False
        if 'fg_number' in GlobalVar.pretrain_task:
            loguru.logger.info('Pretrain task: fg_number, converting atom to fg counts')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.txn.get(str(idx).encode())
        data = pickle.loads(data)
        if check_if_atom_num_bigger_than(data['atomic_num'], 120):
            if not self.safe_mol_indices_init_flag:
                while len(self.safe_mol_indices) < self.safe_mol_num:
                    idx += 1
                    data = self.txn.get(str(idx).encode())
                    data = pickle.loads(data)
                    if not check_if_atom_num_bigger_than(data['atomic_num'], 120):
                        self.safe_mol_indices.append(idx)
                self.safe_mol_indices_init_flag = True
            sample_idx = random.choice(self.safe_mol_indices)
            return self.__getitem__(sample_idx)
        if 'function_group_index' in data and data['function_group_index'] is not None \
                and 'fg_number' in GlobalVar.pretrain_task and 'fg_number' not in data:
            data['fg_number'] = convert_atom_to_fg_nums__to__fg_counts(data['function_group_index'])
        return data


def check_if_atom_num_bigger_than(atomic_num_numpy, num):
    # print(atomic_num_numpy.shape)
    return atomic_num_numpy.shape[0] > num


class FinetuneDataset(Dataset):
    def __init__(self, root, task_name):
        # torch.multiprocessing.set_start_method('spawn')
        self.root = root
        self.task_name = task_name
        self.data = pickle.load(open(os.path.join(self.root, self.task_name, f'{self.task_name}.pkl'), "rb"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CliffDataset(Dataset):
    def __init__(self, root, task_name, mode):
        # torch.multiprocessing.set_start_method('spawn')
        self.root = root
        self.task_name = task_name
        self.data = pickle.load(open(os.path.join(self.root, f'{self.task_name}_{mode}.pkl'), "rb"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

