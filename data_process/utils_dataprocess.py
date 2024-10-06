import multiprocessing
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple
from unittest import TestCase

import joblib
import loguru
import yaml

from utils.userconfig_util import config_current_user, config_dataset_form

# from data_process import algos

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
import math
from data_process import algos

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from collections import defaultdict
import sys
from echo_logger import *
import lmdb

from _config import pdir, get_downstream_task_names
from utils.global_var_util import GlobalVar

# sys.path.append("/mnt/8t/qjb/workspace/loong/")

from compound_tools import mol_to_data_pkl, Compound3DKit, binning_matrix, \
    get_all_matched_fn_ids_returning_tuple, get_dist_bar


# noinspection SpellCheckingInspection

def load_pretrain_dataset(input_path):
    df = pd.read_csv(input_path, header=None)
    smiles_data = list(df[0])
    return smiles_data


def load_finetune_dataset(input_path, target):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']

    labels = input_df[target]
    # convert 0 to -1
    # labels = labels.replace(0, -1)
    # there are no nans
    labels = labels.fillna(-1)
    # print(labels)
    # print(labels.values)

    assert len(smiles_list) == len(labels)
    return smiles_list, labels.values
    # input_df = pd.read_csv(input_path, sep=',')
    # smiles_list = input_df['smiles']
    # rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #
    # labels = input_df[target]
    # # there are no nans
    # labels = labels.fillna(-1)
    #
    # assert len(smiles_list) == len(rdkit_mol_objs_list)
    # assert len(smiles_list) == len(labels)
    # return smiles_list, rdkit_mol_objs_list, labels.values


def load_cliff_dataset(input_path, split):
    input_df = pd.read_csv(input_path, sep=',')
    cliff_data = input_df[input_df['split'] == split]
    smiles_list = cliff_data['smiles'].values

    label_y = cliff_data['y'].values
    label_cliff = cliff_data['cliff_mol'].values

    assert len(smiles_list) == len(label_y)
    assert len(smiles_list) == len(label_cliff)

    return smiles_list, label_y, label_cliff


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles: smiles sequence
        include_chirality: Default=False

    Return:
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def ScaffoldSplitter(task_name, path, smiles_list):
    # path = '/mnt/8t/qjb/workspace/flute/project/finetuneData'

    frac_train, frac_valid, frac_test = 0.8, 0.1, 0.1
    # dataset = pickle.load(open(f'{path}/{task_name}.pkl', 'rb'))

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    # N = len(dataset)
    N = len(smiles_list)

    # smiles_list = [data['smiles'] for data in dataset]

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}

    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=False)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_valid) * N
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    print('train_idx=', train_idx)
    print('valid_idx=', valid_idx)
    print('test_idx=', test_idx)

    split_dict = {
        'train_idx': train_idx,
        'valid_idx': valid_idx,
        'test_idx': test_idx
    }
    pickle.dump(split_dict, open(f"{path}/split/scaffold/{task_name}.pkl", "wb"))

    # train_list, valid_list, test_list = [], [], []
    # for index, data in enumerate(dataset):
    #     if index in train_idx:
    #         train_list.append(data)
    #     elif index in valid_idx:
    #         valid_list.append(data)
    #     else:
    #         test_list.append(data)
    #
    # pickle.dump(train_list, open(f"{path}/split/{task_name}_train.pkl", "wb"))
    # pickle.dump(valid_list, open(f"{path}/split/{task_name}_valid.pkl", "wb"))
    # pickle.dump(test_list, open(f"{path}/split/{task_name}_test.pkl", "wb"))

    # return train_dataset, valid_dataset, test_dataset


def RandomScaffoldSplitter(task_name, path, smiles_list, seed):
    frac_train, frac_valid, frac_test = 0.8, 0.1, 0.1

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    N = len(smiles_list)

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind in range(N):
        scaffold = generate_scaffold(smiles_list[ind], include_chirality=False)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(frac_valid * len(smiles_list)))
    n_total_test = int(np.floor(frac_test * len(smiles_list)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    # print('train_idx=', train_idx)
    # print('valid_idx=', valid_idx)
    # print('test_idx=', test_idx)

    split_dict = {
        'train_idx': train_idx,
        'valid_idx': valid_idx,
        'test_idx': test_idx
    }
    pickle.dump(split_dict, open(f"{path}/split/random_scaffold/{task_name}/{task_name}_{seed}.pkl", "wb"))
    # return train_dataset, valid_dataset, test_dataset


def add_label():
    task_name = 'hiv'
    config = {
        'task_name': task_name,
        # 'path': f'/mnt/8t/qjb/workspace/SCAGE_DATA/finetune_data_loong/{task_name}/raw/{task_name}.csv'
        'path': f'/root/autodl-tmp/data/raw/{task_name}.csv'
    }

    config = get_downstream_task_names(config)
    print(config)

    input_df = pd.read_csv(config['path'], sep=',')
    smiles_list = input_df['smiles'].values

    labels = input_df[config['target']]
    labels = labels.fillna(-1).values

    processed_path = f'/root/autodl-tmp/data/processed/{task_name}.csv'
    processed_file = open(processed_path, 'r')
    processed_smiles = [line.strip() for line in processed_file.readlines()]
    # print(processed_smiles)

    save_path = f'/root/autodl-tmp/data/finetune_data_withlabel/{task_name}.csv'
    save_file = open(save_path, 'w')
    save_file.write(','.join(['smiles', *config['target']]) + '\n')
    for index, smiles in enumerate(tqdm(smiles_list)):
        if smiles in processed_smiles:
            save_file.write(','.join([smiles, *map(str, labels[index])]) + '\n')

    save_file.flush()
    save_file.close()


def split_data():
    # task_name_list = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity', ]
    task_name_list = ['hiv']
    for task_name in task_name_list:
        for seed in range(10):
            print(f'now processing {task_name} {seed}')
            config = {
                'task_name': task_name,
                'path': f'/root/autodl-tmp/data/finetune_data_withlabel/{task_name}.csv'
            }
            save_path = f'/root/autodl-tmp/data/'

            input_df = pd.read_csv(config['path'], sep=',')
            smiles_list = input_df['smiles'].values

            ScaffoldSplitter(task_name, save_path, smiles_list)
            # RandomScaffoldSplitter(task_name, save_path, smiles_list, seed)


# noinspection PyUnresolvedReferences
def _process_smiles_with_label(args):
    smiles, label = args
    mol = AllChem.MolFromSmiles(smiles)
    if mol is not None:
        data = mol_to_data_pkl(mol)  # Assuming this function exists and is provided elsewhere
        data['smiles'] = smiles
        data['label'] = label
        return data
    print_err(f"Invalid smiles: {smiles}")
    return None


def _process_cliff_with_label(args):
    smiles, label_y, label_cliff = args
    mol = AllChem.MolFromSmiles(smiles)
    if mol is not None:
        data = mol_to_data_pkl(mol)  # Assuming this function exists and is provided elsewhere
        data['smiles'] = smiles
        data['label_y'] = label_y
        data['label_cliff'] = label_cliff
        return data
    print_err(f"Invalid smiles: {smiles}")
    return None


# noinspection PyUnresolvedReferences
def _process_smiles_without_label(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    if mol is not None:
        data = mol_to_data_pkl(mol)  # Assuming this function exists and is provided elsewhere
        data['smiles'] = smiles
        return data
    print_err(f"Invalid smiles: {smiles}")
    return None


def process_finetune_data(task_name: str, csv_path: str, target_pkl_path: str, num_limit=None):
    target_dir = os.path.dirname(target_pkl_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # task_name_list = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity']
    task_name_list = ['hiv', 'muv']
    assert task_name in task_name_list
    config = {
        'task_name': task_name,
        'path': f'/mnt/8t/qjb/workspace/SCAGE_DATA/finetune_data_withlabel/{task_name}.csv'
        # 'path': csv_path
    }
    config = get_downstream_task_names(config)
    smiles_list, labels = load_finetune_dataset(config['path'], config['target'])
    if num_limit is not None:
        smiles_list = smiles_list[:num_limit]
        labels = labels[:num_limit]
    total_data = []
    for index, smiles in enumerate(tqdm(smiles_list)):
        mol = AllChem.MolFromSmiles(smiles)
        if mol is not None:
            data = mol_to_data_pkl(mol)
            data['smiles'] = smiles
            data['label'] = labels[index]
            total_data.append(data)

    # pickle.dump(total_data, open(f"/mnt/8t/qjb/workspace/SCAGE_DATA/finetune_data_loong/pkl/{task_name}.pkl", "wb"))
    with open(target_pkl_path, 'wb') as f:
        pickle.dump(total_data, f)


def process_more_finetune_data(task_name: str, target_pkl_path: str, num_cores=40, num_limit=None):
    config = yaml.load(open(pdir + '/config/config_finetune.yaml'), Loader=yaml.FullLoader)
    # config['lr_scheduler']['type'] = params_nni['lr_scheduler']
    config = config_current_user('wd-b100', config)  # TODO: CHANGE IT ACCORDING TO YOUR NEEDS
    config = config_dataset_form('pkl', config)
    config['task_name'] = task_name
    config = get_downstream_task_names(config)
    path_ = config['root'] + f"/{task_name}/{task_name}.csv"
    smiles_list, labels = load_finetune_dataset(path_, config['target'])
    if num_limit is not None:
        smiles_list = smiles_list[:num_limit]
        labels = labels[:num_limit]
    # use joblib to parallel process
    result = joblib.Parallel(n_jobs=num_cores)(
        joblib.delayed(mol_to_data_pkl)(smiles) for smiles in tqdm(smiles_list, desc='Processing finetune data')
    )
    # add label
    for i, item in enumerate(result):
        item['label'] = labels[i]
    with open(target_pkl_path, 'wb') as f:
        pickle.dump(result, f)


def process_finetune_data_parallel(task_name: str, csv_path: str, target_pkl_path: str, num_limit=None, num_cores=4):
    target_dir = os.path.dirname(target_pkl_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    task_name_list = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity']
    assert task_name in task_name_list
    GlobalVar.data_process_style = 'qjb'
    if GlobalVar.data_process_style == 'wd':
        config = {
            'task_name': task_name,
            'path': csv_path,
            'root': pdir + '/datasets/unpacked/finetune/csvs/raw'
        }
    else:
        config = {
            'task_name': task_name,
            'path': csv_path
        }
    config = get_downstream_task_names(config)  # Assuming this function exists and is provided elsewhere
    smiles_list, labels = load_finetune_dataset(config['path'], config['target'])
    # Assuming this function exists and is provided elsewhere

    if num_limit is not None:
        smiles_list = smiles_list[:num_limit]
        labels = labels[:num_limit]

    args = list(zip(smiles_list, labels))

    # Specify the number of CPU cores with max_workers
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(_process_smiles_with_label, args), total=len(args)))

    total_data = [result for result in results if result is not None]
    with open(target_pkl_path, 'wb') as f:
        pickle.dump(total_data, f)


def _merge_files(files_basedir: str, delete_original_file=False):
    if files_basedir.endswith('/'):
        files_basedir = files_basedir[:-1]
    total_block_num = len([
        name for name in os.listdir(files_basedir)
        if os.path.isfile(os.path.join(files_basedir, name)) and name.startswith('block_')
    ])
    loguru.logger.info(f'Total block number: {total_block_num}')
    result_all = []
    for block_index in tqdm(range(total_block_num), desc='Merging files'):
        with open(f'{files_basedir}/block_{block_index}.pkl', 'rb') as f:
            result_block = pickle.load(f)
            result_all.extend(result_block)
    if not os.path.exists(f'{files_basedir}/merged/'):
        os.makedirs(f'{files_basedir}/merged/')
    with open(f'{files_basedir}/merged/all.pkl', 'wb') as f:
        pickle.dump(result_all, f)
    if delete_original_file:
        loguru.logger.warning('Deleting original files. If you want to cancel, press Ctrl+C in 5 seconds.')
        time.sleep(5)
        for block_index in tqdm(range(total_block_num), desc='Deleting original files'):
            os.remove(f'{files_basedir}/block_{block_index}.pkl')


def process_pm6_83m_sample_2m_dataset(
        filepath: str = '/home/wangding/data/datasets/pretrain/pm6_83m_smiles_ramdom_2M.pkl',
        num_limit: int = None,
        block_size: int = 10_0000,
        save_dir: str = '/home/wangding/data/datasets/pretrain/dumped/',
        num_cores: int = 40,
        auto_merge_files: bool = False,
        delete_original_file: bool = False,
):
    # list of smiles str
    with open(filepath, 'rb') as f:
        smiles_list = pickle.load(f)
    smiles_list = smiles_list[:num_limit] if num_limit is not None else smiles_list
    total_block_num = math.ceil(len(smiles_list) / block_size)
    for block_index in range(total_block_num):
        block_start = block_index * block_size
        block_end = min((block_index + 1) * block_size, len(smiles_list))
        block_smiles_list = smiles_list[block_start:block_end]
        # using joblib to parallel process
        result_block = joblib.Parallel(n_jobs=num_cores)(
            joblib.delayed(_process_smiles_without_label)(smiles)
            for smiles in tqdm(block_smiles_list, desc=f'Processing block {block_index}/{total_block_num}')
        )
        # save to pkl
        with open(f'{save_dir}/block_{block_index}.pkl', 'wb') as f:
            pickle.dump(result_block, f)
    if auto_merge_files:
        _merge_files(save_dir, delete_original_file=delete_original_file)


# def send_feishu(title_: str = None, content_: str = None, url_: Union[str, Path] = None, with_machine_info: bool = True,
#                 pre_packed_msg: FeiShuMessage = None):
# @send_feishu(title_="Cliff Data Process", content_="Finished", with_machine_info=True)
def process_cliff_data_parallel(task_name: str, csv_path: str, mode: str, target_pkl_path: str, num_limit=None,
                                num_cores=4):
    target_dir = os.path.dirname(target_pkl_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    smiles_list, label_y, label_cliff = load_cliff_dataset(csv_path, mode)
    # Assuming this function exists and is provided elsewhere

    if num_limit is not None:
        smiles_list = smiles_list[:num_limit]
        label_y = label_y[:num_limit]
        label_cliff = label_cliff[:num_limit]

    args = list(zip(smiles_list, label_y, label_cliff))

    # Specify the number of CPU cores with max_workers
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(_process_cliff_with_label, args), total=len(args)))

    total_data = [result for result in results if result is not None]
    with open(target_pkl_path, 'wb') as f:
        pickle.dump(total_data, f)


def auto_read_list(src_file, key_name='smiles'):
    # read a txt or csv file
    if src_file.endswith('.txt'):
        with open(src_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    elif src_file.endswith('.csv'):
        return pd.read_csv(src_file, header=None)[0].tolist()
    elif src_file.endswith('.pkl'):
        try:
            items = [one[key_name] for one in pickle.load(open(src_file, 'rb'))]
            return items
        except Exception:
            raise ValueError(f"Unsupported file format(pkl): {src_file}")
    else:
        raise ValueError(f"Unsupported file format: {src_file}")


def process_pretrain_data(src_path: str, target_path: str, num_limit=None, num_cores=20):
    target_dir = os.path.dirname(target_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    smiles_list = auto_read_list(src_path)
    print_info("Length of smiles_list: ", len(smiles_list))
    if num_limit is not None:
        smiles_list = smiles_list[:num_limit]
    print_info("Length of smiles_list after limit: ", len(smiles_list))
    # Specify the number of CPU cores with max_workers
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(_process_smiles_without_label, smiles_list), total=len(smiles_list)))

    total_data = [result for result in results if result is not None]
    with open(target_path, 'wb') as f:
        pickle.dump(total_data, f)


# 171, 9MB
# 1710, 90MB
# 17100, 900MB
# 171000, 9GB

# noinspection PyUnresolvedReferences
@deprecated
def process_smiles(smiles_data):
    index, smiles, dump_dir = smiles_data
    mol = AllChem.MolFromSmiles(smiles)
    if mol is not None:
        data = mol_to_data_pkl(mol)
        data['smiles'] = smiles
        with open(Path(dump_dir) / f'data_{index}.pkl', "wb") as f:
            pickle.dump(data, f)


def add_new_fg():
    from compound_tools import get_all_matched_fn_ids_returning_tuple
    task_name = 'bbbp'
    with open(f'/mnt/8t/qjb/workspace/SCAGE_DATA/finetune_data_loong/pkl/{task_name}/{task_name}.pkl', 'rb') as f:
        data = pickle.load(f)
        for i, item in enumerate(tqdm(data)):
            try:
                del item['function_group_bond']
            except:
                print('no function_group_bond')
            smiles = item['smiles']
            mol = AllChem.MolFromSmiles(smiles)
            function_group_index, function_group_bond_index = get_all_matched_fn_ids_returning_tuple(mol, item['edges'])
            # data['function_group_index'] = np.array(function_group_index, 'int64')
            item['function_group_index'] = function_group_index
            item['function_group_bond_index'] = np.array(function_group_bond_index, 'int64')
            data[i] = item
        pickle.dump(data, open(f"/mnt/8t/qjb/workspace/SCAGE_DATA/finetune_data_loong/pkl/{task_name}_new.pkl", "wb"))


def from_pkl_to_lmdb(pkl_path, db_path):
    env = lmdb.open(db_path, map_size=1024 * 1024 * 1024 * 1024, subdir=False, lock=False)
    txn = env.begin(write=True)
    keys = []

    pkl_data = pickle.load(open(pkl_path, "rb"))

    for idx, data in tqdm(enumerate(pkl_data)):
        data = pickle.dumps(data)
        # data = pickle.load(file)
        # buffer = io.BytesIO(file)
        # data = torch.load(file)
        # print(data)
        keys.append(str(idx).encode())
        txn.put(str(idx).encode(), data)

    txn.commit()
    # print(keys)
    with env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', str(len(keys)).encode())


def add_spatial_pos_bar(data):
    data['spatial_pos_bar'] = get_dist_bar(data['spatial_pos'])
    return data


def add_bond_angles(data):
    data['bond_angles'] = Compound3DKit.get_bond_angles(data['atom_pos'], data['angles_atom_index'])
    return data



def change_distance_to_bin(data):
    data['pair_distances'] = binning_matrix(data['pair_distances'], 30, 0, 30)
    return data


def change_angle_to_bin(data):
    # bond_angles = Compound3DKit.get_bond_angles(data['atom_pos'], data['angles_atom_index'])
    data['bond_angles'] = binning_matrix(data['bond_angles'], 20, 0, math.pi)
    return data


def change_functiongroup(data):
    mol = AllChem.MolFromSmiles(data['smiles'])
    function_group_index, function_group_bond_index = get_all_matched_fn_ids_returning_tuple(mol, data['edges'])
    data['function_group_index'] = function_group_index
    data['function_group_bond_index'] = np.array(function_group_bond_index, 'int64')
    return data



