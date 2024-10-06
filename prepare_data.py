import argparse
import multiprocessing
import os
import pickle
import zipfile
from pathlib import Path
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor

import lmdb
import numpy as np
from rdkit.Chem import AllChem
from echo_logger import *

import pandas as pd
from echo_logger import print_info
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from _config import pdir
from data_process.utils_dataprocess import process_smiles
from data_process.compound_tools import mol_to_data_pkl
from _config import get_downstream_task_names

DATASET_DIR = Path(pdir) / 'datasets'


def unzip_datasets(dataset_type: str = 'finetune'):
    assert dataset_type in ['finetune']
    zip_path = DATASET_DIR / f"{dataset_type}_datasets.zip"
    assert zip_path.exists()
    target_dir = DATASET_DIR / 'unpacked' / dataset_type / 'csvs' / 'raw'
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print_info(f"unzip {zip_path} to {target_dir}")


def get_smiles_list_from_csv(csv_file: Path):
    import pandas as pd
    df = pd.read_csv(csv_file)
    return df['smiles'].tolist()


def process_pretrain_data_multi_core(raw_file, dump_dir, range_: Tuple[int, int] = None, num_processes=None):
    base_index = range_[0]
    smiles_list: List[str] = get_smiles_list_from_csv(raw_file)
    if range_ is not None:
        smiles_list = smiles_list[range_[0]: range_[1] if range_[1] < len(smiles_list) else len(smiles_list)]
    dump_d = Path(dump_dir)
    if not dump_d.exists():
        dump_d.mkdir(parents=True)
    if num_processes is None:
        for index, smiles in enumerate(tqdm(smiles_list)):
            process_smiles((index + base_index, smiles, dump_dir))
    else:
        smiles_data = [(index + base_index, smiles, dump_dir) for index, smiles in enumerate(smiles_list)]
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Using starmap if function takes multiple arguments, but here it's packed into a single argument
            list(tqdm(pool.imap(process_smiles, smiles_data), total=len(smiles_data)))

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

def _process_smiles_without_label(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    if mol is not None:
        data = mol_to_data_pkl(mol)  # Assuming this function exists and is provided elsewhere
        data['smiles'] = smiles
        return data
    print_err(f"Invalid smiles: {smiles}")
    return None

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

def process_finetune_data(task_name: str, csv_path: str, target_pkl_path: str, num_limit=None):
    target_dir = os.path.dirname(target_pkl_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    task_name_list = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity']
    assert task_name in task_name_list
    config = {
        'task_name': task_name,
        'path': f'{csv_path}/{task_name}.csv'
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

    with open(f'{target_pkl_path}/{task_name}.pkl', 'wb') as f:
        pickle.dump(total_data, f)

def process_finetune_data_parallel(task_name: str, csv_path: str, target_pkl_path: str, num_limit=None, num_cores=20):
    target_dir = os.path.dirname(target_pkl_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    task_name_list = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity']
    assert task_name in task_name_list
    config = {
        'task_name': task_name,
        'path': f'{csv_path}/{task_name}.csv'
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
    with open(f'{target_pkl_path}/{task_name}.pkl', 'wb') as f:
        pickle.dump(total_data, f)

def load_cliff_dataset(input_path, split):
    input_df = pd.read_csv(input_path, sep=',')
    cliff_data = input_df[input_df['split'] == split]
    smiles_list = cliff_data['smiles'].values

    label_y = cliff_data['y'].values
    label_cliff = cliff_data['cliff_mol'].values

    assert len(smiles_list) == len(label_y)
    assert len(smiles_list) == len(label_cliff)

    return smiles_list, label_y, label_cliff

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
    with open(f'{target_pkl_path}/{task_name}.pkl', 'wb') as f:
        pickle.dump(total_data, f)

def from_pkl_to_lmdb(pkl_path, lmdb_path, start_idx=0):
    env = lmdb.open(lmdb_path, map_size=1024 * 1024 * 1024 * 1024, subdir=False, lock=False)
    txn = env.begin(write=True)

    try:
        keys = pickle.loads(txn.get(b'__keys__'))
    except:
        keys = []

    pkl_data = pickle.load(open(pkl_path, "rb"))

    for idx, data in tqdm(enumerate(pkl_data, start_idx)):
        data = pickle.dumps(data)

        keys.append(str(idx).encode())
        txn.put(str(idx).encode(), data)

    txn.commit()
    with env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', str(len(keys)).encode())

    env.close()

    return len(keys)

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

def split_data():
    task_name_list = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity', ]
    for task_name in task_name_list:
        print(f'now processing {task_name}')
        config = {
            'task_name': task_name,
            'path': f'./data/mpp/raw/{task_name}.csv'
        }
        save_path = f'./data/mpp/split/'

        input_df = pd.read_csv(config['path'], sep=',')
        smiles_list = input_df['smiles'].values

        ScaffoldSplitter(task_name, save_path, smiles_list)


def main():
    parser = argparse.ArgumentParser(description='Preprocessing of data')
    parser.add_argument('--taskname', type=str, default="pretrain", help='data root')
    parser.add_argument('--dataroot', type=str, default="./data/pretrain/pretrain1000.txt",  # for test
                        help='data root')
    parser.add_argument('--datatarget', type=str, default="./data/pretrain/pretrain_data.lmdb",
                        help='dataset name, e.g. data')
    args = parser.parse_args()

    if args.taskname == 'pretrain':
        src_file = f'{args.dataroot}'
        target_file = f'{args.datatarget}'
        process_pretrain_data(src_file, target_file.replace('.lmdb', '.pkl'))
        from_pkl_to_lmdb(target_file.replace('.lmdb', '.pkl'), target_file)
    elif args.taskname in ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity']:
        csv_file = f'{args.dataroot}'
        target_pkl_path = f'{args.datatarget}/'
        process_finetune_data(args.taskname, csv_file, target_pkl_path)
    else:
        csv_file = f'{args.dataroot}'
        target_pkl_path = f'{args.datatarget}/'
        process_finetune_data(args.taskname, csv_file, target_pkl_path)


if __name__ == '__main__':
    main()
    # unzip_datasets()

    # task_name = ''
    # src_file = './data/pretrain/pubchem_sample_100.txt'
    # target_file = './data/pretrain/pubchem_sample_100.pkl'
    # process_pretrain_data(src_file, target_file, num_cores=10)

    # csv_file = './data/mpp/'
    # target_pkl_path = './data/mpp/'
    # process_finetune_data_parallel('bace', csv_file, target_pkl_path)
