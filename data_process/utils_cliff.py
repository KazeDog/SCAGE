import os
import pickle
from unittest import TestCase

import math
import torch
from tqdm import tqdm

from data_process import algos
from data_process.compound_tools import Compound3DKit, binning_matrix, get_dist_bar


# from data_process.data_collator import set_up_spatial_pos


def add_spatial_pos(data):
    data_len = len(data['atomic_num'])
    adj = torch.zeros([data_len, data_len], dtype=torch.bool)
    adj[data['edges'][:, 0], data['edges'][:, 1]] = True
    adj[data['edges'][:, 1], data['edges'][:, 0]] = True
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    spatial_pos = shortest_path_result
    # spatial_pos = torch.from_numpy(shortest_path_result).long()
    # spatial_pos = set_up_spatial_pos(spatial_pos, up=20)
    data['spatial_pos'] = spatial_pos
    return data

def add_spatial_pos_bar(data):
    data['spatial_pos_bar'] = get_dist_bar(data['spatial_pos'])
    return data


def add_bond_angles(data):
    data['bond_angles'] = Compound3DKit.get_bond_angles(data['atom_pos'], data['angles_atom_index'])
    return data


def change_distance_to_bin(data):
    data['pair_distances_bin'] = binning_matrix(data['pair_distances'], 30, 0, 30)
    return data


def change_angle_to_bin(data):
    # bond_angles = Compound3DKit.get_bond_angles(data['atom_pos'], data['angles_atom_index'])
    data['bond_angles_bin'] = binning_matrix(data['bond_angles'], 20, 0, math.pi)
    return data


class TestProcessData(TestCase):
    def test_read_pkl(self):
        task_name = 'bbbp'
        # task_name_list = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity']
        # for task_name in task_name_list:
        with open(f'/root/autodl-tmp/data/cliff_data/cliff_pkl/CHEMBL204_Ki_test.pkl', 'rb') as f:
            # with open('/root/autodl-tmp/data/cliff_data/CHEMBL2971_Ki/pkl/CHEMBL2971_Ki_train.pkl', 'rb') as f:
            datas = pickle.load(f)
            print(datas[0])

    def test_add_spatial_pos(self):
        cliff_path = '/root/autodl-tmp/data/cliff_data/cliff_pkl'
        task_name_list = os.listdir(cliff_path)
        print(task_name_list)
        # task_name_list = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity']
        progress_bar = tqdm(task_name_list, leave=False, ascii=True, position=0)
        for task_name in progress_bar:
            with open(f'/root/autodl-tmp/data/cliff_data/cliff_pkl/{task_name}', 'rb') as f:
                data = pickle.load(f)
                for i, item in enumerate(tqdm(data, position=1)):
                    item = add_spatial_pos(item)
                    data[i] = item
            with open(f"/root/autodl-tmp/data/cliff_data/cliff_pkl/{task_name}", "wb") as f:
                pickle.dump(data, f)

            progress_bar.set_description(f'now processing {task_name}', refresh=True)

    def test_add_spatial_pos_bar(self):
        cliff_path = '/root/autodl-tmp/data/cliff_data/cliff_pkl'
        task_name_list = os.listdir(cliff_path)
        print(task_name_list)
        # task_name_list = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity']
        progress_bar = tqdm(task_name_list, leave=False, ascii=True, position=0)
        for task_name in progress_bar:
            with open(f'/root/autodl-tmp/data/cliff_data/cliff_pkl/{task_name}', 'rb') as f:
                data = pickle.load(f)
                for i, item in enumerate(tqdm(data, position=1)):
                    item = add_spatial_pos_bar(item)
                    data[i] = item
            with open(f"/root/autodl-tmp/data/cliff_data/cliff_pkl/{task_name}", "wb") as f:
                pickle.dump(data, f)

            progress_bar.set_description(f'now processing {task_name}', refresh=True)

    def test_add_bond_angles(self):
        cliff_path = '/root/autodl-tmp/data/cliff_data/cliff_pkl'
        task_name_list = os.listdir(cliff_path)
        # task_name_list = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity']
        progress_bar = tqdm(task_name_list, leave=False, ascii=True, position=0)
        for task_name in progress_bar:
            with open(f'/root/autodl-tmp/data/cliff_data/cliff_pkl/{task_name}', 'rb') as f:
                data = pickle.load(f)
                for i, item in enumerate(tqdm(data, position=1)):
                    item = add_bond_angles(item)
                    data[i] = item
            with open(f"/root/autodl-tmp/data/cliff_data/cliff_pkl/{task_name}", "wb") as f:
                pickle.dump(data, f)

            progress_bar.set_description(f'now processing {task_name}', refresh=True)

    def test_change_distance_to_bin(self):
        cliff_path = '/root/autodl-tmp/data/cliff_data/cliff_pkl'
        task_name_list = os.listdir(cliff_path)
        # task_name_list = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity']
        progress_bar = tqdm(task_name_list, leave=False, ascii=True, position=0)
        for task_name in progress_bar:
            with open(f'/root/autodl-tmp/data/cliff_data/cliff_pkl/{task_name}', 'rb') as f:
                data = pickle.load(f)
                for i, item in enumerate(tqdm(data, position=1)):
                    item = change_distance_to_bin(item)
                    data[i] = item
            with open(f"/root/autodl-tmp/data/cliff_data/cliff_pkl/{task_name}", "wb") as f:
                pickle.dump(data, f)

            progress_bar.set_description(f'now processing {task_name}', refresh=True)

    def test_change_angle_to_bin(self):
        cliff_path = '/root/autodl-tmp/data/cliff_data/cliff_pkl'
        task_name_list = os.listdir(cliff_path)
        # task_name_list = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipophilicity']
        progress_bar = tqdm(task_name_list, leave=False, ascii=True, position=0)
        for task_name in progress_bar:
            with open(f'/root/autodl-tmp/data/cliff_data/cliff_pkl/{task_name}', 'rb') as f:
                data = pickle.load(f)
                for i, item in enumerate(tqdm(data, position=1)):
                    item = change_angle_to_bin(item)
                    data[i] = item
            with open(f"/root/autodl-tmp/data/cliff_data/cliff_pkl/{task_name}", "wb") as f:
                pickle.dump(data, f)

            progress_bar.set_description(f'now processing {task_name}', refresh=True)
