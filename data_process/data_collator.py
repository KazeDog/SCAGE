import random
from copy import deepcopy
from typing import List

import numpy as np
import torch
from echo_logger import *

from data_process.masks import *
from data_process.paddings import *
from naive_fg import all_possible_fg_nums
from utils.global_var_util import GlobalVar
from .compound_tools import CompoundKit, Compound3DKit, get_dist_bar
from .function_group_constant import nfg
import random
from copy import deepcopy
from typing import List

import numpy as np
import torch
from echo_logger import *

from data_process.masks import *
from data_process.paddings import *
from naive_fg import all_possible_fg_nums
from utils.global_var_util import GlobalVar
from .compound_tools import CompoundKit, Compound3DKit, get_dist_bar
from .function_group_constant import nfg


atom_id_names = set(list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names)
additional_popping_attributes = {
    'pair_distances', 'triple_angles', 'atom_pos', 'edges', 'attention_mask',
    'label', 'angel_mask', 'atom_length', 'angles_atom_index', 'angles_bond_index', 'morgan2048_fp',
    'bond_angles', 'edge_distances', 'edge_types', 'atom_dist_bar', 'edge_dist_bar', 'atom_bond_dist_bar',
    'spatial_pos_bar', 'atom_bond_distances', 'text_name_embedding',
    'function_group_index', 'function_group_bond_index', 'spatial_pos', 'pair_distances_bin', 'bond_angles_bin',
    'label_y', 'label_cliff', 'fg_number', 'bond_distances'
}
none_int_names = {
    'van_der_waals_radis', 'partial_charge', 'mass', 'atom_pos', 'pair_distances',
    'triple_angles', 'bond_distances', 'bond_angles', 'edge_distances', 'atom_dist_bar',
    'edge_dist_bar', 'atom_bond_dist_bar', 'atom_bond_distances', 'text_name_embedding', 'label_y', 'fg_atom_count',
    'bond_distances', 'label'
}
non_numpy_names = {
    'smiles',  # str
    'angles_atom_index',  # list
    'angles_bond_index'  # list
}


# @profile
def collator_finetune_pkl(items: List[Dict[str, Union[List[int], np.ndarray, str, int]]]):
    """
    Args:
        items: list of length batch_size, each item is a dict.

    Returns:

    """
    max_len = max([len(item['atomic_num']) for item in items])
    max_edges = max([len(item['edges']) for item in items])
    max_angels = max([len(item['angles_atom_index']) for item in items])
    if GlobalVar.debug_dataloader_collator:
        print("items[0]['smiles']", items[0]['smiles'])
    item_pop = ['spatial_pos', 'pair_distances_bin', 'bond_angles_bin']
    for i in items[0].keys():
        if i not in atom_id_names and i not in additional_popping_attributes:
            item_pop.append(i)

    for item in items:
        # print(item.keys())
        for i in item_pop:
            if item.get(i) is not None:
                item.pop(i)
        data_len = len(item['atomic_num'])
        edge_len = len(item['edges'])

        item['atom_length'] = data_len
        item['edge_length'] = edge_len

        if GlobalVar.is_mol_net_tasks:
            item['bond_angles'] = padding_1d_sequence(
                Compound3DKit.get_angles_list(item['bond_angles'], item['angles_bond_index']), max_angels, -5)
            # item['bond_angles'] = padding_1d_sequence(
            #     binning_matrix(Compound3DKit.get_angles_list(item['bond_angles'], item['angles_bond_index']), 12,
            #                    -math.pi, math.pi), max_angels, -5)
        else:
            item['bond_angles'] = padding_1d_sequence(item['bond_angles'], max_angels, -5)
        item['angles_atom_index'] = padding_2d_loong(item['angles_atom_index'], max_angels, 3)
        # print(item['angles_bond_index'])
        item['angles_bond_index'] = padding_2d_loong(item['angles_bond_index'], max_angels, 2)

        for name in atom_id_names:
            item[name] = padding_1d_sequence(item[name], max_len, -1) + 1

        item['atom_dist_bar'] = get_dist_bar(item['pair_distances'], GlobalVar.dist_bar)
        item['pair_distances'] = padding_2d_sequence(item['pair_distances'], max_len, 0)
        item['edge_distances'] = padding_2d_sequence(item['edge_distances'], max_edges, 0)
        item['edge_types'] = padding_1d_sequence(item['edge_types'], max_edges, -1) + 1

        item['bond_distances'] = padding_pair_distances_loong(item['pair_distances'], item['edges'], max_edges, 0)
        item['atom_bond_distances'] = padding_atom_bond_distances(item['atom_bond_distances'], max_len, max_edges, 0)

        item['function_group_index'] = padding_function_group_index(item['function_group_index'], max_len)
        item['function_group_bond_index'] = padding_1d_sequence(item['function_group_bond_index'], max_edges, -1)

        item['atom_pos'] = padding_atoms_pos(item['atom_pos'], max_len, 0)

        item['edges'] = padding_2d_loong(item['edges'], max_edges, 2)

        item["atom_attention_mask"] = attention_mask_using_0(data_len + 1, max_len + 1, 0)
        item["bond_attention_mask"] = attention_mask_using_0(edge_len + 1, max_edges + 1, 0)
        item["atom_bond_attention_mask"] = attention_mask_atom_bond(data_len + 1, edge_len + 1, max_len + 1,
                                                                    max_edges + 1, 0)

        item["atom_mask"] = atom_mask(data_len + 1, max_len + 1, 0)

    # concat all the data into a dict
    data = {}
    for name in items[0].keys():
        data[name] = np.array([item[name] for item in items])
        if name not in none_int_names:
            data[name] = torch.tensor(data[name], dtype=torch.int)
        else:
            data[name] = torch.tensor(data[name], dtype=torch.float)

    return data


def collator_pretrain_pkl_bin(items: List[Dict[str, Union[List[int], np.ndarray, str, int]]]):
    """
    Args:
        items: list of length batch_size, each item is a dict.

    Returns:

    """
    max_len = max([len(item['atomic_num']) for item in items])
    max_edges = max([len(item['edges']) for item in items])
    max_angels = max([len(item['angles_atom_index']) for item in items])
    if GlobalVar.debug_dataloader_collator:
        print("items[0]['smiles']", items[0]['smiles'])
    item_pop = ['label', 'atom_bond_dist_bar']
    for i in items[0].keys():
        if i not in atom_id_names and i not in additional_popping_attributes:
            item_pop.append(i)

    for item in items:
        # print(item.keys())
        for i in item_pop:
            if item.get(i) is not None:
                item.pop(i)
        data_len = len(item['atomic_num'])
        edge_len = len(item['edges'])

        item['atom_length'] = data_len
        item['edge_length'] = edge_len

        item['bond_angles'] = padding_1d_sequence(item['bond_angles'], max_angels, -5)

        item['angles_atom_index'] = padding_2d_loong(item['angles_atom_index'], max_angels, 3)
        # print(item['angles_bond_index'])
        item['angles_bond_index'] = padding_2d_loong(item['angles_bond_index'], max_angels, 2)

        for name in atom_id_names:
            item[name] = padding_1d_sequence(item[name], max_len, -1) + 1
        item['atom_dist_bar'] = get_dist_bar(item['pair_distances'], GlobalVar.dist_bar)
        item['pair_distances'] = padding_2d_sequence(item['pair_distances'], max_len, 0)
        item['pair_distances_bin'] = padding_2d_sequence(item['pair_distances_bin'], max_len, 0)

        item['bond_angles'] = padding_1d_sequence(item['bond_angles'], max_angels, -5)
        item['bond_angles_bin'] = padding_1d_sequence(item['bond_angles_bin'], max_angels, -5)

        item['edge_distances'] = padding_2d_sequence(item['edge_distances'], max_edges, 0)

        item['spatial_pos'] = padding_2d_sequence(item['spatial_pos'], max_len, -1)


        item['edge_types'] = padding_1d_sequence(item['edge_types'], max_edges, -1) + 1

        item['bond_distances'] = padding_1d_sequence(item['bond_distances'], max_edges, 0)
        item['atom_bond_distances'] = padding_atom_bond_distances(item['atom_bond_distances'], max_len, max_edges, 0)


        item['function_group_index'] = padding_function_group_index(item['function_group_index'], max_len)
        item['function_group_bond_index'] = padding_1d_sequence(item['function_group_bond_index'], max_edges, -1)

        item['atom_pos'] = padding_atoms_pos(item['atom_pos'], max_len, 0)

        item['edges'] = padding_2d_loong(item['edges'], max_edges, 2)

        item["atom_attention_mask"] = attention_mask_using_0(data_len + 1, max_len + 1, 0)
        item["bond_attention_mask"] = attention_mask_using_0(edge_len + 1, max_edges + 1, 0)
        item["atom_bond_attention_mask"] = attention_mask_atom_bond(data_len + 1, edge_len + 1, max_len + 1,
                                                                    max_edges + 1, 0)


    # concat all the data into a dict
    data = {}
    for name in items[0].keys():
        data[name] = np.array([item[name] for item in items])
        if name not in none_int_names:
            data[name] = torch.tensor(data[name], dtype=torch.int)
        else:
            data[name] = torch.tensor(data[name], dtype=torch.float)
    if 'fg_number' in GlobalVar.pretrain_task:
        assert 'fg_number' in data.keys()
    return data

def collator_cliff_pkl(items: List[Dict[str, Union[List[int], np.ndarray, str, int]]]):
    """
    Args:
        items: list of length batch_size, each item is a dict.

    Returns:

    """
    max_len = max([len(item['atomic_num']) for item in items])
    max_edges = max([len(item['edges']) for item in items])
    max_angels = max([len(item['angles_atom_index']) for item in items])
    if GlobalVar.debug_dataloader_collator:
        print("items[0]['smiles']", items[0]['smiles'])
    item_pop = ['label', 'atom_bond_dist_bar']
    for i in items[0].keys():
        if i not in atom_id_names and i not in additional_popping_attributes:
            item_pop.append(i)

    for item in items:
        # print(item.keys())
        smiles = item['smiles']
        for i in item_pop:
            try:
                item.pop(i)
            except:
                continue
        data_len = len(item['atomic_num'])
        edge_len = len(item['edges'])

        item['atom_length'] = data_len
        item['edge_length'] = edge_len

        item['bond_angles'] = padding_1d_sequence(item['bond_angles'], max_angels, -5)

        item['angles_atom_index'] = padding_2d_loong(item['angles_atom_index'], max_angels, 3)

        item['angles_bond_index'] = padding_2d_loong(item['angles_bond_index'], max_angels, 2)

        for name in atom_id_names:
            item[name] = padding_1d_sequence(item[name], max_len, -1) + 1

        item['atom_dist_bar'] = get_dist_bar(item['pair_distances'], GlobalVar.dist_bar)
        item['pair_distances'] = padding_2d_sequence(item['pair_distances'], max_len, 0)
        item['pair_distances_bin'] = padding_2d_sequence(item['pair_distances_bin'], max_len, 0)

        item['bond_angles'] = padding_1d_sequence(item['bond_angles'], max_angels, -5)
        item['bond_angles_bin'] = padding_1d_sequence(item['bond_angles_bin'], max_angels, -5)

        item['edge_distances'] = padding_2d_sequence(item['edge_distances'], max_edges, 0)
        item['spatial_pos'] = padding_2d_sequence(item['spatial_pos'], max_len, -1)
        item['spatial_pos_bar'] = get_dist_bar(item['spatial_pos'], GlobalVar.dist_bar)

        item['edge_types'] = padding_1d_sequence(item['edge_types'], max_edges, -1) + 1

        item['bond_distances'] = padding_pair_distances_loong(item['pair_distances'], item['edges'], max_edges, 0)
        item['atom_bond_distances'] = padding_atom_bond_distances(item['atom_bond_distances'], max_len, max_edges, 0)

        item['function_group_index'] = padding_function_group_index(item['function_group_index'], max_len)
        item['function_group_bond_index'] = padding_1d_sequence(item['function_group_bond_index'], max_edges, -1)

        item['atom_pos'] = padding_atoms_pos(item['atom_pos'], max_len, 0)

        item['edges'] = padding_2d_loong(item['edges'], max_edges, 2)

        item["atom_attention_mask"] = attention_mask_using_0(data_len + 1, max_len + 1, 0)
        item["bond_attention_mask"] = attention_mask_using_0(edge_len + 1, max_edges + 1, 0)
        item["atom_bond_attention_mask"] = attention_mask_atom_bond(data_len + 1, edge_len + 1, max_len + 1,
                                                                    max_edges + 1, 0)

        item['function_group_index_mask'] = fg_mask(data_len, max_len, nfg() + 1, 0)

    # concat all the data into a dict
    data = {}
    for name in items[0].keys():
        data[name] = np.array([item[name] for item in items])
        if name not in none_int_names:
            data[name] = torch.tensor(data[name], dtype=torch.int)
        else:
            data[name] = torch.tensor(data[name], dtype=torch.float)

    return data
