import sys
from functools import cache
from unittest import TestCase

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from echo_logger import *

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from _config import pdir
from data_process.compound_tools import get_all_matched_fn_ids_returning_tuple, str_one_fg_matches
from data_process.function_group_constant import FUNCTION_GROUP_LIST_FROM_DAYLIGHT
from data_process.utils_dataprocess import auto_read_list
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as mpl
from rdkit import Chem
from tqdm import tqdm
import json


def inspect_fg_groups_and_edge_scores(smiles_list_, num_limit=None, save_filepath=None):
    mpl.rcParams['text.usetex'] = False
    fg_status_dict = {one: (f'ID: {index + 1}', 0) for index, one in enumerate(FUNCTION_GROUP_LIST_FROM_DAYLIGHT)}
    fg_edge_score_dict = {}
    total_num = min(len(smiles_list_), num_limit) if num_limit else len(smiles_list_)

    smiles_list_ = smiles_list_[:total_num]
    for index, smiles in enumerate(tqdm(smiles_list_)):
        mol = Chem.MolFromSmiles(smiles)
        l_part_new_dict, r_part_new_dict = str_one_fg_matches(get_all_matched_fn_ids_returning_tuple(mol))
        for _, fg_smiles_list in l_part_new_dict.items():
            for fg_smiles in fg_smiles_list:
                fg_status_dict[fg_smiles] = fg_status_dict.get(fg_smiles, ("None", 0))[0], \
                    fg_status_dict.get(fg_smiles, ("None", 0))[1] + 1
        for _, fg_edge_score in r_part_new_dict.items():
            fg_edge_score_dict[fg_edge_score] = fg_edge_score_dict.get(fg_edge_score, 0) + 1
    fg_edge_score_dict['total_num'] = total_num
    fg_status_dict['total_num'] = total_num
    print_info(dumps_json(fg_status_dict))
    print_info(dumps_json(fg_edge_score_dict))
    if save_filepath:
        with open(save_filepath, 'w') as file:
            json.dump([fg_status_dict, fg_edge_score_dict], file)
    return fg_status_dict, fg_edge_score_dict


def inspect_fg_groups_and_edge_scores_parallel(smiles_list_, num_limit=None, save_filepath=None, num_workers=13,
                                               chunk_size=1399):
    mpl.rcParams['text.usetex'] = False
    fg_status_dict = {one: (f'ID: {index + 1}', 0) for index, one in enumerate(FUNCTION_GROUP_LIST_FROM_DAYLIGHT)}
    fg_edge_score_dict = {}
    total_num = min(len(smiles_list_), num_limit) if num_limit else len(smiles_list_)

    smiles_list_ = smiles_list_[:total_num]

    # Break the data into chunks

    def process_chunk(smiles_chunk):
        local_fg_status_updates = {}
        local_fg_edge_score_updates = {}
        for smiles in smiles_chunk:
            mol = Chem.MolFromSmiles(smiles)
            l_part_new_dict, r_part_new_dict = str_one_fg_matches(get_all_matched_fn_ids_returning_tuple(mol))

            for _, fg_smiles_list in l_part_new_dict.items():
                for fg_smiles in fg_smiles_list:
                    if fg_smiles in local_fg_status_updates:
                        local_fg_status_updates[fg_smiles] = local_fg_status_updates[fg_smiles][0], \
                            local_fg_status_updates[fg_smiles][1] + 1
                    else:
                        local_fg_status_updates[fg_smiles] = fg_status_dict.get(fg_smiles, ("None", 0))[0], 1
            for _, fg_edge_score in r_part_new_dict.items():
                if fg_edge_score in local_fg_edge_score_updates:
                    local_fg_edge_score_updates[fg_edge_score] += 1
                else:
                    local_fg_edge_score_updates[fg_edge_score] = 1
        return local_fg_status_updates, local_fg_edge_score_updates

    # Parallel processing with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Process each chunk in parallel
        chunks = [smiles_list_[i:i + chunk_size] for i in range(0, len(smiles_list_), chunk_size)]
        for chunk in tqdm(chunks, desc="Processing chunks"):
            futures = [executor.submit(process_chunk, chunk)]
            for future in as_completed(futures):
                local_fg_status_updates, local_fg_edge_score_updates = future.result()
                for fg_smiles, (fg_id, frequency) in local_fg_status_updates.items():
                    fg_status_dict[fg_smiles] = fg_id, fg_status_dict.get(fg_smiles, ("None", 0))[1] + frequency
                for fg_edge_score, frequency in local_fg_edge_score_updates.items():
                    fg_edge_score_dict[fg_edge_score] = fg_edge_score_dict.get(fg_edge_score, 0) + frequency

    fg_edge_score_dict['total_num'] = total_num
    fg_status_dict['total_num'] = total_num
    print_info(dumps_json(fg_status_dict))
    print_info(dumps_json(fg_edge_score_dict))
    if save_filepath:
        with open(save_filepath, 'w') as file:
            json.dump([fg_status_dict, fg_edge_score_dict], file)
    return fg_status_dict, fg_edge_score_dict


def inspect_num_atoms_and_num_bonds_parallel(smiles_list_, num_limit=None, save_filepath=None, num_workers=13):
    total_num = min(len(smiles_list_), num_limit) if num_limit else len(smiles_list_)

    smiles_list_ = smiles_list_[:total_num]
    total_atom_numpy = np.zeros(total_num)
    total_bond_numpy = np.zeros(total_num)

    # Break the data into chunks
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Process each chunk in parallel
        for index, smiles in enumerate(tqdm(smiles_list_)):
            mol = Chem.MolFromSmiles(smiles)
            total_atom_numpy[index] = mol.GetNumAtoms()
            total_bond_numpy[index] = mol.GetNumBonds()

    total_atom_numpy = total_atom_numpy.astype(int)
    total_bond_numpy = total_bond_numpy.astype(int)
    # print_info(f"Total atom num: {total_atom_numpy}")
    # print_info(f"Total bond num: {total_bond_numpy}")
    if save_filepath:
        with open(save_filepath, 'w') as file:
            json.dump([total_atom_numpy.tolist(), total_bond_numpy.tolist()], file)

    return total_atom_numpy, total_bond_numpy



@cache
def get_spatial_pos_frequency_arr():
    with open(Path(pdir) / 'dump/spatial_pos_status.json') as file:
        data = json.load(file)
        to_return = []
        for id, frequency in data.items():
            to_return.append(20_0000 * 10 / (frequency + 1))
        return np.log(np.array(to_return) + 1)


@cache
def get_pair_distances_frequency_arr():
    with open(Path(pdir) / 'dump/pair_distances_status.json') as file:
        data = json.load(file)
        to_return = []
        for id, frequency in data.items():
            to_return.append(20_0000 * 10 / (frequency + 1))
        return np.log(np.array(to_return) + 1)


@cache
def get_angle_frequency_arr():
    with open(Path(pdir) / 'dump/angle_status_20.json') as file:
        data = json.load(file)
        to_return = []
        for id, frequency in data.items():
            to_return.append(20_0000 / (frequency + 1))
        return np.log(np.array(to_return) + 1)