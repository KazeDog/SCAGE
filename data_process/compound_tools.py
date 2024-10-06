#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
| Tools for compound features.
| Adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
"""
import math
from typing import List, Tuple
from unittest import TestCase

import numpy as np
import pubchempy as pcp
import torch
from echo_logger import *
from rdkit import Chem
from rdkit.Chem import AllChem
# noinspection PyUnresolvedReferences
from rdkit.Chem import rdDepictor
# noinspection PyUnresolvedReferences
from rdkit.Chem import rdchem
# noinspection PyUnresolvedReferences
from rdkit.Chem.Draw import rdMolDraw2D
# noinspection PyUnresolvedReferences
from rdkit.Chem.rdchem import Mol

from _config import set_up_spatial_pos
from data_process.function_group_constant import FUNCTION_GROUP_LIST_FROM_DAYLIGHT, get_fn_group_id2smarts, \
    get_independent_fn_group_ids
from data_process.distant_util import add_spatial_pos
from utils.global_var_util import GlobalVar

# noinspection PyUnresolvedReferences
from rdkit.Chem import rdDepictor
# noinspection PyUnresolvedReferences
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG





def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except Exception:
        return False

def get_fg_list(n_atom: int, fg_dict: Dict[int, List[List[int]]]) -> Tuple[List[int], List[int]]:
    fg = [0 for _ in range(n_atom)]
    fg_index_arr = [0 for _ in range(n_atom)]
    for idx in range(n_atom):
        max_length = 0
        max_length_key = 0
        fg_index = 0
        for fn_id, matches in fg_dict.items():
            for i in range(len(matches)):
                if idx in matches[i] and len(matches[i]) > max_length:
                    max_length = len(matches[i])
                    max_length_key = fn_id
                    fg_index = len(CompoundKit.fg_mo_list) * (i + 1) + fn_id
        fg[idx] = max_length_key
        fg_index_arr[idx] = fg_index
    return fg, fg_index_arr


def match_fg(mol, edges) -> Tuple[List[int], List[int]]:
    n_atom = len(mol.GetAtoms())
    # fg_dict = {}
    fg_dict: Dict[int, List[List[int]]] = {}
    # fn_group_idx -> list of atom index that belong to the functional group
    idx = 0
    for sn in CompoundKit.fg_mo_list:
        matches = mol.GetSubstructMatches(sn)
        idx += 1
        # match_num = 0
        fg_dict.setdefault(idx, [])
        for match in matches:
            fg_dict[idx].append(list(match))
    fg_dict = clean_fg_dict_with_max_coverage_fn_group_priority(fg_dict)
    fg, fg_index_arr = get_fg_list(n_atom, fg_dict)
    edges_index = [0 if fg_index_arr[edges[i][0]] == fg_index_arr[edges[i][1]] else 1 for i in range(len(edges))]
    return fg, edges_index


def get_all_matched_fn_ids_str(mol) -> Dict[int, List[str]]:
    """
    Get all matched functional group ids for each atom in the molecule.
    Args:
        mol: rdkit mol object.
    Returns:
        all_matched_fn_ids: dict. atom_index -> list of functional group ids.
    """
    fg_dict: Dict[int, List[List[int]]] = {}
    id2smarts = get_fn_group_id2smarts()
    result: Dict[int, List[str]] = {}  # atom_index: fn_strs
    # fn_group_idx -> list of atom index that belong to the functional group
    idx = 0
    for sn in CompoundKit.fg_mo_list:
        matches = mol.GetSubstructMatches(sn)
        idx += 1
        # match_num = 0
        fg_dict.setdefault(idx, [])
        for match in matches:
            fg_dict[idx].append(match)
    fg_dict = clean_fg_dict_with_max_coverage_fn_group_priority(fg_dict)
    # process get_independent_fn_group_ids()
    all_matched_fn_ids_independent: Dict[int, List[int]] = {}  # atom_index: fn_indices
    for smarts, fn_index in get_independent_fn_group_ids().items():
        matched_atom_ids = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        matched_atom_ids = set([i for j in matched_atom_ids for i in j])
        for atom_index, _ in enumerate(mol.GetAtoms()):
            all_matched_fn_ids_independent[atom_index] = all_matched_fn_ids_independent.get(atom_index, [])
            if atom_index in matched_atom_ids:
                all_matched_fn_ids_independent[atom_index].append(fn_index)
    # print_info(dumps_json(fg_dict, depth=2, ensure_ascii=False))
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        result[atom_idx] = []
        for fn_id, matches in fg_dict.items():
            for match in matches:
                if atom_idx in match:
                    result[atom_idx].append(id2smarts[fn_id])
    for atom_idx, fn_indices in all_matched_fn_ids_independent.items():
        result[atom_idx] = result.get(atom_idx, []) + [id2smarts[i] for i in fn_indices]
    return clean_result(result)


def clean_result(_result_: Dict[int, List[Union[int, str]]]):
    """
    Clean result.
    """
    result = {}
    for k, v in _result_.items():
        result[k] = list(set(v))
    return result

# @print_json
def get_all_matched_fn_ids_returning_tuple(mol: Mol, edges=None) -> Tuple[Dict[int, List[int | str]], List[int]]:
    """
        Get all matched functional group ids for each atom in the molecule.
        Args:
            mol: rdkit mol object.
        Returns:
            all_matched_fn_ids: dict. atom_index -> list of functional group ids.
        """
    fg_dict: Dict[int, List[List[int]]] = {}
    result: Dict[int, List[int]] = {}  # atom_index: fn_indices
    result_with_fg_only_index: Dict[int, List[int]] = {}  # atom_index: fn_indices
    result_bond: List[int] = []

    fg_num = len(CompoundKit.fg_mo_list)

    # fn_group_idx -> list of atom index that belong to the functional group
    idx = 0
    for sn in CompoundKit.fg_mo_list:
        matches = mol.GetSubstructMatches(sn)
        idx += 1
        # match_num = 0
        fg_dict.setdefault(idx, [])
        for match in matches:
            atom_list = []
            for atom_idx in match:
                atom_list.append(atom_idx)
            fg_dict[idx].append(atom_list)
    fg_dict = clean_fg_dict_with_max_coverage_fn_group_priority(fg_dict)
    # print_info('fg_dict', fg_dict)
    # process get_independent_fn_group_ids()
    all_matched_fn_ids_independent: Dict[int, List[int]] = {}  # atom_index: fn_indices
    for smarts, fn_index in get_independent_fn_group_ids().items():
        matched_atom_ids = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        matched_atom_ids = set([i for j in matched_atom_ids for i in j])
        for atom_index, _ in enumerate(mol.GetAtoms()):
            all_matched_fn_ids_independent[atom_index] = all_matched_fn_ids_independent.get(atom_index, [])
            if atom_index in matched_atom_ids:
                all_matched_fn_ids_independent[atom_index].append(fn_index)
    # print_info(dumps_json(fg_dict, depth=2, ensure_ascii=False))
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        result[atom_idx] = []
        result_with_fg_only_index[atom_idx] = []
        for fn_id, matches in fg_dict.items():
            index = 0
            for match in matches:
                if atom_idx in match:
                    result[atom_idx].append(fn_id)
                    result_with_fg_only_index[atom_idx].append(fn_id + fg_num * index)
                index += 1

    # edge(i, j):
    # 0: i and j belongs to no functional group
    # 1: one of i and j belongs to any functional group, while the other does not belong to any functional group
    # 2: one of i and j belongs to any functional group, while the other node belongs to other functional group, and i and j are not in the same functional group
    # 2: one of i and j belongs to any functional group, while the other node belongs to other functional group, and i and j are not in the same functional group
    # 3+: i and j belongs to the same functional group. if same one, 3; if same 2, 4; if same 3, 5; ...

    # print_info('result_with_fg_only_index', result_with_fg_only_index)
    if edges is None:
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # i->j
            edges += [(i, j)]
        edges = np.array(edges, dtype='int64')
    for edge in edges:

        if result_with_fg_only_index[edge[0]] == [] and result_with_fg_only_index[edge[1]] == []:
            result_bond.append(0)
        elif result_with_fg_only_index[edge[0]] == [] or result_with_fg_only_index[edge[1]] == []:
            result_bond.append(1)
        else:
            set1 = set(result_with_fg_only_index[edge[0]])
            set2 = set(result_with_fg_only_index[edge[1]])
            # print_info("Set 1 and Set 2", set1, set2)
            intersection = set1 & set2
            # print_info("Intersection", intersection)
            to_add = min(len(intersection), GlobalVar.max_fn_group_edge_type_additional_importance_score)
            result_bond.append(2 + to_add)

    for atom_idx, fn_indices in all_matched_fn_ids_independent.items():
        result[atom_idx] = result.get(atom_idx, []) + fn_indices

    for atom_idx, fgs in result.items():
        if not fgs:
            result[atom_idx] = [0]
    # # return enumerate result_bond
    # result_bond = [(edges[index][0], edges[index][1], result_bond[index]) for index in range(len(edges))]
    return clean_result(result), result_bond


def str_one_fg_matches(one_data: Tuple[Dict[int, List[int | str]], List[int]]):
    l_part, r_part = one_data
    l_part_new = {}
    r_part_new = {}
    for atom_index, matched_fg_indices in l_part.items():
        for fg_index in matched_fg_indices:
            str_one = get_fn_group_id2smarts()[fg_index] if fg_index in get_fn_group_id2smarts() else "None"
            l_part_new[atom_index] = l_part_new.get(atom_index, []) + [str_one]
    for edge_index, edge_score in enumerate(r_part):
        r_part_new[edge_index] = edge_score
    return l_part_new, r_part_new


class Test_get_all_matched_fn_ids_returning_tuple(TestCase):
    def test12367128366175(self):
        smiles__ = "C1CC2=C3C(=CC=C2)C(=CN3C1)[C@H]4[C@@H](C(=O)NC4=O)C5=CNC6=CC=CC=C65"
        mol_ = Chem.MolFromSmiles(smiles__)
        # get edges
        edges = []
        for bond in mol_.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        print_info(edges)
        result, result_bond = get_all_matched_fn_ids_returning_tuple(mol_, edges)
        print_info('result', result)
        print_info('result_bond', result_bond)


def clean_fg_dict_with_max_coverage_fn_group_priority(fg_dict: Dict[int, List[List[int]]]) \
        -> Dict[int, List[List[int]]]:
    """
    Clean functional group dictionary with max coverage functional group priority.
    That is, if an atom is in multiple functional groups, it will be assigned to the functional group with the largest
    number of atoms.

    However, the condition is that the atoms in the smaller functional group should be a subset of the atoms in the
    larger functional group.
    Args:
        fg_dict: functional group dictionary. {fn_id -> list of matches}. each match is a list of atom indices.
    Returns:
        cleaned functional group dictionary.
    """
    all_matches = set(tuple(match) for matches in fg_dict.values() for match in matches)
    for fn_id, matches in fg_dict.items():
        for idx, match in enumerate(matches):
            for other_match in all_matches:
                if set(match) < set(other_match):
                    matches[idx] = []
    fg_dict = {fn_id: [match for match in matches if match] for fn_id, matches in fg_dict.items()}
    return fg_dict

def rd_chem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    return alist.index(elem) if elem in alist else len(alist) - 1


def get_bond_feature_dims(list_acquired_feature_names):
    """tbd"""
    list_bond_feat_dim = list(map(len, [CompoundKit.bond_vocab_dict[name] for name in list_acquired_feature_names]))
    # +1 for self loop edges
    return [_l + 1 for _l in list_bond_feat_dim]


class CompoundKit(object):
    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],
        "chiral_tag": rd_chem_enum_to_list(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "hybridization": rd_chem_enum_to_list(rdchem.HybridizationType.values),
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'atom_is_in_ring': [0, 1],
    }

    # float features
    atom_float_names = ["van_der_waals_radis", "partial_charge", 'mass']
    # bond_float_feats= ["bond_length", "bond_angle"]     # optional

    ### functional groups
    fg_smarts_list = FUNCTION_GROUP_LIST_FROM_DAYLIGHT
    fg_mo_list: List[Mol] = [Chem.MolFromSmarts(smarts) for smarts in FUNCTION_GROUP_LIST_FROM_DAYLIGHT]

    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167

    period_table = Chem.GetPeriodicTable()

    ### atom
    @staticmethod
    def get_atom_value(atom, name):
        """get atom values"""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        # elif name == 'implicit_valence':
        #     return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif name == 'mass':
            return int(atom.GetMass())
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        # elif name == 'num_radical_e':
        #     return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring':
            return int(atom.IsInRing())
        # elif name == 'valence_out_shell':
        #     return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        elif name == 'function_group_index':
            return CompoundKit.get_function_group_index(atom)
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        """get atom features id"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        """get atom features size"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return len(CompoundKit.atom_vocab_dict[name])

    ### bond

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == 'bond_dir':
            return bond.GetBondDir()
        elif name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        elif name == 'is_conjugated':
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        """get bond features id"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return safe_index(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    ### fingerprint

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """get morgan fingerprint"""
        nBits = CompoundKit.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        """get morgan2048 fingerprint"""
        nBits = CompoundKit.morgan2048_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """get maccs fingerprint"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    ### functional groups

    @staticmethod
    def get_function_group_index(mol, edges: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        get daylight functional group counts.

        Args:
            mol: rdkit mol object.
            edges: np.ndarray of shape (E, 2), where E is the number of edges.

        Returns:
            fg_index: list of functional group index. Length is the number of atoms.
            bond_fg: list of whether the bond is in the functional group.
                    If the bond is in the functional group, the value is 0, otherwise 1.
                    Length is the number of edges.
        """
        # fg_index, bond_fg = match_fg(mol, CompoundKit.fg_mo_list, edges)
        fg_index, bond_fg = match_fg(mol, edges)
        return fg_index, bond_fg

    @staticmethod
    def get_ring_size(mol):
        """return (N,6) list"""
        rings = mol.GetRingInfo()
        rings_info = []
        for r in rings.AtomRings():
            rings_info.append(r)
        ring_list = []
        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = 0
                for r in rings_info:
                    if len(r) == ringsize and atom.GetIdx() in r:
                        num_of_ring_at_ringsize += 1
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9
                atom_result.append(num_of_ring_at_ringsize)

            ring_list.append(atom_result)
        return ring_list

    @staticmethod
    def atom_to_feat_vector(atom):
        """ tbd """
        atom_names = {
            "atomic_num": safe_index(CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": safe_index(CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()),
            "degree": safe_index(CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()),
            "explicit_valence": safe_index(CompoundKit.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()),
            "formal_charge": safe_index(CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()),
            "hybridization": safe_index(CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()),
            # "implicit_valence": safe_index(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "is_aromatic": safe_index(CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "total_numHs": safe_index(CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()),
            # 'num_radical_e': safe_index(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),
            'atom_is_in_ring': safe_index(CompoundKit.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())),
            # 'valence_out_shell': safe_index(CompoundKit.atom_vocab_dict['valence_out_shell'],
            #                                 CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())),
            'van_der_waals_radis': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'partial_charge': CompoundKit.check_partial_charge(atom),
            'mass': atom.GetMass(),
        }
        return atom_names

    # noinspection PyUnresolvedReferences
    @staticmethod
    def get_atom_names(mol):
        """get atom name list
        """
        atom_features_dicts = []
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))

        return atom_features_dicts

    @staticmethod
    def check_partial_charge(atom):
        """tbd"""
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc:
            # unsupported atom, replace nan with 0
            pc = 0
        if pc == float('inf'):
            # max 4 for other atoms, set to 10 here if inf is get
            pc = 10
        return pc


# noinspection PyPep8Naming
class Compound3DKit(object):
    @staticmethod
    def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    # noinspection SpellCheckingInspection
    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
            # MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            index_ = np.argmin([x[1] for x in res])
            energy = res[index_][1]
            conf = new_mol.GetConformer(id=int(index_))
        except Exception:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        if return_energy:
            return new_mol, atom_poses, energy
        else:
            return new_mol, atom_poses

    @staticmethod
    def get_MMFF_atom_poses_nonminimum(mol, numConfs=None, return_energy=False, percent=75):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs, randomSeed=42)
            # MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            energies = [x[1] for x in res]
            energy_threshold = np.percentile(energies, percent)
            closest_index = np.argmin(np.abs(np.array(energies) - energy_threshold))
            # index_ = np.argmin([x[1] for x in res])
            conf = new_mol.GetConformer(id=int(closest_index))
            energy = energies[closest_index]
        except Exception:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        if return_energy:
            return new_mol, atom_poses, energy
        else:
            return new_mol, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_pair_distances(atom_poses):
        """get pair distance"""
        atom_number = len(atom_poses)
        pair_distances = []
        for i in range(atom_number):
            for j in range(atom_number):
                pair_distances.append(np.linalg.norm(atom_poses[i] - atom_poses[j]))
        pair_distances = np.array(pair_distances, 'float32')
        pair_distances = pair_distances.reshape(atom_number, atom_number)
        return pair_distances

    @staticmethod
    def get_bond_angles(atom_poses, angles_atom_index):
        """get triple angles"""
        angles_atom_index = np.array(angles_atom_index, 'int64')
        angles_number = len(angles_atom_index)
        angles_list = np.zeros(angles_number, 'float32')
        for i in range(angles_number):
            angles_list[i] = _get_angle(atom_poses[angles_atom_index[i][0]] - atom_poses[angles_atom_index[i][1]],
                                        atom_poses[angles_atom_index[i][2]] - atom_poses[angles_atom_index[i][1]])
        return angles_list

    @staticmethod
    def get_edge_distances(atom_poses, edges):
        edges_number = len(edges)

        bond_poses = []
        bond_distances = []

        for i in range(edges_number):
            bond_poses.append((atom_poses[edges[i][0]] + atom_poses[edges[i][1]]) / 2)

        for i in range(edges_number):
            for j in range(edges_number):
                bond_distances.append(np.linalg.norm(bond_poses[i] - bond_poses[j]))

        bond_distances = np.array(bond_distances, 'float32')
        bond_distances = bond_distances.reshape(edges_number, edges_number)
        return bond_distances

    @staticmethod
    def get_atom_bond_distances(atom_poses, edges):
        atom_number = len(atom_poses)
        edges_number = len(edges)

        bond_poses = []
        atom_bond_distances = []

        for i in range(edges_number):
            bond_poses.append((atom_poses[edges[i][0]] + atom_poses[edges[i][1]]) / 2)

        for i in range(atom_number):
            for j in range(edges_number):
                atom_bond_distances.append(np.linalg.norm(atom_poses[i] - bond_poses[j]))

        atom_bond_distances = np.array(atom_bond_distances, 'float32')
        atom_bond_distances = atom_bond_distances.reshape(atom_number, edges_number)
        return atom_bond_distances

    @staticmethod
    def get_angles_list(bond_angles, angles_bond_index):
        angles_bond_index = np.array(angles_bond_index)
        if len(angles_bond_index.shape) == 0:
            return np.array([0])
        else:
            angles_list = bond_angles[angles_bond_index[:, 0], angles_bond_index[:, 1]]
            return angles_list

    @staticmethod
    def get_bond_distances(pair_distances, edges):
        edges_number = len(edges)
        bond_distances = []
        for i in range(edges_number):
            bond_distances.append(pair_distances[edges[i][0]][edges[i][1]])
        bond_distances = np.array(bond_distances, 'float32')
        return bond_distances

def _get_angle(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    vec1 = vec1 / (norm1 + 1e-5)  # 1e-5: prevent numerical errors
    vec2 = vec2 / (norm2 + 1e-5)
    angle = np.arccos(np.dot(vec1, vec2))
    return angle


def find_angel_index(edges):
    """
    Find the angel index.

    Parameters
    ----------
    edges: np.ndarray
        The edges.
    atom_num: int
        The number of atoms.

    Returns
    -------
    np.ndarray
        The angel index.
    """
    angles_atom_index = []
    angles_bond_index = []
    for ii in range(len(edges)):
        for jj in range(len(edges)):
            i = edges[ii]
            j = edges[jj]
            if ii != jj:
                if i[1] == j[0]:
                    if angles_atom_index.count([j[1], i[1], i[0]]) == 0:
                        angles_atom_index.append([i[0], i[1], j[1]])
                        angles_bond_index.append([ii, jj])
                elif i[0] == j[1]:
                    if angles_atom_index.count([j[0], i[0], i[1]]) == 0:
                        angles_atom_index.append([i[1], i[0], j[0]])
                        angles_bond_index.append([ii, jj])
                elif i[0] == j[0]:
                    if angles_atom_index.count([j[1], i[0], i[1]]) == 0:
                        angles_atom_index.append([i[1], i[0], j[1]])
                        angles_bond_index.append([ii, jj])
                elif i[1] == j[1]:
                    if angles_atom_index.count([j[0], i[1], i[0]]) == 0:
                        angles_atom_index.append([i[0], i[1], j[0]])
                        angles_bond_index.append([ii, jj])

    return angles_atom_index, angles_bond_index

def get_dist_bar(dist: np.ndarray, percentiles: list):
    # print(percentiles)
    try:
        result = np.percentile(dist, percentiles)
    except Exception as e:
        result = np.zeros(len(percentiles))

    return result

def get_spatial_pos(data_len, edges):
    adj = torch.zeros([data_len, data_len], dtype=torch.bool)
    adj[edges[:, 0], edges[:, 1]] = True
    adj[edges[:, 1], edges[:, 0]] = True
    import algos
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    spatial_pos = shortest_path_result
    # spatial_pos = torch.from_numpy(shortest_path_result).long()
    spatial_pos = set_up_spatial_pos(spatial_pos, up=20)
    return spatial_pos


def binning_matrix(matrix, m, range_min, range_max):
    # 计算每个 bin 的范围
    bin_width = (range_max - range_min) / m

    # 将矩阵中的元素映射到 bin 中
    bin_indices = np.floor((matrix - range_min) / bin_width).astype(int)

    # 将超出范围的值限制在范围内
    bin_indices = np.clip(bin_indices, 0, m - 1)

    return bin_indices

def get_atom_poses(mol, skip_big_mol=True, skip_atom_num_bar=200):
    smiles = None
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
        smiles = mol
    atom_n = len(mol.GetAtoms())
    if skip_big_mol and atom_n > skip_atom_num_bar:
        return None
    if atom_n <= 400:
        _, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    return smiles, atom_poses



def mol_to_data_pkl(mol, pre_calculated_compose=None):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if len(mol.GetAtoms()) == 0:
        return None
    """tbd"""
    if len(mol.GetAtoms()) <= 400:
        if pre_calculated_compose is not None:
            mol, atom_poses = mol, pre_calculated_compose
        else:
            mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
            # mol, atom_poses = Compound3DKit.get_MMFF_atom_poses_nonminimum(mol, numConfs=50, percent=0)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)


    atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names

    data = {}

    ### atom features
    data = {name: [] for name in atom_id_names}

    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])

    data['edges'] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j
        data['edges'] += [(i, j)]

    #### self loop
    if len(data['edges']) == 0:
        N = len(data[atom_id_names[0]])
        for i in range(N):
            data['edges'] += [(i, i)]

    ### make ndarray and check length
    for name in list(CompoundKit.atom_vocab_dict.keys()):
        data[name] = np.array(data[name], 'int64')
    for name in CompoundKit.atom_float_names:
        data[name] = np.array(data[name], 'float32')
    data['edges'] = np.array(data['edges'], 'int64')

    angles_atom_index, angles_bond_index = find_angel_index(data['edges'])
    data['angles_atom_index'] = angles_atom_index
    data['angles_bond_index'] = angles_bond_index
    if len(angles_atom_index) == 0:
        # print("error")
        # print(item['smiles'])
        data['angles_atom_index'] = [[0, 0, 0]]
        data['angles_bond_index'] = [[0, 0]]

    atom_poses = np.array(atom_poses, 'float32')
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    function_group_index, function_group_bond_index = get_all_matched_fn_ids_returning_tuple(mol, data['edges'])
    data['function_group_index'] = function_group_index
    data['function_group_bond_index'] = np.array(function_group_bond_index, 'int64')
    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['pair_distances'] = Compound3DKit.get_pair_distances(atom_poses)
    data['bond_distances'] = Compound3DKit.get_bond_distances(data['pair_distances'], data['edges'])
    data['edge_types'] = np.array(get_edge_types(mol))
    data['bond_angles'] = Compound3DKit.get_bond_angles(atom_poses, data['angles_atom_index'])
    data['edge_distances'] = Compound3DKit.get_edge_distances(atom_poses, data['edges'])

    data = add_spatial_pos(data)
    data['atom_bond_distances'] = Compound3DKit.get_atom_bond_distances(atom_poses, data['edges'])
    data['pair_distances_bin'] = binning_matrix(data['pair_distances'], 30, 0, 30)
    data['bond_angles_bin'] = binning_matrix(data['bond_angles'], 20, 0, math.pi)
    data['spatial_pos'] = get_spatial_pos(len(data['atom_pos']), data['edges'])

    return data


def get_edge_group_indices(mol):
    if len(mol.GetAtoms()) <= 400:
        mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names

    data = {}

    ### atom features
    data = {name: [] for name in atom_id_names}

    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])

    ### bond and bond features
    data['edges'] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j
        data['edges'] += [(i, j)]


    data['edges'] = np.array(data['edges'], 'int64')

    function_group_index, function_group_bond = CompoundKit.get_function_group_index(mol, data['edges'])
    return function_group_index, function_group_bond


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        # print(atom_scores[atom.GetIdx()])
        atom.SetAtomMapNum(atom.GetIdx())
    return mol



def mol_with_atom_index(mol, style='clear'):
    assert style in ['traditional', 'clear', 'all']
    if style == 'clear':
        for atom in mol.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()))
    elif style == 'traditional':
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
    elif style == 'all':
        for atom in mol.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()))
            atom.SetAtomMapNum(atom.GetIdx())
    return mol


def get_bond_type_str(bond: Chem.Bond) -> str:
    a1 = bond.GetBeginAtom().GetSymbol()
    a2 = bond.GetEndAtom().GetSymbol()
    # Sort symbols to account for symmetry (e.g., C=O is the same as O=C)
    a1, a2 = sorted([a1, a2])

    # Determine bond type
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        bond_str = f"{a1}-{a2}"
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        bond_str = f"{a1}={a2}"
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        bond_str = f"{a1}#{a2}"
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        bond_str = f"{a1}~={a2}"
    else:
        bond_str = f"{a1} <{bond_type}> {a2}"
    return bond_str

def get_edge_types(mol) -> List[int]:
    """
    Args:
        mol:

    Returns:
        List of edge types. Len: edge_num
        each entry is the edge type index.
    """
    edge_types = []
    for idx, bond in enumerate(mol.GetBonds()):
        bond_str = get_bond_type_str(bond)
        edge_types.append(GlobalVar.get_embedding_dict().get(bond_str, GlobalVar.get_embedding_dict()["others"]))
    return edge_types


class Test_main(TestCase):
    def test55888(self):
        smiles = "c1ccccc1"
        mol = AllChem.MolFromSmiles(smiles)
        data = mol_to_data_pkl(mol)
        print_info(data)
        # angle = Compound3DKit.get_angles_list(data['bond_angles'], data['angles_bond_index'])
        # print_info(angle)

