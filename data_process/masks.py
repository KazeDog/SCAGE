import numpy as np


def attention_mask_using_0(sequence_len, max_length, padding_value=0):
    assert sequence_len <= max_length
    matrix = np.zeros((max_length, max_length)) + padding_value
    matrix[:sequence_len, :sequence_len] = 1
    return matrix


def attention_mask_atom_bond(atom_len, edge_len, max_atom, max_edge, padding_value=0):
    assert atom_len <= max_atom
    assert edge_len <= max_edge
    matrix = np.zeros((max_atom, max_edge)) + padding_value
    matrix[:atom_len, :edge_len] = 1
    return matrix


def angel_mask(sequence_len, max_length, padding_value=0):
    assert sequence_len <= max_length
    matrix = np.zeros((max_length, max_length, max_length)) + padding_value
    matrix[:sequence_len, :sequence_len, :sequence_len] = 1
    return matrix


def fg_mask(data_len, max_len, max_fg_num, padding_value=0):
    assert data_len <= max_len
    matrix = np.zeros((max_len, max_fg_num)) + padding_value
    matrix[:data_len, :] = 1
    return matrix


def fg_bond_mask(data_len, max_len, padding_value=0):
    assert data_len <= max_len
    matrix = np.zeros(max_len) + padding_value
    matrix[:data_len] = 1
    return matrix

def atom_mask(sequence_len, max_length, padding_value=0):
    assert sequence_len <= max_length
    matrix = np.zeros(max_length) + padding_value
    matrix[:sequence_len] = 1
    return matrix
