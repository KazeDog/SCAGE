import numpy as np

from data_process.function_group_constant import nfg


def padding_pair_distances_loong(pair_distances, edges, max_length, padding_value=0, item=0):
    matrix = np.zeros((max_length)) + padding_value
    for i in range(len(edges)):
    #     try:
    #         matrix[i] = pair_distances[edges[i][0]][edges[i][1]]
    #         print(edges[i][0], edges[i][1])
    #     except:
    #         print('WARNING!!!! ', edges[i][0], edges[i][1])
        matrix[i] = pair_distances[edges[i][0]][edges[i][1]]

    return matrix


def padding_function_group_index(function_group_index, max_length, padding_value=-1):
    matrix = np.zeros((max_length, nfg() + 1)) + padding_value
    matrix[:len(function_group_index), :] = 0
    for i in range(len(function_group_index)):
        for j in function_group_index[i]:
            matrix[i, j] = 1
    return matrix


def padding_1d_sequence(sequence, max_length, padding_value=0):
    return np.concatenate([sequence, [padding_value] * (max_length - len(sequence))])


def padding_2d_sequence(matrix_, max_length: int, padding_value=0):
    """matrix_: shape of [n_atoms, n_atoms]"""
    # return np.concatenate([sequence, [[padding_value] * len(sequence[0])] * (max_length - len(sequence))])
    matrix = np.zeros((max_length, max_length)) + padding_value
    matrix[:len(matrix_[0]), :len(matrix_[0])] = matrix_
    return matrix


def padding_3d_sequence(cube_, max_length, padding_value=0):
    """cube_: shape of [max_length, max_length, max_length]"""
    cube = np.zeros((max_length, max_length, max_length)) + padding_value
    cube[:len(cube_[0][0]), :len(cube_[0][0]), :len(cube_[0][0])] = cube_
    return cube


def padding_edge_features(edge_feature, edge_index, max_length, padding_value=0):
    """
    Padding an edge sequence to a given length.

    Parameters
    ----------
    edge_feature: np.ndarray
        [n_edges]
        A edge sequence.
    edge_index: np.ndarray
        [n_edges, 2]
        A edge index sequence.
    max_length: int
        The length of the padded sequence.
    padding_value: int, optional
        The value used for padding.

    Returns
    -------
    np.ndarray
        The padded edge sequence.
        [max_length, max_length]
    """
    # matrix = [[padding_value] * max_length] * max_length
    # for i, j in edge_index:
    #     matrix[i][j] = edge_feature[i]
    #
    # return matrix

    matrix = np.zeros((max_length, max_length)) + padding_value
    for i, j in edge_index:
        matrix[i][j] = edge_feature[i]
    return matrix


def padding_edge_features_loong(edge_feature, max_length, padding_value=0):
    # edge_feature = edge_feature[:len(edge_feature) - atom_num]
    # edge_feature = edge_feature[::2]
    # print('edge_feature=', edge_feature)
    matrix = np.zeros((max_length)) + padding_value
    matrix[:len(edge_feature)] = edge_feature
    return matrix


def padding_adj(edges, max_length, padding_value=0):
    # construct the Adj matrix
    adj = np.zeros((max_length, max_length)) + padding_value
    # print(edges)
    for i, j in edges:
        # if i!=j:
        adj[i][j] = 1
    return adj


def padding_2d_loong(edges, max_length, dim):
    # construct the Adj matrix
    if len(edges) == 0:
        print(edges)
        return np.zeros((max_length, max_length)) - 1
    Adj = np.zeros((max_length, dim)) - 1
    Adj[:len(edges), :] = edges
    return Adj


def padding_edges(edges, max_length):
    # construct the Adj matrix
    adj = np.zeros((max_length, 2))
    adj[:len(edges), :] = edges
    return adj


def padding_atoms_pos(sequence, max_length, padding_value=0):
    matrix = np.zeros((max_length, 3)) + padding_value
    matrix[:len(sequence), :] = sequence
    return matrix


def padding_atom_bond_distances(distances, max_atom, max_edges, padding_value=0):
    matrix = np.zeros((max_atom, max_edges)) + padding_value
    matrix[:len(distances), :len(distances[0])] = distances
    return matrix

def padding_sp_sequence(matrix_, max_length: int, padding_value=0):
    """matrix_: shape of [n_atoms, n_atoms]"""
    # return np.concatenate([sequence, [[padding_value] * len(sequence[0])] * (max_length - len(sequence))])
    matrix = np.zeros((max_length, max_length)) + padding_value
    matrix[0, 0] = 0
    matrix[0, 1:len(matrix_[0]) + 1] = 1
    matrix[1:len(matrix_[0]) + 1, 0] = 1
    matrix[1:len(matrix_[0]) + 1, 1:len(matrix_[0]) + 1] = matrix_
    return matrix
