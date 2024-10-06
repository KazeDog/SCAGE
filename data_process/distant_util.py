import torch
from data_process import algos

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