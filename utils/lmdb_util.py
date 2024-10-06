import json
import pickle
from collections import Counter
from unittest import TestCase

import lmdb
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from _config import pdir, set_up_spatial_pos
from data_process.utils_dataprocess import add_spatial_pos_bar, add_spatial_pos


def read_lmdb(db_path, idx):
    env = lmdb.open(db_path, subdir=False, lock=False, readonly=True)
    txn = env.begin()
    return txn.get(str(idx).encode())


def extract_data_from_lmdb(read_path, write_path):
    read_env = lmdb.open(read_path, subdir=False, lock=False, readonly=True)
    write_env = lmdb.open(write_path, map_size=1024 * 1024 * 1024 * 1024, subdir=False, lock=False)
    read_txn = read_env.begin()
    write_txn = write_env.begin(write=True)

    read_length = int(read_txn.get(b'__len__').decode())

    oriange = 0
    nums = read_length
    keys_write = []
    for idx in tqdm(range(oriange, oriange + nums)):
        data = read_txn.get(str(idx).encode())

        idx_write = idx - oriange
        keys_write.append(str(idx_write).encode())
        write_txn.put(str(idx_write).encode(), data)

    write_txn.commit()
    with write_env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys_write))
        txn.put(b'__len__', str(len(keys_write)).encode())
    read_env.close()
    write_env.close()


def concat_lmdb(read_path_list, write_path):
    write_env = lmdb.open(write_path, map_size=1024 * 1024 * 1024 * 1024, subdir=False, lock=False)
    write_txn = write_env.begin(write=True)

    keys_write = []
    length = 0
    for read_path in read_path_list:
        read_env = lmdb.open(read_path, subdir=False, lock=False, readonly=True)
        read_txn = read_env.begin()

        read_length = int(read_txn.get(b'__len__').decode())

        for idx in tqdm(range(read_length)):
            data = read_txn.get(str(idx).encode())

            keys_write.append(str(length).encode())
            write_txn.put(str(length).encode(), data)
            length += 1

        read_env.close()

    write_txn.commit()
    with write_env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys_write))
        txn.put(b'__len__', str(len(keys_write)).encode())
    write_env.close()


def add_attr_lmdb(write_path):
    env = lmdb.open(write_path, map_size=1024 * 1024 * 1024 * 1024, subdir=False, lock=False)
    txn = env.begin(write=True)

    length = int(txn.get(b'__len__'))
    for idx in tqdm(range(length)):
        data = txn.get(str(idx).encode())
        data = pickle.loads(data)
        # data = add_spatial_pos(data)
        # data = add_bond_angles(data)

        if 'spatial_pos_bar' not in data:
            data = add_spatial_pos_bar(data)
            # data = change_functiongroup(data)
            # del data['bond_angles']
            txn.put(str(idx).encode(), pickle.dumps(data))

    txn.commit()
    env.close()