import pickle

from torch.utils.data import Subset

__all__ = [
    'ScaffoldSplitter',
    'RandomScaffoldSplitter',
]

from utils.userconfig_util import get_split_dir


def create_splitter(split_type, seed):
    """Return a splitter according to the ``split_type``"""
    if split_type == 'scaffold':
        splitter = ScaffoldSplitter()
    elif split_type == 'random_scaffold':
        splitter = RandomScaffoldSplitter(seed)
    else:
        raise ValueError('%s not supported' % split_type)
    return splitter


class Splitter(object):
    """
    The abstract class of splitters which split up dataset into train/valid/test
    subsets.
    """

    def __init__(self):
        super(Splitter, self).__init__()


class ScaffoldSplitter(Splitter):

    def __init__(self):
        super(ScaffoldSplitter, self).__init__()

    @staticmethod
    def split(dataset, task_name):
        split_idx = pickle.load(open(get_split_dir() / f'scaffold/{task_name}.pkl', 'rb'))
        train_dataset = Subset(dataset, split_idx['train_idx'])
        valid_dataset = Subset(dataset, split_idx['valid_idx'])
        test_dataset = Subset(dataset, split_idx['test_idx'])
        return train_dataset, valid_dataset, test_dataset


class RandomScaffoldSplitter(Splitter):

    def __init__(self, seed):
        super(RandomScaffoldSplitter, self).__init__()
        self.seed = seed

    def split(self, dataset, task_name):
        seed_ = self.seed
        split_idx = pickle.load(open(get_split_dir() / f'random_scaffold/{task_name}/{task_name}_{seed_}.pkl', 'rb'))
        train_dataset = Subset(dataset, split_idx['train_idx'])
        valid_dataset = Subset(dataset, split_idx['valid_idx'])
        test_dataset = Subset(dataset, split_idx['test_idx'])
        return train_dataset, valid_dataset, test_dataset
