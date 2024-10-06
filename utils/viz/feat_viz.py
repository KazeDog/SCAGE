from typing import Iterable, Union

import numpy as np
from echo_logger import print_info
from matplotlib import pyplot as plt

from _config import task_configs
from data_process.function_group_constant import reference_fn_group
from datasets.dataloader import FinetuneDataset
from utils.userconfig_util import get_dataset_dir
from utils.viz.naive_fg_model import convert_atom_to_fg_nums__to__fg_counts

rf = reference_fn_group
rf.update({"0": "None"})
# convert into int keys
rf = {int(k): v for k, v in rf.items()}
all_possible_fg_nums = len(rf)
from data_process.compound_tools import get_all_matched_fn_ids_returning_tuple
from rdkit import Chem


def get_instant_function_group_index_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return get_all_matched_fn_ids_returning_tuple(mol)[0]


def get_summery(dataset: Union[FinetuneDataset, str], type_='sum'):
    all_possible_labels = get_all_possible_labels(dataset)
    if isinstance(dataset, str):
        dataset_name = dataset
        dataset = FinetuneDataset(get_dataset_dir('wd-iota'), dataset)
    operation_count_dict = {}
    y2x_sum = {}
    for label in all_possible_labels:
        y2x_sum[label] = np.zeros(all_possible_fg_nums)
    for item in dataset:
        label = get_real_label(dataset_name, item['label'], all_possible_labels=all_possible_labels)
        if isinstance(label, np.ndarray) or isinstance(label, list):
            label = tuple(label)
        if label in y2x_sum:
            y2x_sum[label] += convert_atom_to_fg_nums__to__fg_counts(
                get_instant_function_group_index_from_smiles(item['smiles']))
            if label not in operation_count_dict:
                operation_count_dict[label] = 1
            else:
                operation_count_dict[label] += 1
    # get avg
    if type_ == 'avg':
        for label in operation_count_dict.keys():
            y2x_sum[label] /= operation_count_dict[label]
    elif type_ == 'sum':
        pass
    else:
        raise ValueError(f"Unknown type_: {type_}")
    return y2x_sum


def get_real_label(dataset: str, raw_label, classes_for_regression=5, all_possible_labels=None):
    if dataset in ['bbbp', 'bace', 'clintox']:
        return raw_label
    elif dataset in ['esol', 'freesolv', 'lipophilicity']:
        threshold = (max(all_possible_labels) - min(all_possible_labels)) / classes_for_regression
        return int((raw_label - min(all_possible_labels)) // threshold)
    elif dataset in ['sider', 'tox21', 'toxcast']:
        return 'default(no distinguish)'
    else:
        raise ValueError(f"dataset {dataset} not found in task_configs")


def get_all_possible_labels(dataset: str, classes_for_regression=5) -> set:
    if dataset in ['bbbp', 'bace', 'clintox']:
        all_possible_labels = set()
        dataset = FinetuneDataset(get_dataset_dir('wd-iota'), dataset)
        for item in dataset:
            label = item['label']
            if isinstance(label, np.ndarray) or isinstance(label, list):
                label = tuple(label)
            all_possible_labels.add(label)
    elif dataset in ['esol', 'freesolv', 'lipophilicity']:
        all_possible_labels = set()
        for item in FinetuneDataset(get_dataset_dir('wd-iota'), dataset):
            all_possible_labels.add(item['label'][0])
        all_possible_labels = sorted(all_possible_labels)
        # we need to classify the regression labels into classes
        threshold = (max(all_possible_labels) - min(all_possible_labels)) / classes_for_regression
        all_possible_labels = {int((label - min(all_possible_labels)) // threshold) for label in all_possible_labels}
    elif dataset in ['sider', 'tox21', 'toxcast']:
        all_possible_labels = {'default(no distinguish)'}
    else:
        raise ValueError(f"dataset {dataset} not found in task_configs")

    return all_possible_labels


def plot_summary(y2x_sum, title='Sum of Features by Class'):
    labels = np.arange(len(next(iter(y2x_sum.values()))))  # Use the length of one value in the dict
    fig, ax = plt.subplots()

    # Plot each class sum dynamically for each label found in y2x_sum
    for label, sums in y2x_sum.items():
        ax.plot(labels, sums, label=f'Label {label}', marker='o')

    ax.set_xlabel('Features')
    ax.set_ylabel('Sum')
    ax.set_title('Sum of Features by Class')
    ax.legend()

    # Since there could be more features, setting the ticks accordingly
    ax.set_xticks(labels)
    ax.set_xticklabels(labels, rotation=45)
    # set size of figure
    fig.set_size_inches(40, 5)
    plt.grid(False)
    plt.show()


def plot_many_summaries(list_of_y2x_sum, list_of_subtitles, use_log_y=True):
    def plot_summary_one(y2x_sum, ax, subtitle):
        labels = np.arange(len(next(iter(y2x_sum.values()))))  # Length of one entry in the dictionary
        # Plot each class sum dynamically for each label found in y2x_sum
        for label, sums in y2x_sum.items():
            ax.plot(labels, sums, label=f'Label {label}', marker='o')

        ax.set_xlabel('Features')
        ax.set_ylabel('Sum')
        ax.set_title(subtitle)
        ax.legend()
        if use_log_y:
            ax.set_yscale('log')
        # Since there could be more features, setting the ticks accordingly
        ax.set_xticks(labels)
        ax.set_xticklabels(labels, rotation=45)
        plt.grid(False)

    num_datasets = len(list_of_y2x_sum)
    fig, axs = plt.subplots(nrows=num_datasets, ncols=1, figsize=(40, 5 * num_datasets))

    if num_datasets == 1:
        axs = [axs]  # Make it iterable for a single subplot scenario

    for ax, y2x_sum, subtitle in zip(axs, list_of_y2x_sum, list_of_subtitles):
        plot_summary_one(y2x_sum, ax, subtitle)

    plt.tight_layout()
    plt.show()


# print(get_all_possible_labels(dataset))
# print(dataset[0].keys())
# print(dataset[0]['smiles'], dataset[0]['label'])
# print(convert_atom_to_fg_nums__to__fg_counts(dataset[0]['function_group_index']))
# plot_summary(get_summery(dataset))

# plot_many_summaries(
#     [get_summery('bbbp'), get_summery('bace'), get_summery('clintox'), get_summery('esol'), get_summery('freesolv')]
#     , ['bbbp', 'bace', 'clintox', 'esol (num shows relative value. 0 is the smallest)',
#        'freesolv (num shows relative value. 0 is the smallest)']
# )

# plot_many_summaries(
#     [get_summery('sider'), get_summery('tox21'), get_summery('toxcast'), get_summery('lipophilicity')]
#     , ['sider', 'tox21', 'toxcast', 'lipophilicity (num shows relative value. 0 is the smallest)']
# )

# all
plot_many_summaries(
    [get_summery(dataset_name) for dataset_name in ['sider', 'tox21', 'toxcast', 'clintox', 'bbbp', 'bace', 'freesolv',
                                                    'esol', 'lipophilicity']]
    , ['sider(no distinguish)', 'tox21(no distinguish)', 'toxcast(no distinguish)', 'clintox', 'bbbp', 'bace',
         'freesolv (num shows relative value. 0 is the smallest)', 'esol (num shows relative value. 0 is the smallest)',
            'lipophilicity (num shows relative value. 0 is the smallest)']
)
#
#
# #
# for dataset_name in [
#     # "sider",
#     # "tox21",
#     # "toxcast",
#     "clintox",
#     "bbbp",
#     "bace",
#     "freesolv",
#     "esol"
# ]:
#     print_info("All possible labels for dataset: ", dataset_name)
#     print(get_all_possible_labels(dataset_name))
#     plot_summary(get_summery(dataset_name))
# dataset = FinetuneDataset(get_dataset_dir('wd-iota'), 'freesolv')
# print(dataset[0]['label'][0])
