import json
import os
import shutil
from functools import cache
from pathlib import Path

import numpy as np
from echo_logger import print_info

from data_process.function_group_constant import nfg
from utils.global_var_util import GlobalVar

pdir: str = os.path.dirname(os.path.realpath(__file__))

task_configs = {
    'bace': (["Class"], 'classification', 'bce'),
    'bbbp': (["p_np"], 'classification', 'bce'),
    'clintox': (['CT_TOX', 'FDA_APPROVED'], 'classification', 'bce'),
    'hiv': (["HIV_active"], 'classification', 'bce'),
    'muv': ([
                'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
                'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
                'MUV-652', 'MUV-466', 'MUV-832'
            ], 'classification', 'bce'),
    'sider': ([
                  "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues",
                  "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders",
                  "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
                  "Reproductive system and breast disorders",
                  "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                  "General disorders and administration site conditions", "Endocrine disorders",
                  "Surgical and medical procedures", "Vascular disorders",
                  "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders",
                  "Congenital, familial and genetic disorders", "Infections and infestations",
                  "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
                  "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
                  "Ear and labyrinth disorders", "Cardiac disorders",
                  "Nervous system disorders", "Injury, poisoning and procedural complications"
              ], 'classification', 'bce'),
    'tox21': ([
                  "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                  "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
              ], 'classification', 'bce'),
    'CYP3A4': (["label"], 'classification', 'bce'),
    'esol': (["measured log solubility in mols per litre"], 'regression', 'mse'),
    'freesolv': (["expt"], 'regression', 'mse'),
    'lipophilicity': (['exp'], 'regression', 'mse')
}


def set_up_spatial_pos(matrix_, up=10):
    matrix_[matrix_ > up] = up
    return matrix_


def write_record(path, message):
    file_obj = open(path, 'a')
    file_obj.write('{}\n'.format(message))
    file_obj.close()


def copyfile(srcfile, path):
    if not os.path.isfile(srcfile):
        print(f"{srcfile} not exist!")
    else:
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy(srcfile, os.path.join(path, fname))


# noinspection SpellCheckingInspection
def model_is_dp(param_keys):
    return all([key.startswith('module') for key in param_keys])


@cache
def get_audit_atom_fg_frequency_arr(style: str = 'new', scale: int = 4):
    if nfg() == 143:
        style = 'traditional'
    # print_info(f"Using FG style: {style}, total fg num: {nfg() + 1}")
    from_audit_json_file = 'fg_status_dict_200000_with_NOS.json' if style == 'new' else 'atom_fg_status.json'
    with open(Path(pdir) / 'dump' / from_audit_json_file) as file:
        data = json.load(file)
        total_mol_num = data['total_num']
        data.pop('total_num')
        to_return = []
        for fg_smiles, (fg_id, frequency) in data.items():
            to_return.append(total_mol_num * scale / (frequency + 1))
        return np.log(np.array(to_return) + 1)


# noinspection PyTypeChecker
def get_downstream_task_names(config):
    task_name_ = config['task_name']
    if task_name_ in task_configs:
        target, task, loss_type = task_configs[task_name_]
    elif task_name_ == 'toxcast':
        import pandas as pd
        if GlobalVar.data_process_style == 'wd':
            raw_path = os.path.join(config['root'])
            csv_file = 'toxcast.csv'
        else:
            raw_path = os.path.join(config['root'], 'toxcast', 'raw')
            csv_file = os.listdir(raw_path)[0]
        input_df = pd.read_csv(os.path.join(raw_path, csv_file), sep=',')
        target = list(input_df.columns)[1:]
        task = 'classification'
        loss_type = 'bce'
    else:
        raise ValueError(f'{task_name_} not supported')

    config['target'] = target
    config['task'] = task
    config['loss_type'] = loss_type
    config['num_tasks'] = len(target)

    return config
