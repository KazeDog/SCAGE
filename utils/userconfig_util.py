from functools import lru_cache, cache
from pathlib import Path
from typing import Dict, Any

__CURRENT_USER__ = None
__CURRENT_DATASET_FORM__ = None

import yaml

from _config import pdir


def config_current_user(username: str, config_: Dict[str, Any]):
    """
    in config.config_finetune.yaml:
    userconfig:
      wd:
        dataset_dir: ...
      qjb:
        dataset_dir: ...
      ...
    """
    userconfig = config_['userconfig']
    if username in userconfig:
        dataset_dir = userconfig[username]['dataset_dir']
        config_['root'] = dataset_dir
        global __CURRENT_USER__
        __CURRENT_USER__ = username
    else:
        raise ValueError(f"username {username} not found in userconfig")
    return config_


@lru_cache()
def get_dataset_dir(username: str = None):
    if username is None:
        global __CURRENT_USER__
        path = f"config/config_finetune.yaml"
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        return Path(config['userconfig'][__CURRENT_USER__]['dataset_dir'])
    else:
        path = pdir + f"/config/config_finetune.yaml"
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        return Path(config['userconfig'][username]['dataset_dir'])


def get_current_user():
    global __CURRENT_USER__
    return __CURRENT_USER__


@lru_cache()
def get_split_dir():
    global __CURRENT_USER__
    path = pdir + f"/config/config_finetune.yaml"
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    return Path(config['userconfig'][__CURRENT_USER__]['split_dir'])


@cache
def drop_last_flag(dataset_len: int, batch_size: int):
    """
    if dataset_len % batch_size == 1, drop_last=True.
    """
    return dataset_len % batch_size == 1


def config_dataset_form(form: str, config_: Dict[str, Any]):
    assert form in ['pkl']
    global __CURRENT_DATASET_FORM__
    __CURRENT_DATASET_FORM__ = form
    config_['dataset_form'] = form
    return config_


def get_dataset_form():
    return __CURRENT_DATASET_FORM__
