import os
import json
import pickle
from typing import Dict, List


def write_config(
        config: Dict,
        save_path: str
) -> None:
    """
    Write configuration to save log folder
    :param config: dict configuration
    :param save_path: path to save log folder
    :return:
    """
    dst_path = os.path.join(save_path, "config.json")
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def pickle_load(path: str) -> List:
    """
    Load compressed pickle data
    :param path: path to pickle data
    :return:
        (List) data to load into dataloader
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def pickle_save(
        data: List,
        path: str
) -> None:
    """
    Save compressed pickle data to file
    :param data: processed data
    :param path:
    :return:
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)
