import os
import glob

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from omegaconf import DictConfig
from typing import List, Tuple, Dict

from concurrent.futures import ProcessPoolExecutor
from dataset.data_processor import DataProcessor
from common.io import pickle_load, pickle_save


class CarlaDataset(Dataset):
    """
    Class to handle CarlaDataset
    """

    def __init__(
            self,
            configs: DictConfig,
            data_folder: str
    ):
        self._configs = configs
        self._max_workers = configs.model.train.max_workers

        self._data_processor = DataProcessor(configs, map_path=f"{data_folder}/static.csv")
        self._data = self._get_data(data_folder=data_folder)

    def _get_data(
            self,
            data_folder: str
    ) -> List:
        """
        Process data in data_folder
        Cache data as a pickle file
        :param data_folder:
        :return:
        """
        compressed_file = f"{data_folder}/dynamic_by_ts_compressed.pkl"

        # use pre-saved pickle file if exists
        if os.path.exists(compressed_file):
            container = pickle_load(compressed_file)
        else:
            container = list()
            list_data = glob.glob(f"{data_folder}/dynamic_by_ts/*.csv")

            # TODO: bug allocating memory?
            # --- process data in parallel ---
            # with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            #     for inp_proc, out_proc in tqdm(
            #             executor.map(self._data_processor.process, list_data),
            #             total=len(list_data)
            #     ):
            #         if inp_proc is not None and out_proc is not None:
            #             container.append((inp_proc, out_proc))

            # --- process data in sequence ---
            for data_path in tqdm(list_data, total=len(list_data)):
                inp_proc, out_proc = self._data_processor.process(data_path)
                if inp_proc is not None and out_proc is not None:
                    container.append((inp_proc, out_proc))

            # --- save pickle ---
            pickle_save(container, compressed_file)

        return container

    def __getitem__(
            self,
            index: int
    ) -> Tuple[Dict, torch.Tensor]:
        """
        Get data at index
            - Input in dict, each value in tensor
                + traffic_light
                + map
                + agent
                + others
            - Output in tensor
        :param index:
        :return:
        """
        return self._data[index]

    def __len__(self) -> int:
        """
        Number of sample in data
        :return:
        """
        return len(self._data)
