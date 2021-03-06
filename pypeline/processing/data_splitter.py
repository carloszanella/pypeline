from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import numpy as np


class DataSplitter(metaclass=ABCMeta):
    @abstractmethod
    def split(self, ids: np.array, val_split: float) -> Tuple[np.ndarray, np.ndarray]:
        pass


class TrainValSplitter(DataSplitter):
    def split(
        self, ids: np.array, val_split: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        ds_size = len(ids)
        val_size = int(np.floor(val_split * ds_size))
        val_ids = np.random.choice(ids, size=val_size, replace=False)
        train_ids = ids[~np.isin(ids, val_ids)]

        return train_ids, val_ids
