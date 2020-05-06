from typing import List, Tuple

import numpy as np


class DataSplitter:
    def split(self, ids: np.array, val_split: float) -> Tuple[np.array, np.array]:
        pass


class TrainValSplitter(DataSplitter):
    def split(self, ids: np.array, val_split: float = 0.2) -> Tuple[np.array, np.array]:
        ds_size = len(ids)
        val_size = int(np.floor(val_split * ds_size))
        val_ids = np.random.choice(ids, size=val_size, replace=False)
        train_ids = np.delete(ids, val_ids)

        return train_ids, val_ids
