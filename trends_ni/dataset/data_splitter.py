from typing import List, Tuple

import numpy as np


class TrainValSplitter:
    def __init__(self, ids: np.array):
        self.ids = ids
        self.ds_size = len(ids)

    def split(self, val_split: float = 0.2) -> Tuple[np.array, np.array]:
        val_size = int(np.floor(val_split * self.ds_size))
        val_ids = np.random.choice(self.ids, size=val_size, replace=False)
        train_ids = np.delete(self.ids, val_ids)

        return train_ids, val_ids
