from typing import List

import numpy as np


class TrainValSplitter:
    def __init__(self, train_ids: List[float] = TRAIN_IDS):
        self.train_ids = train_ids
        self.ds_size = len(train_ids)

    def split(self, val_split: float = 0.2):
        val_size = int(np.floor(val_split * self.ds_size))
        val_ids = self.train_ids[np.random.randint(0, self.ds_size, val_size).astype(int)]
        train_ix = self.train_ids[~self.train_ids.isin(val_ids)]
        val_ix = self.train_ids[self.train_ids.isin(val_ids)]

        return train_ix, val_ix