from typing import List, Tuple

import numpy as np


class TrainValSplitter:
    def __init__(self, train_ids: List[float]):
        self.train_ids = train_ids
        self.ds_size = len(train_ids)

    def split(self, val_split: float = 0.2) -> Tuple[np.array, np.array]:
        val_size = int(np.floor(val_split * self.ds_size))
        val_ids = np.random.choice(self.train_ids, size=val_size, replace=False)
        train_ids = self.train_ids.drop(val_ids)

        return train_ids, val_ids
