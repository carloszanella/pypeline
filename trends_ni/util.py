import pandas as pd
import numpy as np

from trends_ni.structure import structure


def get_train_ids() -> np.ndarray:
    train_scores = pd.read_csv(structure.raw.y_train, index_col=0)
    train_ids = train_scores.index.values
    return train_ids