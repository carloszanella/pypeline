from typing import List
import pandas as pd

from trends_ni.structure import structure


def get_train_ids() -> pd.Index:
    train_scores = pd.read_csv(structure.raw.train_scores, index_col=0)
    TRAIN_IDS = train_scores.index
    return TRAIN_IDS