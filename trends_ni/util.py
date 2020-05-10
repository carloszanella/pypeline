from typing import List

import pandas as pd
import numpy as np

from trends_ni.structure import structure


def get_train_ids() -> np.ndarray:
    train_scores = pd.read_csv(structure.raw.y_train, index_col=0)
    train_ids = train_scores.index.values
    return train_ids


def make_submission(predictions: np.ndarray, ids: List[float]) -> pd.DataFrame:
    columns = pd.Series(["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"])
    piled_preds = predictions.reshape(-1, 1)

    index = []
    for i in ids:
        index.append(str(i) + "_" + columns)

    index = pd.concat(index)

    submission = pd.DataFrame(piled_preds, index=index)

    return submission
