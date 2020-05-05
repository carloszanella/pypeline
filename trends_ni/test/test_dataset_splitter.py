from trends_ni.dataset.data_splitter import TrainValSplitter
import numpy as np


def test_train_val_splitter():
    ids = np.arange(1, 100, 50)
    train_splitter = TrainValSplitter()
    val_split = 0.2
    train_ix, val_ix = train_splitter.split(ids, val_split)

    assert len(val_ix) + len(train_ix) == len(ids)
    assert int(val_split * len(ids)) == len(val_ix)
