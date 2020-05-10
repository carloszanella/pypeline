from trends_ni.processing.data_splitter import TrainValSplitter
import numpy as np


def test_train_val_splitter():
    ids = np.arange(300, 500)
    train_splitter = TrainValSplitter()
    val_split = 0.2
    train_ix, val_ix = train_splitter.split(ids, val_split)

    assert len(val_ix) + len(train_ix) == len(ids)
    assert int(val_split * len(ids)) == len(val_ix)
