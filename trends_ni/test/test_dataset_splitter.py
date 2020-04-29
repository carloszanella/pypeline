from trends_ni.dataset.data_splitter import TrainValSplitter
from trends_ni.util import get_train_ids


def test_train_val_splitter():
    train_ids = get_train_ids()
    train_splitter = TrainValSplitter(train_ids)
    val_split = 0.2
    train_ix, val_ix = train_splitter.split(val_split)

    assert len(val_ix) + len(train_ix) == len(train_ids)
    assert int(val_split * len(train_ids)) == len(val_ix)
