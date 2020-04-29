from trends_ni.util import get_train_ids


def test_get_train_ids():
    ids = get_train_ids()
    assert ids.all()
