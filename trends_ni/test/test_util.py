from trends_ni.util import get_train_ids, make_submission


def test_get_train_ids():
    ids = get_train_ids()
    assert ids.all()


def test_make_submission(sample_predictions, sample_ids):
    sub = make_submission(sample_predictions, sample_ids)
    assert sub.all().all()
    assert sub.shape == (len(sample_ids) * 5, 1)
