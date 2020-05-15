from unittest.mock import Mock

import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from trends_ni.training.models import BenchmarkModel, SKLearnWrapper as SKW, MultiModelWrapper, ModelEnsembler


@pytest.fixture
def X():
    return np.random.random((100, 2))


@pytest.fixture
def y():
    return np.ones((100, 5))


def test_benchmark_model_fit(tiny_files_structure):
    bm_model = BenchmarkModel()
    y_train = pd.read_csv(tiny_files_structure.raw.y_train, index_col=0)
    bm_model.fit(pd.DataFrame([0]).values, y_train)
    assert bm_model.params["mean_values"].all()
    assert bm_model.params["mean_values"].shape == (5,)


def test_benchmark_model_predict(tiny_files_structure):
    bm_model = BenchmarkModel()
    x_train = np.ones((10, 5))
    y_train = (
        pd.read_csv(tiny_files_structure.raw.y_train, index_col=0).fillna(0).values
    )
    bm_model.fit(x_train, y_train)
    y_pred = bm_model.predict(x_train)

    assert y_pred.shape == y_train.shape
    assert np.isclose(y_pred.mean(axis=0), bm_model.params["mean_values"]).all()


def test_sklearn_wrapper(X, y):
    params = {"fit_intercept": True}
    lin_reg = SKW(model=LinearRegression(**params))
    lin_reg.fit(X, y)
    assert lin_reg.predict(np.random.random((10, 2))).all()
    assert lin_reg.params


def test_multi_model_wrapper(X, y):
    params = {"fit_intercept": True}
    multi_model = MultiModelWrapper(models=[SKW(LinearRegression(**params))] * 5)
    multi_model.fit(X, y)
    assert multi_model.predict(np.random.random((10, 2))).all()
    assert len(multi_model.params) == 5


@pytest.fixture
def children_and_ensembler():
    children = [SKW(LinearRegression())] * 3
    ensembler = SKW(LinearRegression())
    return children, ensembler


def test_model_ensembler_get_params(children_and_ensembler):
    children, ensembler = children_and_ensembler
    model_ensembler = ModelEnsembler(children=children, ensembler=ensembler)
    assert model_ensembler.params
    assert "ensembler" in model_ensembler.params.keys()


def test_model_ensembler_train(X, y, children_and_ensembler):
    children, ensembler = children_and_ensembler
    model_ensembler = ModelEnsembler(children=children, ensembler=ensembler)

    model_ensembler.fit(X, y)
    assert model_ensembler.predict(np.random.random((10, 2))).all()


def test_ensembler_child_dataset(X, y, children_and_ensembler):
    children, ensembler = children_and_ensembler

    model_ensembler = ModelEnsembler(children=children, ensembler=ensembler)
    model_ensembler.predict_with_children = Mock(spec=model_ensembler.predict_with_children)
    model_ensembler.oof_predict_with_children = Mock(spec=model_ensembler.oof_predict_with_children)

    model_ensembler.make_child_dataset(X)
    model_ensembler.predict_with_children.assert_called_once_with(X)

    model_ensembler.make_child_dataset(X, y)
    model_ensembler.oof_predict_with_children.assert_called_once_with(X, y)


def test_predict_with_children(X, y, children_and_ensembler):
    children, ensembler = children_and_ensembler
    _ = [child.fit(X, y) for child in children]

    model_ensembler = ModelEnsembler(children=children, ensembler=ensembler)
    preds = model_ensembler.predict_with_children(X)
    assert preds.shape[0] == X.shape[0]
    assert preds.shape[1] == y.shape[1] * len(children)


def test_oof_predict_with_children(X, y, children_and_ensembler):
    children, ensembler = children_and_ensembler
    model_ensembler = ModelEnsembler(children=children, ensembler=ensembler)
    preds = model_ensembler.oof_predict_with_children(X, y)
    assert preds.shape[0] == X.shape[0]
    assert preds.shape[1] == y.shape[1] * len(children)
