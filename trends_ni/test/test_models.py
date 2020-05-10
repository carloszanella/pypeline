import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from trends_ni.training.models import BenchmarkModel, SKLearnWrapper


def test_benchmark_model_fit(tiny_files_structure):
    bm_model = BenchmarkModel()
    y_train = pd.read_csv(tiny_files_structure.raw.y_train, index_col=0)
    bm_model.fit(pd.DataFrame([0]), y_train)
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


def test_sklearn_wrapper_init():
    params = {"fit_intercept": True}
    lin_reg = SKLearnWrapper(model=LinearRegression(**params))
    lin_reg.fit(np.random.random((100, 2)), np.ones((100,5)))
    assert lin_reg.predict(np.random.random((10, 2))).all()
    assert lin_reg.params
