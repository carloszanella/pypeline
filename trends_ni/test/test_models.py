import pandas as pd
import numpy as np

from trends_ni.train_model.models import Model, BenchmarkModel


def test_model_class():
    mod = Model()
    mod.predict(pd.DataFrame())
    mod.fit(pd.DataFrame(), pd.DataFrame())


def test_benchmark_model_fit(tiny_files_structure):
    bm_model = BenchmarkModel()
    y_train = pd.read_csv(tiny_files_structure.raw.y_train, index_col=0)
    bm_model.fit(pd.DataFrame([0]), y_train)
    assert bm_model.mean_values.all()
    assert bm_model.mean_values.shape == (5, )


def test_benchmark_model_predict(tiny_files_structure):
    bm_model = BenchmarkModel()
    x_train = pd.DataFrame(np.ones((10, 5)))
    y_train = pd.read_csv(tiny_files_structure.raw.y_train, index_col=0)
    bm_model.fit(x_train, y_train)
    y_pred = bm_model.predict(x_train)

    assert y_pred.shape == y_train.shape
    assert np.isclose(y_pred.mean(axis=0), bm_model.mean_values).all()
