import pandas as pd

from trends_ni.train_model.models import Model, BenchmarkModel


def test_model_class():
    mod = Model()
    mod.predict(pd.DataFrame())
    mod.fit(pd.DataFrame(), pd.DataFrame())


def test_benchmark_model(tiny_files_structure):
    bm_model = BenchmarkModel()
    bm_model.fit()