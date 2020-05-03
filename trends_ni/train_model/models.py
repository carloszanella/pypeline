from __future__ import annotations
import pandas as pd
import numpy as np


class Model:
    def predict(self, X: pd.DataFrame) -> np.array:
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Model:
        pass


class BenchmarkModel(Model):
    def __init__(self):
        mean_values = None

    def predict(self, X: pd.DataFrame) -> np.array:
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Model:
        pass