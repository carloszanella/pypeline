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
        self.mean_values = None

    def predict(self, X: pd.DataFrame) -> np.array:
        size = X.shape[0]
        y_pred = np.ones((size, len(self.mean_values))) * self.mean_values.values

        return y_pred

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Model:
        self.mean_values = y.mean(axis=0)