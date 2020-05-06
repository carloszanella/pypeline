from __future__ import annotations
import numpy as np


class Model:
    def __init__(self):
        self.version = None

    def predict(self, X: np.array) -> np.array:
        pass

    def fit(self, X: np.array, y: np.array) -> Model:
        pass


class BenchmarkModel(Model):
    def __init__(self):
        self.mean_values = None
        self.version = "benchmark_0.1"

    def predict(self, X: np.array) -> np.array:
        size = X.shape[0]
        y_pred = np.ones((size, len(self.mean_values))) * self.mean_values.values

        return y_pred

    def fit(self, X: np.array, y: np.array) -> Model:
        self.mean_values = y.mean(axis=0)
        return self
