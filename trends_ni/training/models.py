from __future__ import annotations
import numpy as np


class Model:
    def __init__(self):
        self.version = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        pass


class BenchmarkModel(Model):
    def __init__(self):
        self.mean_values = None
        self.version = "benchmark_0.1"

    def predict(self, X: np.ndarray) -> np.ndarray:
        size = X.shape[0]
        y_pred = np.ones((size, len(self.mean_values))) * self.mean_values

        return y_pred

    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        self.mean_values = y.mean(axis=0)
        return self
