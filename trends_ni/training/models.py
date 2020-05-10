from __future__ import annotations
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor


class Model(metaclass=ABCMeta):
    def __init__(self, version: str):
        self.version = version
        self.params = None

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        pass


class BenchmarkModel(Model):
    def __init__(self):
        super().__init__(version="benchmark_0.1")
        self.params = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        size = X.shape[0]
        y_pred = np.ones((size, len(self.params["mean_values"]))) * self.params["mean_values"]

        return y_pred

    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        self.params["mean_values"] = y.mean(axis=0)
        return self


class SKLearnWrapper(Model):
    def __init__(self, model: BaseEstimator):
        name = type(model).__name__
        super().__init__(version=name)
        self.model = MultiOutputRegressor(model, -1)
        self.params = model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        self.model.fit(X, y)
        return self