from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
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
        self.model = model
        self.params = model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        self.model.fit(X, y)
        return self


class MultiModelWrapper(Model):
    def __init__(self, models: List[Model]):
        super().__init__(version="multi-model")
        assert len(models) == 5, "Number of models must be five"
        self.models = models
        self.params = [model.params for model in models]

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.zeros((X.shape[0], 5))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)

        return predictions

    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        for i, model in enumerate(self.models):
            model.fit(X, y[:, i])

        return self


class ModelEnsembler(Model):
    def __init__(self, ensembler: Model, children: List[Model], n_fols: int = 5):
        super().__init__(version=f"ensemble-{ensembler.version}")
        self.ensembler = ensembler
        self.children = children
        self.params = self.get_ensemble_params()
        self.n_folds = n_folds

    def predict(self, X: np.ndarray) -> np.ndarray:
        child_predictions = self.make_child_dataset(X)
        predict_ds = np.concatenate([X, child_predictions], axis=1)
        predictions = self.ensembler.predict(predict_ds)

        return predictions

    def fit(self, X: np.ndarray, y: np.ndarray) -> Model:
        child_predictions = self.make_child_dataset(X, y)

        return self

    def get_ensemble_params(self):
        params = {}

        for child in self.children:
            params[child.version] = child.params

        params["ensembler"] = self.ensembler.params

        return params

    def make_child_dataset(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:

        if y is None:
            dataset = self.predict_with_children(X)

        else:
            dataset = self.oof_predict_with_children(X, y)

        return dataset

    def predict_with_children(self, X: np.ndarray) -> np.ndarray:
        preds = [child.predict(X) for child in self.children]
        preds = np.concatenate(preds, axis=1)

        return preds

    def oof_predict_with_children(self, X, y) -> np.ndarray:
        kfold = KFold(self.n_folds)
        preds = np.zeros(y.shape)
        for train_ix, val_ix in kfold.split(X):
            pass  # TODO