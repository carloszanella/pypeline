import pickle
from pathlib import Path
from typing import Tuple, List

import dask.dataframe as dd
import pandas as pd

from trends_ni.evaluation.score import Score


class ModelTrainer:
    def train_model(self, X_train: dd.DataFrame, y_train: dd.DataFrame):
        pass

    def fit(self, X_train: dd.DataFrame, y_train: dd.DataFrame):
        pass

    def save_model(self, path: Path):
        pass


class BenchmarkModelTrainer(ModelTrainer):
    """Benchmark Linear Model"""

    def __init__(
            self,
            version: str,
            model: BaseEstimator = MultiOutputRegressor(LinearRegression(), -1),
            save_model: bool = False
    ):
        self.model = model
        self.version = "benchmark_" + version
        self.ds_builder = ds_builder
        self.out_dir = OUT_DIR / "model"
        self.save = save_model

    def train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> BaseEstimator:
        self.fit(X_train, y_train)

        scores, weighted_score = self.get_training_metrics(X_train, y_train)

        results_dict = {
            "model": self.model,
            "scores": scores,
            "weighted_score": weighted_score,
        }

        if self.save:
            self.save_model(results_dict, dataset_id)

        return self.model

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        print("Training model on data")
        self.model.fit(X_train, y_train)
        print("Finished model training")

    def get_training_metrics(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Tuple[List[float], float]:
        scores, weighted_score = Score.evaluate_predictions(y_train, self.model.predict(X_train))
        print("Training scores")
        print("###############")
        print("MAE: ", scores)
        print("Weighted Score: ", weighted_score)

        return scores, weighted_score

    def save_model(self, results: dict, dataset_id: str):
        file_name = f"model_{self.version}_ds_{dataset_id}"
        path = self.out_dir / file_name
        self.out_dir.mkdir(exist_ok=True)
        print(f"Saving model on path {path}")
        with open(path, "wb") as f:
            pickle.dump(results, f)