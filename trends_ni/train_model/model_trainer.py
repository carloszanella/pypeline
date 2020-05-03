import pickle
from logging import getLogger
from pathlib import Path
from typing import Tuple, List

import pandas as pd

from trends_ni.entities import TrainingResults
from trends_ni.evaluation.score import Score
from trends_ni.train_model.models import Model

BENCHMARK_MODEL_VERSION = "0.0.1"
log = getLogger(__name__)


class ModelTrainer:
    def __init__(self, save: bool = False):
        self.version = None
        self.save = save

    def train_model(self, model: Model, X_train: pd.DataFrame, y_train: pd.DataFrame, out_path: Path) -> TrainingResults:
        results = TrainingResults()
        results.model_version = self.version
        model.fit(X_train, y_train)
        results.model = model

        scores, weighted_score = Score.evaluate_predictions(y_train, model.predict(X_train))
        results.scores, results.weighted_score = scores, weighted_score

        if self.save:
            self.save_results(out_path, results)

        results.print_score_results()

        return results

    def save_results(self, path: Path, results: TrainingResults):
        path.parent.mkdir(exist_ok=True)
        log.info(f"Saving model on path {path}")

        with open(path, "wb") as fp:
            pickle.dump(results, fp)
