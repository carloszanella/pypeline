import pickle
from logging import getLogger
from pathlib import Path

import numpy as np

from trends_ni.entities import TrainingResults
from trends_ni.evaluation.score import Score
from trends_ni.training.models import Model

log = getLogger(__name__)


class ModelTrainer:
    def __init__(self, model: Model, save: bool = False):
        self.save = save
        self.model = model

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, out_path: Path) -> TrainingResults:
        results = TrainingResults()
        results.model_version = self.model.version
        self.model.fit(X_train, y_train)
        results.model = self.model

        scores, weighted_score = Score.evaluate_predictions(y_train, self.model.predict(X_train))
        results.scores, results.weighted_score = scores, weighted_score

        if self.save:
            self.save_results(out_path, results)

        results.model_path = out_path
        results.print_score_results()

        return results

    def save_results(self, path: Path, results: TrainingResults):
        path.parent.mkdir(exist_ok=True)
        log.info(f"Saving model on path {path}")

        with open(path, "wb") as fp:
            pickle.dump(results, fp)
