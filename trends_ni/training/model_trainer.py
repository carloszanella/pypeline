from logging import getLogger

import numpy as np

from trends_ni.entities import TrainingResults
from trends_ni.evaluation.score import Score
from trends_ni.training.models import Model

log = getLogger(__name__)


class ModelTrainer:
    def __init__(self, model: Model):
        self.model = model

    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> TrainingResults:
        results = TrainingResults()
        results.model_version = self.model.version
        self.model.fit(X_train, y_train)
        results.model = self.model

        scores, weighted_score = Score.evaluate_predictions(
            y_train, self.model.predict(X_train)
        )
        results.train_mae, results.train_weighted_mae = scores, weighted_score

        results.print_score_results()

        return results
