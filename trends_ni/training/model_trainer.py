from logging import getLogger, DEBUG

import numpy as np

from trends_ni.entities import TrainingResults
from trends_ni.evaluation.score import Score
from trends_ni.training.models import Model

log = getLogger(__name__)
log.setLevel(DEBUG)


class ModelTrainer:
    def train_model(
        self, model: Model, X_train: np.ndarray, y_train: np.ndarray
    ) -> TrainingResults:
        results = TrainingResults()
        results.model_version = model.version
        model.fit(X_train, y_train)
        results.model = model
        results.model_params = model.params

        scores, weighted_score = Score.evaluate_predictions(
            y_train, model.predict(X_train)
        )
        results.train_mae, results.train_weighted_mae = scores, weighted_score

        return results
