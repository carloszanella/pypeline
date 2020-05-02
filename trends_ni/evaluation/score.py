from typing import Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error


class Score:

    @staticmethod
    def evaluate_predictions(y_true, y_pred) -> Tuple[list, float]:
        weights = np.array([0.3, 0.175, 0.175, 0.175, 0.175])
        scores = []
        for i in range(5):
            y_t, y_p = y_true.values[:, i], y_pred[:, i]
            scores.append(mean_absolute_error(y_t, y_p))

        return scores, float(np.dot(scores, weights))