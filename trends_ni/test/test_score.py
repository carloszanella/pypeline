from trends_ni.evaluation.score import Score
import numpy as np


def test_score_evaluate_predictions():
    y_true = np.zeros((5, 5))
    y_pred = np.ones((5, 5))

    scores, weighted_score = Score.evaluate_predictions(y_true, y_pred)
    assert sum(scores) == 5
    assert np.isclose(weighted_score, 1)
