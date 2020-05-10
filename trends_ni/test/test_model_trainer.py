from pathlib import Path

import pandas as pd

from trends_ni.training.model_trainer import ModelTrainer
from trends_ni.training.models import BenchmarkModel


def test_train_model(tiny_files_structure):
    trainer = ModelTrainer()
    X_train = pd.read_csv(tiny_files_structure.raw.correlations, index_col=0)
    y_train = (
        pd.read_csv(tiny_files_structure.raw.y_train, index_col=0).fillna(0).values
    )

    results = trainer.train_model(BenchmarkModel(), X_train, y_train)
    assert results.model.params["mean_values"].all()
    assert results.train_mae
    assert results.train_weighted_mae
