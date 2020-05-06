import os
from pathlib import Path

import pandas as pd

from trends_ni.training.model_trainer import ModelTrainer
from trends_ni.training.models import BenchmarkModel


def test_train_model(tiny_files_structure):
    trainer = ModelTrainer(BenchmarkModel())
    X_train = pd.read_csv(tiny_files_structure.raw.correlations, index_col=0)
    y_train = pd.read_csv(tiny_files_structure.raw.y_train, index_col=0).fillna(0)

    results = trainer.train_model(X_train, y_train, Path())
    assert results.model.mean_values.all()
    assert results.scores
    assert results.weighted_score


def test_save_results(tiny_files_structure):
    trainer = ModelTrainer(BenchmarkModel(), save=True)
    X_train = pd.read_csv(tiny_files_structure.raw.correlations, index_col=0)
    y_train = pd.read_csv(tiny_files_structure.raw.y_train, index_col=0).fillna(0)
    path = Path("test.pkl")

    results = trainer.train_model(X_train, y_train, path)

    assert path.exists()
    os.remove(path)