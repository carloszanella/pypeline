from unittest.mock import Mock

import numpy as np
import pandas as pd

from trends_ni.entities import TrainingResults
from trends_ni.processing.datasets import BenchmarkDataset
from trends_ni.orchestrator.pipeline_orchestrator import PipelineOrchestrator
from trends_ni.training.model_trainer import ModelTrainer
from trends_ni.training.models import BenchmarkModel


def test_pipeline_orchestrator_run_calls(tiny_files_structure, sample_ids):
    orchestrator = PipelineOrchestrator(
        ds_builder=Mock(spec=BenchmarkDataset), model_trainer=Mock(spec=ModelTrainer)
    )
    orchestrator.get_model_path = Mock(spec=orchestrator.get_model_path)
    orchestrator.build_datasets = Mock(
        spec=orchestrator.build_datasets, return_value=(0, 0, 0, 0)
    )
    orchestrator.splitter.split = Mock(
        spec=orchestrator.splitter.split, return_value=(0, 0)
    )
    orchestrator.evaluate_validation_set = Mock(
        spec=orchestrator.evaluate_validation_set
    )
    orchestrator.run_pipeline(sample_ids)

    orchestrator.splitter.split.assert_called_once_with(sample_ids, 0.2)
    orchestrator.build_datasets.assert_called_once()
    orchestrator.get_model_path.assert_called_once()
    orchestrator.model_trainer.train_model.assert_called_once()
    orchestrator.evaluate_validation_set.assert_called_once()


def test_pipeline_orchestrator_build_datasets(sample_ids, tiny_files_structure):
    orchestrator = PipelineOrchestrator(
        ds_builder=BenchmarkDataset(file_structure=tiny_files_structure),
        model_trainer=ModelTrainer(BenchmarkModel()),
    )
    X_train, y_train, X_val, y_val = orchestrator.build_datasets(
        sample_ids[:-2], sample_ids[-2:]
    )

    assert X_train.any()
    assert X_val.any()
    assert y_train.any()
    assert y_val.any()


def test_pipeline_orchestrator_scale_datasets(tiny_files_structure):
    orchestrator = PipelineOrchestrator(
        ds_builder=BenchmarkDataset(file_structure=tiny_files_structure),
        model_trainer=ModelTrainer(BenchmarkModel()),
    )
    x_tr = pd.DataFrame(np.random.random((20, 20)))
    x_val = pd.DataFrame(np.random.random((10, 20)))
    x_tr_scaled, x_val_scaled = orchestrator.scale_datasets(x_tr, x_val)

    assert x_tr_scaled.any()
    assert x_val_scaled.any()


def test_get_model_path(tiny_files_structure):
    orchestrator = PipelineOrchestrator(
        ds_builder=BenchmarkDataset(file_structure=tiny_files_structure),
        model_trainer=ModelTrainer(BenchmarkModel()),
    )

    path = orchestrator.get_model_path()
    assert path
    assert orchestrator.model_trainer.model.version in path.stem


def test_evaluate_validation_set(tiny_files_structure):
    orchestrator = PipelineOrchestrator(
        ds_builder=BenchmarkDataset(file_structure=tiny_files_structure),
        model_trainer=ModelTrainer(BenchmarkModel()),
    )

    x_val = np.random.random((10, 20))
    y_val = np.random.random((10, 5))

    results = TrainingResults(model=BenchmarkModel())
    results.model.predict = lambda *args: np.random.random((10, 5))
    orchestrator.evaluate_validation_set(results, x_val, y_val)

    assert results.validation_weighted_mae
    assert results.validation_mae
