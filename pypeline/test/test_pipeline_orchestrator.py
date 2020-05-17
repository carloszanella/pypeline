import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from pypeline.entities import TrainingResults
from pypeline.processing.dataset_builder import DatasetBuilder
from pypeline.processing.datasets import BenchmarkDataset
from pypeline.pipeline.pipeline_runner import PipelineRunner, MultipleModelRunner
from pypeline.training.model_trainer import ModelTrainer
from pypeline.training.models import BenchmarkModel


def test_pipeline_runner_run_calls(tiny_files_structure, sample_ids):
    runner = PipelineRunner(file_structure=tiny_files_structure,)
    runner.ds_builder = Mock(spec=DatasetBuilder)
    runner.model_trainer = Mock(spec=ModelTrainer)

    runner.get_model_path = Mock(spec=runner.get_model_path)
    runner.build_datasets = Mock(spec=runner.build_datasets, return_value=(0, 0, 0, 0))
    runner.splitter.split = Mock(spec=runner.splitter.split, return_value=(0, 0))
    runner.evaluate_validation_set = Mock(spec=runner.evaluate_validation_set)
    runner.run_pipeline(
        sample_ids, dataset=BenchmarkDataset(), model=BenchmarkModel(),
    )

    runner.splitter.split.assert_called_once_with(sample_ids, 0.2)
    runner.build_datasets.assert_called_once()
    runner.get_model_path.assert_called_once()
    runner.model_trainer.train_model.assert_called_once()
    runner.evaluate_validation_set.assert_called_once()


def test_pipeline_runner_build_datasets(sample_ids, tiny_files_structure):
    runner = PipelineRunner(
        file_structure=tiny_files_structure,
    )
    X_train, y_train, X_val, y_val = runner.build_datasets(
        BenchmarkDataset(), sample_ids[:-2], sample_ids[-2:]
    )

    assert X_train.any()
    assert X_val.any()
    assert y_train.any()
    assert y_val.any()


def test_pipeline_runner_scale_datasets():
    runner = PipelineRunner()
    x_tr = pd.DataFrame(np.random.random((20, 20)))
    x_val = pd.DataFrame(np.random.random((10, 20)))
    x_tr_scaled, x_val_scaled = runner.scale_datasets(x_tr, x_val)

    assert x_tr_scaled.any()
    assert x_val_scaled.any()


def test_get_model_path():
    runner = PipelineRunner()

    model = BenchmarkModel()
    dataset = BenchmarkDataset()
    path = runner.get_model_path(model=model, dataset=dataset)
    assert path
    assert model.version in path.stem
    assert dataset.version in path.stem


def test_evaluate_validation_set():
    runner = PipelineRunner()

    x_val = np.random.random((10, 20))
    y_val = np.random.random((10, 5))

    results = TrainingResults(model=BenchmarkModel())
    results.model.predict = lambda *args: np.random.random((10, 5))
    runner.evaluate_validation_set(results, x_val, y_val)

    assert results.validation_weighted_mae
    assert results.validation_mae


def test_save_results():
    runner = PipelineRunner()
    path = Path("test.pkl")
    results = TrainingResults(model_path=path)

    runner.save_results(results)

    assert path.exists()
    os.remove(path)


def test_add_info_to_results(sample_ids):
    runner = PipelineRunner()
    ds, model = BenchmarkDataset(), BenchmarkModel()
    results = TrainingResults()
    runner.add_information_to_results(results, ds, model, sample_ids, sample_ids)

    assert results.model_path
    assert results.train_ids.any()
    assert results.val_ids.any()
    assert results.dataset_version


def test_multiple_model_runner_run_pipelines(sample_ids, tiny_files_structure):
    training_list = [(BenchmarkDataset(), BenchmarkModel())] * 2
    runner = PipelineRunner(tiny_files_structure)
    runner.splitter.split = lambda *args: (sample_ids, sample_ids)
    multi_runner = MultipleModelRunner(training_list)
    results = multi_runner.run_multiple_pipelines(
        sample_ids, runner, 0.5
    )

    assert results
