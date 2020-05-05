import numpy as np
import pandas as pd

from trends_ni.dataset.dataset import BenchmarkDataset
from trends_ni.orchestrator.pipeline_orchestrator import PipelineOrchestrator
from trends_ni.train_model.model_trainer import ModelTrainer
from trends_ni.train_model.models import BenchmarkModel


def test_pipeline_orchestrator_run_calls():
    orchestrator = PipelineOrchestrator()
    orchestrator.run_pipeline()


def test_pipeline_orchestrator_build_datasets(sample_ids, tiny_files_structure):
    orchestrator = PipelineOrchestrator(
        ds_builder=BenchmarkDataset(file_structure=tiny_files_structure),
        model_trainer=ModelTrainer(BenchmarkModel())
    )
    X_train, y_train, X_val, y_val = orchestrator.build_datasets(sample_ids[:-2], sample_ids[-2:])

    assert X_train.any()
    assert X_val.any()
    assert y_train.any()
    assert y_val.any()


def test_pipeline_orchestrator_scale_datasets(tiny_files_structure):
    orchestrator = PipelineOrchestrator(
        ds_builder=BenchmarkDataset(file_structure=tiny_files_structure),
        model_trainer=ModelTrainer(BenchmarkModel())
    )
    x_tr = pd.DataFrame(np.random.random((20, 20)))
    x_val = pd.DataFrame(np.random.random((10, 20)))
    x_tr_scaled, x_val_scaled = orchestrator.scale_datasets(x_tr, x_val)

    assert x_tr_scaled.any()
    assert x_val_scaled.any()
