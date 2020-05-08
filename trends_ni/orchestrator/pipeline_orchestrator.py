from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from trends_ni.evaluation.score import Score
from trends_ni.processing.data_splitter import TrainValSplitter, DataSplitter
from trends_ni.processing.dataset_builder import DatasetBuilder
from trends_ni.entities import TrainingResults
from trends_ni.structure import structure, Structure
from trends_ni.training.model_trainer import ModelTrainer


class PipelineOrchestrator:
    def __init__(
        self,
        ds_builder: DatasetBuilder,
        model_trainer: ModelTrainer,
        file_structure: Structure = structure,
        splitter: DataSplitter = TrainValSplitter(),
        scaler: StandardScaler = StandardScaler(),
        seed: int = 42,
    ):
        self.ds_builder = ds_builder
        self.model_trainer = model_trainer
        self.scaler = scaler
        self.splitter = splitter
        self.seed = seed
        self.structure = file_structure

    def run_pipeline(self, ids: List[float], val_split: float = 0.2) -> TrainingResults:
        np.random.seed(self.seed)

        # Split data
        train_ids, val_ids = self.splitter.split(ids, val_split)

        # Build datasets:
        X_train, y_train, X_val, y_val = self.build_datasets(train_ids, val_ids)

        # Train model
        model_path = self.get_model_path()
        results = self.model_trainer.train_model(X_train, y_train, model_path)

        # export results TODO
        self.evaluate_validation_set(results, X_val, y_val)
        return results

    def build_datasets(
        self, train_ids: np.ndarray, val_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        train_ds_id = f"{self.ds_builder.version}_{len(train_ids)}_{self.seed}"
        train_ds_path = structure.dataset / train_ds_id

        val_ds_id = f"{self.ds_builder.version}_{len(val_ids)}_{self.seed}"
        val_ds_path = structure.dataset / val_ds_id

        X_train, y_train = self.ds_builder.maybe_build_dataset(train_ids, train_ds_path, "train")
        X_val, y_val = self.ds_builder.maybe_build_dataset(val_ids, val_ds_path, "val")
        
        X_train, X_val = self.scale_datasets(X_train, X_val)

        return X_train, y_train, X_val, y_val

    def scale_datasets(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        return X_train_scaled, X_val_scaled

    def get_model_path(self) -> Path:
        model_dir = self.structure.model
        model_id = f"{self.model_trainer.model.version}_{self.ds_builder.version}_{self.seed}.pkl"
        model_path = model_dir / model_id
        return model_path

    def evaluate_validation_set(self, results: TrainingResults, X_val: np.ndarray, y_val: np.ndarray):
        y_val_pred = results.model.predict(X_val)
        val_mae, val_weighted_mae = Score.evaluate_predictions(y_val, y_val_pred)
        results.validation_mae = val_mae
        results.validation_weighted_mae = val_weighted_mae