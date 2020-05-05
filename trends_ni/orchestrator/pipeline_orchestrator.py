from typing import List, Tuple
import pandas as pd
import numpy as np

from trends_ni.dataset.data_splitter import TrainValSplitter
from trends_ni.dataset.dataset_builder import DatasetBuilder
from trends_ni.entities import RawData
from trends_ni.structure import structure
from trends_ni.train_model.model_trainer import ModelTrainer


class PipelineOrchestrator:
    def __init__(
        self,
        ds_builder: DatasetBuilder,
        model_trainer: ModelTrainer,
        splitter: TrainValSplitter = TrainValSplitter(),
        seed: int = 42,
        val_split: float = 0.2,
    ):
        self.ds_builder = ds_builder
        self.model_trainer = model_trainer
        self.splitter = splitter
        self.seed = seed
        self.val_split = val_split

    def run_pipeline(self, ids: List[float]):
        np.random.seed(self.seed)
        splitter = TrainValSplitter(ids)

        # Split data
        train_ids, val_ids = splitter.split(self.val_split)

        # Build datasets:
        X_train, y_train, X_val, y_val = self.build_datasets(train_ids, val_ids)

        # Train model
        result = self.model_trainer.train_model(X_train, y_train)

    def build_datasets(
        self, train_ids: np.array, val_ids: np.array
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        train_ds_id = f"{self.ds_builder.version}_{len(train_ids)}_{self.seed}"
        train_ds_path = structure.dataset / train_ds_id

        val_ds_id = f"{self.ds_builder.version}_{len(val_ids)}_{self.seed}"
        val_ds_path = structure.dataset / val_ds_id

        X_train, y_train = self.ds_builder.maybe_build_dataset(train_ids, train_ds_path)
        X_val, y_val = self.ds_builder.maybe_build_dataset(val_ids, val_ds_path)
        
        X_train, X_val = self.scale_datasets(X_train, X_val)

        return X_train, y_train, X_val, y_val

    def get_validation_metrics(self, scores: List[float], weighted_score: float):
        print("\nValidation scores")
        print("###############")
        print("MAE: ", scores)
        print("Weighted Score: ", weighted_score)

        return scores, weighted_score

    def scale_datasets(self, X_train, X_val):
        pass
