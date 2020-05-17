import pickle
from logging import getLogger, DEBUG
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from pypeline.evaluation.score import Score
from pypeline.processing.data_splitter import TrainValSplitter, DataSplitter
from pypeline.processing.dataset_builder import DatasetBuilder
from pypeline.entities import TrainingResults
from pypeline.processing.datasets import Dataset
from pypeline.structure import structure, Structure
from pypeline.training.model_trainer import ModelTrainer
from pypeline.training.models import Model

log = getLogger(__name__)
log.setLevel(DEBUG)


class PipelineRunner:
    def __init__(
        self,
        file_structure: Structure = structure,
        splitter: DataSplitter = TrainValSplitter(),
        scaler: StandardScaler = StandardScaler(),
        seed: int = 42,
        save_results: bool = False,
        save_dataset: bool = False,
    ):
        self.ds_builder = DatasetBuilder(file_structure)
        self.model_trainer = ModelTrainer()
        self.scaler = scaler
        self.splitter = splitter
        self.seed = seed
        self.structure = file_structure
        self.save_res = save_results
        self.save_dataset = save_dataset
        np.random.seed(self.seed)

    def run_pipeline(
        self, ids: List[float], dataset: Dataset, model: Model, val_split: float = 0.2,
    ) -> TrainingResults:

        train_ids, val_ids = self.splitter.split(ids, val_split)

        X_train, y_train, X_val, y_val = self.build_datasets(
            dataset, train_ids, val_ids
        )

        results = self.model_trainer.train_model(model, X_train, y_train)

        self.add_information_to_results(results, dataset, model, train_ids, val_ids)

        self.evaluate_validation_set(results, X_val, y_val)

        if self.save_res:
            self.save_results(results)

        return results

    def build_datasets(
        self, dataset: Dataset, train_ids: np.ndarray, val_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        train_ds_id = f"{dataset.version}_{len(train_ids)}_{self.seed}"
        train_ds_path = structure.dataset / train_ds_id

        val_ds_id = f"{dataset.version}_{len(val_ids)}_{self.seed}"
        val_ds_path = structure.dataset / val_ds_id

        X_train, y_train = self.ds_builder.maybe_build_dataset(
            train_ids, dataset, train_ds_path, "train"
        )
        X_val, y_val = self.ds_builder.maybe_build_dataset(
            val_ids, dataset, val_ds_path, "val"
        )

        if self.save_dataset:
            log.info(f"Saving training dataset to path: {train_ds_path}.")
            X_train.to_parquet(train_ds_path)
            log.info(f"Saving training dataset to path: {val_ds_path}.")
            X_val.to_parquet(val_ds_path)

        X_train, X_val = self.scale_datasets(X_train, X_val)

        return X_train, y_train, X_val, y_val

    def scale_datasets(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        return X_train_scaled, X_val_scaled

    def add_information_to_results(
        self,
        results: TrainingResults,
        dataset: Dataset,
        model: Model,
        train_ids: np.ndarray,
        val_ids: np.ndarray,
    ):
        results.model_path = self.get_model_path(model, dataset)
        results.train_ids = train_ids
        results.val_ids = val_ids
        results.dataset_version = dataset.version

    def get_model_path(self, model: Model, dataset: Dataset) -> Path:
        model_dir = self.structure.model
        model_id = f"{model.version}_{dataset.version}_{self.seed}.pkl"
        model_path = model_dir / model_id
        return model_path

    def evaluate_validation_set(
        self, results: TrainingResults, X_val: np.ndarray, y_val: np.ndarray
    ):
        y_val_pred = results.model.predict(X_val)
        val_mae, val_weighted_mae = Score.evaluate_predictions(y_val, y_val_pred)
        results.validation_mae = val_mae
        results.validation_weighted_mae = val_weighted_mae
        results.print_score_results()

    def save_results(self, results: TrainingResults):
        results.model_path.parent.mkdir(exist_ok=True)
        log.info(f"Saving model on path {results.model_path}")

        with open(results.model_path, "wb") as fp:
            pickle.dump(results, fp)


class MultipleModelRunner:
    def __init__(self, training_list: List[Tuple[Dataset, Model]]):
        self.training_list = training_list

    def run_multiple_pipelines(
        self, ids: List[float], runner: PipelineRunner, val_split: float = 0.2,
    ) -> List[TrainingResults]:
        results = []

        for ds, model in self.training_list:
            results.append(runner.run_pipeline(ids, ds, model, val_split))

        return results
