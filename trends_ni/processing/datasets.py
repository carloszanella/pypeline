from logging import getLogger, DEBUG
from pathlib import Path
from typing import List

from trends_ni.processing.dataset_builder import DatasetBuilder
from trends_ni.entities import SubjectFMRI, RawData
from trends_ni.structure import structure, Structure

import dask.dataframe as dd
import numpy as np
import dask.array as da

FMRI_DS_VERSION = "fmri_ds_0.1"
SIMPLE_CORR_VERSION = "rcorr_ds_0.1"
BM_DS_VERSION = "bm_ds_0.1"
log = getLogger(__name__)
log.setLevel(DEBUG)


class BenchmarkDataset(DatasetBuilder):
    def __init__(
        self, save_dataset: bool = False, file_structure: Structure = structure
    ):
        super().__init__(BM_DS_VERSION, save_dataset, file_structure)

    def build_dataset(self, raw: RawData, out_path: Path) -> dd.DataFrame:
        size = raw.y.shape[0]
        return dd.from_array(np.array([np.nan] * size).reshape(-1, 1))

    def load_data(self, ids: np.ndarray, set_id: str) -> RawData:
        raw = RawData(ids, set_id)
        raw.load_data_in_memory(y_path=self.structure.raw.y_train)
        return raw


class FMRIDataset(DatasetBuilder):
    def __init__(
        self, save_dataset: bool = False, file_structure: Structure = structure
    ):
        super().__init__(FMRI_DS_VERSION, save_dataset, file_structure)
        self.n_maps = 53

    def build_dataset(self, raw: RawData, ds_path: Path) -> dd.DataFrame:
        raw.load_data_in_memory(fmri_path=structure.raw.fmri_map)
        ddf = self.make_fmri_features(raw.fmri_maps)

        if self.save_dataset:
            log.info(f"Saving dataset to path: {ds_path}.")
            ddf.to_parquet(ds_path)

        return ddf

    def make_fmri_column_names(self) -> List[str]:
        mean_col_names = [f"m_{i + 1}" for i in range(self.n_maps)]
        std_col_names = [f"s_{i + 1}" for i in range(self.n_maps)]
        return mean_col_names + std_col_names

    def make_fmri_features(self, subjects: List[SubjectFMRI]) -> dd.DataFrame:
        rows = []
        col_names = self.make_fmri_column_names()

        for subj in subjects:

            mean_row = subj.fmri_map.reshape(self.n_maps, -1).mean(
                axis=-1, keepdims=True
            )
            std_row = subj.fmri_map.reshape(self.n_maps, -1).std(axis=-1, keepdims=True)

            rows.append(da.concatenate([mean_row, std_row], axis=0))

        fmri_array = da.concatenate(rows, axis=-1).T

        return dd.from_dask_array(fmri_array, columns=col_names)

    def load_data(self, ids: np.ndarray, set_id: str) -> RawData:
        raw = RawData(ids, set_id)
        raw.load_data_in_memory(fmri_path=self.structure.raw.fmri_map)
        return raw


class SimpleCorrelationsDataset(DatasetBuilder):
    def __init__(
        self, save_dataset: bool = False, file_structure: Structure = structure
    ):
        super().__init__(SIMPLE_CORR_VERSION, save_dataset, file_structure)

    def build_dataset(self, raw: RawData, out_path: Path) -> dd.DataFrame:
        return raw.correlations

    def load_data(self, ids: np.ndarray, set_id: str) -> RawData:
        raw = RawData(ids, set_id)
        raw.load_data_in_memory(self.structure.raw.correlations)
        return raw
