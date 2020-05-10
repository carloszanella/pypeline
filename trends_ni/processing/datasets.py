from abc import ABCMeta, abstractmethod
from logging import getLogger, DEBUG
from pathlib import Path
from typing import List

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


class Dataset(metaclass=ABCMeta):
    def __init__(
        self, version: str,
    ):
        self.version = version

    @abstractmethod
    def build_dataset(self, raw: RawData, out_path: Path, save: bool = False) -> dd.DataFrame:
        pass

    @abstractmethod
    def load_data(
        self, ids: np.ndarray, set_id: str, file_structure: Structure
    ) -> RawData:
        pass


class BenchmarkDataset(Dataset):
    def __init__(self):
        super().__init__(BM_DS_VERSION)

    def build_dataset(self, raw: RawData, out_path: Path, save: bool = False) -> dd.DataFrame:
        size = raw.y.shape[0]
        return dd.from_array(np.array([np.nan] * size).reshape(-1, 1))

    def load_data(
        self, ids: np.ndarray, set_id: str, file_structure: Structure
    ) -> RawData:
        raw = RawData(ids, set_id)
        raw.load_data_in_memory()
        return raw


class FMRIDataset(Dataset):
    def __init__(self):
        super().__init__(FMRI_DS_VERSION)
        self.n_maps = 53

    def build_dataset(self, raw: RawData, out_path: Path, save: bool = False) -> dd.DataFrame:
        raw.load_data_in_memory(fmri_path=structure.raw.fmri_map)
        ddf = self.make_fmri_features(raw.fmri_maps)

        if save:
            log.info(f"Saving dataset to path: {out_path}.")
            ddf.to_parquet(out_path)

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

    def load_data(
        self, ids: np.ndarray, set_id: str, file_structure: Structure
    ) -> RawData:
        raw = RawData(ids, set_id)
        raw.load_data_in_memory(fmri_path=file_structure.raw.fmri_map)
        return raw


class SimpleCorrelationsDataset(Dataset):
    def __init__(self):
        super().__init__(SIMPLE_CORR_VERSION)

    def build_dataset(self, raw: RawData, out_path: Path, save: bool = False) -> dd.DataFrame:
        return raw.correlations.fillna(0)

    def load_data(
            self, ids: np.ndarray, set_id: str, file_structure: Structure
    ) -> RawData:
        raw = RawData(ids, set_id)
        raw.load_data_in_memory(file_structure.raw.correlations)
        return raw
