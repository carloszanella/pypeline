from logging import getLogger
from pathlib import Path
from typing import List, Tuple

from trends_ni.dataset.dataset_builder import DatasetBuilder
from trends_ni.entities import SubjectFMRI, RawData
from trends_ni.structure import structure

import dask.dataframe as dd
import dask.array as da
import pandas as pd

DS_VERSION = "0.1.0"
log = getLogger(__name__)


class FMRIDatasetBuilder(DatasetBuilder):
    def __init__(self, version: str = DS_VERSION, save_dataset: bool = False):
        self.version = "fmri_ds_" + version
        self.save_dataset = save_dataset
        self.n_maps = 53

    def maybe_build_dataset(
        self, data: RawData, dataset_id: str, out_dir: Path
    ) -> Tuple[pd.DataFrame, pd.Series]:

        log.info(f"Building {data.set_id} dataset {dataset_id}.")
        ds_path = out_dir / dataset_id
        y = self.process_target(data)

        if ds_path.exists():
            log.info(f"Found existing dataset in path: {ds_path}")
            ddf = dd.read_parquet(ds_path)
            df = ddf.compute()
        else:
            df = self.build_dataset(data, ds_path)

        log.info(
            f"Finished building dataset. Final shape: ({len(data.ids)}, {df.shape[1]})."
        )
        return df, y

    def build_dataset(self, data: RawData, ds_path: Path):
        data.load_data_in_memory(fmri_path=structure.raw.fmri_map)
        ddf = self.make_fmri_features(data.fmri_maps)

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
