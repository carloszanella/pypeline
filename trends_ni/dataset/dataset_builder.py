from pathlib import Path
from typing import List, Tuple

from trends_ni.entities import SubjectFMRI
from trends_ni.structure import structure

import dask.dataframe as dd
import dask.array as da
import pandas as pd

DS_VERSION = "0.1.0"
DATA_DIR = structure.raw
OUT_DIR = structure.dataset


class DatasetBuilder:
    def build_dataset(self, train_ids: List[int], dataset_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def load_y_values(self) -> pd.DataFrame:
        y_data = pd.read_csv(self.data_dir / "train_scores.csv", index_col=0)
        y_filtered = y_data.loc[self.train_ids]
        y_filled = y_filtered.fillna(0)

        return y_filled


class FMRIDatasetBuilder(DatasetBuilder):
    def __init__(
        self, version: str = DS_VERSION, save_data: bool = False, set_id: str = "train"
    ):
        self.train_ids = None
        self.version = version
        self.save_data = save_data
        self.set_id = set_id
        self.data_dir = DATA_DIR
        self.out_dir = OUT_DIR

    def build_dataset(self, train_ids: List[int], dataset_id: str):
        self.train_ids = train_ids

        print(f"Building {self.set_id} dataset.")
        ds_path = self.out_dir / dataset_id

        df, y = self.maybe_build_dataset(ds_path)

        return df, y

    def maybe_build_dataset(self, ds_path: Path):
        y = self.load_y_values()

        if ds_path.exists():
            print(f"Found existing dataset in path: {ds_path}")
            ddf = dd.read_parquet(ds_path)
            return ddf.compute(), y

        else:
            subjects_fmri = self.load_subjects_data()
            ddf = self.process_data(subjects_fmri)
            y = self.load_y_values()
            print(
                f"Finished building dataset. Final shape: ({len(self.train_ids)}, {ddf.shape[1]})."
            )

            if self.save_data:
                print(f"Saving dataset to path: {ds_path}.")
                ddf.to_parquet(ds_path)

            return ddf, y

    def load_subjects_data(self):
        print("Loading subjects FMRI structures.")
        subjects = [SubjectFMRI(id=i) for i in self.train_ids]
        _ = list(map(lambda x: x.load_data(), subjects))
        print("Finished loading the structures.")

        return subjects

    def process_data(self, subjects: List[SubjectFMRI]) -> dd.DataFrame:
        print("Building features.")
        col_names = self.make_fmri_column_names()

        fmri_array = self.make_fmri_features(subjects)

        ddf = dd.from_dask_array(fmri_array, columns=col_names)

        return ddf

    def make_fmri_column_names(self, n_structures: int = 53) -> List[str]:
        mean_col_names = [f"m_{i + 1}" for i in range(n_structures)]
        std_col_names = [f"s_{i + 1}" for i in range(n_structures)]
        return mean_col_names + std_col_names

    def make_fmri_features(self, subjects: List[SubjectFMRI], n_structures: int = 53):
        rows = []

        for subj in subjects:

            mean_row = subj.fmri_map.reshape(n_structures, -1).mean(
                axis=-1, keepdims=True
            )
            std_row = subj.fmri_map.reshape(n_structures, -1).std(
                axis=-1, keepdims=True
            )

            rows.append(da.concatenate([mean_row, std_row], axis=0))

        return da.concatenate(rows, axis=-1).T
