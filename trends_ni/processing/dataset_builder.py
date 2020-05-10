from abc import ABCMeta, abstractmethod
from logging import getLogger, DEBUG
from pathlib import Path
from typing import Tuple

from trends_ni.entities import RawData
import pandas as pd
import numpy as np
import dask.dataframe as dd

from trends_ni.structure import Structure, structure

log = getLogger(__name__)
log.setLevel(DEBUG)


class DatasetBuilder(metaclass=ABCMeta):
    def __init__(
        self,
        version: str,
        save_dataset: bool = False,
        file_structure: Structure = structure,
    ):
        self.version = version
        self.save_dataset = save_dataset
        self.structure = file_structure

    def maybe_build_dataset(
        self, ids: np.array, dataset_path: Path, set_id: str,
    ) -> Tuple[pd.DataFrame, np.array]:
        log.info(f"Building {set_id} dataset. WIll be saved on {dataset_path}.")

        raw = self.load_data(ids, set_id)
        y = self.process_target(raw)

        if dataset_path.exists():
            log.info(f"Found existing dataset in path: {dataset_path}")
            ddf = dd.read_parquet(dataset_path)
            df = ddf.compute()
        else:
            df = self.build_dataset(raw, dataset_path)

        return df, y

    @abstractmethod
    def build_dataset(self, raw: RawData, out_path: Path) -> dd.DataFrame:
        pass

    def process_target(self, data: RawData) -> np.array:
        y = data.y
        y = y.fillna(0)
        return y.values

    @abstractmethod
    def load_data(self, ids: np.array, set_id: str) -> RawData:
        pass
