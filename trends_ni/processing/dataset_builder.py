from logging import getLogger, DEBUG
from pathlib import Path
from typing import Tuple

from trends_ni.entities import RawData
import pandas as pd
import numpy as np
import dask.dataframe as dd

from trends_ni.processing.datasets import Dataset
from trends_ni.structure import Structure, structure

log = getLogger(__name__)
log.setLevel(DEBUG)


class DatasetBuilder:
    def __init__(
        self,
        file_structure: Structure = structure,
    ):
        self.structure = file_structure

    def maybe_build_dataset(
        self, ids: np.array, dataset: Dataset, dataset_path: Path, set_id: str,
    ) -> Tuple[pd.DataFrame, np.array]:
        log.info(f"Building {set_id} dataset. WIll be saved on {dataset_path}.")

        raw = dataset.load_data(ids, set_id, self.structure)
        y = self.process_target(raw)

        if dataset_path.exists():
            log.info(f"Found existing dataset in path: {dataset_path}")
            ddf = dd.read_parquet(dataset_path)
            df = ddf.compute()
        else:
            df = dataset.build_dataset(raw, dataset_path).compute()

        return df, y

    def process_target(self, data: RawData) -> np.array:
        y = data.y
        y = y.fillna(0)
        return y.values
