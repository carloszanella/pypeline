from logging import getLogger
from pathlib import Path
from typing import Tuple

from trends_ni.entities import RawData
import pandas as pd
import dask.dataframe as dd

log = getLogger(__name__)


class DatasetBuilder:
    def __init__(self):
        self.version = None

    def maybe_build_dataset(
        self, data: RawData, dataset_path: Path
    ) -> Tuple[pd.DataFrame, pd.Series]:
        log.info(f"Building {data.set_id} dataset. WIll be saved on {dataset_path}.")
        y = self.process_target(data)

        if dataset_path.exists():
            log.info(f"Found existing dataset in path: {dataset_path}")
            ddf = dd.read_parquet(dataset_path)
            df = ddf.compute()
        else:
            df = self.build_dataset(data, dataset_path)

        return df, y

    def build_dataset(self, data: RawData, path: Path):
        pass

    def process_target(self, data: RawData) -> pd.Series:
        y = data.y
        y = y.fillna(0)
        return y

