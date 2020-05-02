from logging import getLogger
from pathlib import Path
from typing import Tuple

from trends_ni.entities import RawData
import pandas as pd

DS_VERSION = "0.1.0"


class DatasetBuilder:
    def maybe_build_dataset(
        self, raw: RawData, dataset_id: str, out_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def build_dataset(self, data: RawData, path: Path):
        pass

    def process_target(self, data: RawData) -> pd.Series:
        pass
