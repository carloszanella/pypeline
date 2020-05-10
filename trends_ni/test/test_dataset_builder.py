import os
from pathlib import Path
from unittest.mock import Mock
import pytest

import pandas as pd

from trends_ni.processing.dataset_builder import DatasetBuilder
from trends_ni.processing.datasets import Dataset


@pytest.fixture()
def sample_dataset():
    class DS(Dataset):
        def __init__(self):
            super().__init__("none")

        def build_dataset(self, raw, out_path, save=False):
            pass

        def load_data(self, ids, set_id: str, fs):
            pass

    return DS


def test_dataset_maybe_build(raw_data_sample, sample_ids, sample_dataset):
    ds_builder = DatasetBuilder()
    ds = sample_dataset()
    ds.load_data = Mock(spec=ds.load_data, return_value=raw_data_sample)
    ds.build_dataset = Mock(spec=ds.build_dataset)
    ds_builder.process_target = Mock(spec=ds_builder.process_target)

    path = Path() / "test_id"
    pd.DataFrame({"one": [1, 2], "two": [2, 1]}).to_parquet(
        path, compression="UNCOMPRESSED"
    )

    ds_builder.maybe_build_dataset(sample_ids, ds, path, "test")

    ds.build_dataset.assert_not_called()
    ds_builder.process_target.assert_called_once_with(raw_data_sample)
    os.remove(path)


def test_dataset_maybe_build_2(raw_data_sample, sample_ids, sample_dataset):
    ds_builder = DatasetBuilder()
    ds = sample_dataset()
    ds.load_data = Mock(spec=ds.load_data, return_value=raw_data_sample)
    ds.build_dataset = Mock(spec=ds.build_dataset)
    ds_builder.process_target = Mock(spec=ds_builder.process_target)

    path = Path("test_id")

    ds_builder.maybe_build_dataset(sample_ids, ds, path, "test")

    ds.build_dataset.assert_called_once_with(raw_data_sample, path)
    ds_builder.process_target.assert_called_once_with(raw_data_sample)
