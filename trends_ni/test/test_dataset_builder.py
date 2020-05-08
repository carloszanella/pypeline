import os
from pathlib import Path
from unittest.mock import Mock
import pytest

import pandas as pd

from trends_ni.processing.dataset_builder import DatasetBuilder


@pytest.fixture()
def sample_dataset():
    class DS(DatasetBuilder):
        def __init__(self):
            self.version = None
            self.save_dataset = None
            self.structure = None

        def build_dataset(self, raw, out_path):
            pass

        def load_data(self, ids, set_id: str):
            pass

    return DS


def test_dataset_maybe_build(raw_data_sample, sample_ids, sample_dataset):
    ds_builder = sample_dataset()
    ds_builder.load_data = Mock(spec=ds_builder.load_data, return_value=raw_data_sample)
    ds_builder.build_dataset = Mock(spec=ds_builder.build_dataset)
    ds_builder.process_target = Mock(spec=ds_builder.process_target)

    path = Path() / "test_id"
    pd.DataFrame({"one": [1, 2], "two": [2, 1]}).to_parquet(
        path, compression="UNCOMPRESSED"
    )

    ds_builder.maybe_build_dataset(sample_ids, path, "test")

    ds_builder.build_dataset.assert_not_called()
    ds_builder.process_target.assert_called_once_with(raw_data_sample)
    os.remove(path)


def test_dataset_maybe_build_2(raw_data_sample, sample_ids, sample_dataset):
    ds_builder = sample_dataset()
    ds_builder.load_data = Mock(spec=ds_builder.load_data, return_value=raw_data_sample)
    ds_builder.build_dataset = Mock(spec=ds_builder.build_dataset)
    ds_builder.process_target = Mock(spec=ds_builder.process_target)

    path = Path("test_id")

    ds_builder.maybe_build_dataset(sample_ids, path, "test")

    ds_builder.build_dataset.assert_called_once_with(raw_data_sample, path)
    ds_builder.process_target.assert_called_once_with(raw_data_sample)
