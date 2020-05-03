from pathlib import Path
from unittest.mock import Mock

import numpy as np

from trends_ni.dataset.dataset_builder import DatasetBuilder


def test_dataset_builder_instantiation(raw_data_sample):
    ds_builder = DatasetBuilder()
    ds_builder.process_target = Mock(spec=ds_builder.process_target)
    ds_builder.build_dataset = lambda *args: 1
    ids = np.array([10001, 10002])
    raw_data_sample.load_data_in_memory()

    ds_builder.load_data(ids, "train")
    ds_builder.maybe_build_dataset(ids, Path("test"), "test")
    ds_builder.process_target.assert_called_once_with(None)

