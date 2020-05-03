import os
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

from trends_ni.dataset.dataset import FMRIDataset
from trends_ni.structure import structure


def test_fmri_dataset_builder_intantiation():
    fmri_ds = FMRIDataset()
    assert fmri_ds.version
    assert fmri_ds.n_maps


def test_fmri_dataset_maybe_build_1(raw_data_sample):
    fmri_ds = FMRIDataset()
    fmri_ds.build_dataset = Mock(spec=fmri_ds.build_dataset)
    fmri_ds.process_target = Mock(spec=fmri_ds.process_target)
    fmri_ds.maybe_build_dataset(raw_data_sample, Path("test_id"))
    fmri_ds.build_dataset.assert_called_once_with(raw_data_sample, Path("test_id"))
    fmri_ds.process_target.assert_called_once_with(raw_data_sample)


def test_fmri_dataset_maybe_build_2(raw_data_sample):
    fmri_ds = FMRIDataset()
    fmri_ds.build_dataset = Mock(spec=fmri_ds.build_dataset)
    fmri_ds.process_target = Mock(spec=fmri_ds.process_target)

    path = Path() / "test_id"
    pd.DataFrame({"one": [1, 2], "two": [2, 1]}).to_parquet(path, compression="UNCOMPRESSED")

    fmri_ds.maybe_build_dataset(raw_data_sample, path)

    fmri_ds.build_dataset.assert_not_called()
    fmri_ds.process_target.assert_called_once_with(raw_data_sample)
    os.remove(path)


def test_fmri_build_dataset_calls(raw_data_sample):
    fmri_ds = FMRIDataset()
    fmri_ds.make_fmri_features = Mock(spec=fmri_ds.make_fmri_features)
    fmri_ds.build_dataset(raw_data_sample, Path())
    fmri_ds.make_fmri_features.assert_called_once()


def test_fmri_process_data(raw_data_sample):
    fmri_ds = FMRIDataset()
    raw_data_sample.load_data_in_memory(fmri_path=structure.raw.fmri_map)
    ddf = fmri_ds.make_fmri_features(raw_data_sample.fmri_maps)
    assert ddf.compute().any().any()
    assert ddf.shape[1] == fmri_ds.n_maps * 2