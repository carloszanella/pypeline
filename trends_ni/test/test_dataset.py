import os
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

from trends_ni.dataset.dataset import FMRIDataset
from trends_ni.structure import structure


def test_fmri_dataset_builder_instantiation():
    fmri_ds = FMRIDataset()
    assert fmri_ds.version
    assert fmri_ds.n_maps


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