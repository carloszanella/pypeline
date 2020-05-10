from dataclasses import dataclass
from pathlib import Path
import numpy as np

from trends_ni.entities import RawData
import pytest


ASSETS_DIR = Path("/Users/carloszanella/dev/study/kaggle/trends/trends_ni/test/assets")


@pytest.fixture()
def tiny_files_structure():
    @dataclass
    class Structure:
        data_root: Path = ASSETS_DIR / "data"
        dataset: Path = data_root / "processing"
        model: Path = data_root / "model" / "model_{version}_{seed}_ds_{version}_{rows}_{cols}.pkl"

        @dataclass
        class raw:
            raw_data: Path = ASSETS_DIR / "data" / "raw"
            correlations: Path = raw_data / "fnc.csv"
            loading: Path = raw_data / "loading.csv"
            fmri_map: Path = raw_data / "fMRI_{set_id}/{id}.mat"
            icn: Path = raw_data / "ICN_numbers.csv"
            y_train: Path = raw_data / "train_scores.csv"

    test_structure = Structure()
    return test_structure


@pytest.fixture()
def raw_data_sample():
    raw = RawData([10001, 10002])
    return raw


@pytest.fixture()
def sample_ids():
    return np.array([10001, 10002, 10004, 10005])


@pytest.fixture()
def sample_predictions():
    return np.array(
        [
            [50.10695451, 47.61899496, 54.78646754, 47.06653996, 51.58750525],
            [50.10695451, 47.61899496, 54.78646754, 47.06653996, 51.58750525],
            [50.10695451, 47.61899496, 54.78646754, 47.06653996, 51.58750525],
            [50.10695451, 47.61899496, 54.78646754, 47.06653996, 51.58750525],
        ]
    )
