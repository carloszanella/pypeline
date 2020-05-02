from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/Users/carloszanella/dev/study/kaggle/trends/trends_ni/")


@dataclass
class Structure:
    data_root: Path = ROOT / "data"
    dataset: Path = data_root / "dataset"
    model: Path = data_root / "model" / "model_{version}_{seed}_ds_{version}_{rows}_{cols}.pkl"

    @dataclass
    class raw:
        raw_data: Path = ROOT / "data" / "raw"
        correlations: Path = raw_data / "fnc.csv"
        loading: Path = raw_data / "loading.csv"
        fmri_map: Path = raw_data / "fMRI_{set_id}/{id}.mat"
        icn: Path = raw_data / "ICN_numbers.csv"
        y_train: Path = raw_data / "train_scores.csv"


structure = Structure()
