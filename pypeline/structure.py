from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/Users/carloszanella/dev/study/kaggle/pypeline/pypeline/")


@dataclass
class Structure:
    data_root: Path = ROOT / "data"
    dataset: Path = data_root / "processing"
    model: Path = data_root / "model"

    @dataclass
    class raw:
        raw_data: Path = ROOT / "data" / "raw"
        correlations: Path = raw_data / "fnc.csv"
        loading: Path = raw_data / "loading.csv"
        fmri_map: Path = raw_data / "fMRI_{set_id}/{id}.mat"
        icn: Path = raw_data / "ICN_numbers.csv"
        y_train: Path = raw_data / "train_scores.csv"


structure = Structure()
