from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/Users/carloszanella/dev/study/kaggle/trends/trends_ni/")


@dataclass
class Structure:
    data_root: Path = ROOT / "data"
    dataset: Path = data_root / "dataset" / "{version}_{rows}_{cols}"
    model: Path = data_root / "model" / "model_{version}_{seed}_ds_{version}_{rows}_{cols}.pkl"

    @dataclass
    class raw:
        raw_data: Path = ROOT / "data" / "raw"
        fmri_map: Path = raw_data / "fMRI_{set_id}/{id}.mat"
        train_scores: Path = raw_data / "train_scores.csv"


structure = Structure()
