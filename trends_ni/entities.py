from dataclasses import dataclass
from pathlib import Path
from typing import List

import dask.dataframe as dd
import h5py
import dask.array as da
import pandas as pd
from trends_ni.structure import structure


class SubjectFMRI:
    def __init__(self, id: int, set_id: str = "train"):
        self.id = id
        self.set_id = set_id
        self.fmri_map = None

    def load_data(self, fmri_path: str):
        f = h5py.File(fmri_path, "r")
        self.fmri_map = da.array(f["SM_feature"])

    def compute(self):
        return self.fmri_map.compute()


@dataclass
class RawData:
    ids: List[int]
    set_id: str = "train"
    correlations: dd.DataFrame = None
    fmri_maps: List[SubjectFMRI] = None
    loadings: dd.DataFrame = None
    icn: pd.Series = None
    y: pd.Series = None

    def load_data_in_memory(
        self,
        correlations: bool = True,
        fmri: bool = False,
        loadings: bool = False,
        icn: bool = False,
    ):
        # load y
        self.load_y()

        # maybe load correlations
        if correlations:
            self.load_correlations()

        # maybe load fmri data
        if fmri:
            self.load_fmri()

        # maybe load loading data
        if loadings:
            self.loadings = self.load_loading_data()

        # maybe load ICN
        if icn:
            self.icn = self.load_icn()

    def load_y(self, path: Path) -> pd.Series:
        y_train = pd.read_csv(path, index_col=0)
        self.y = y_train.loc[self.ids]

    def load_correlations(self, path: Path) -> dd.DataFrame:
        corr_ddf = dd.read_csv(path).set_index("Id")
        self.correlations = corr_ddf.loc[self.ids]

    def load_fmri(self, path: Path) -> List[SubjectFMRI]:
        subjects_fmri = [SubjectFMRI(id, self.set_id) for id in self.ids]
        self.fmri_maps = subjects_fmri
        _ = [subj.load_data(str(path).format(set_id=self.set_id, id=subj.id)) for subj in self.fmri_maps]

    def load_loading_data(self) -> dd.DataFrame:
        pass

    def load_icn(self) -> pd.Series:
        pass

