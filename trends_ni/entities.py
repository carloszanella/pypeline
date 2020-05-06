from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import List

import dask.dataframe as dd
import h5py
import dask.array as da
import pandas as pd

from trends_ni.structure import structure
from trends_ni.training.models import Model

log = getLogger(__name__)


@dataclass
class SubjectFMRI:

    id: int
    set_id: str = "train"
    fmri_map: da.array = None

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
        correlations_path: Path = None,
        y_path: Path = structure.raw.y_train,
        fmri_path: Path = None,
        loadings_path: Path = None,
        icn_path: Path = None,
    ):
        # load y
        self.load_y(y_path)

        # maybe load correlations
        if correlations_path:
            self.load_correlations(correlations_path)

        # maybe load fmri data
        if fmri_path:
            self.load_fmri(fmri_path)

        # maybe load loading data
        if loadings_path:
            self.loadings = self.load_loading_data(loadings_path)

        # maybe load ICN
        if icn_path:
            self.icn = self.load_icn(icn_path)

    def load_y(self, path: Path):
        y_train = pd.read_csv(path, index_col=0)
        self.y = y_train.loc[self.ids]

    def load_correlations(self, path: Path):
        corr_ddf = dd.read_csv(path).set_index("Id")
        self.correlations = corr_ddf.loc[self.ids]

    def load_fmri(self, path: Path):
        subjects_fmri = [SubjectFMRI(id, self.set_id) for id in self.ids]
        self.fmri_maps = subjects_fmri
        _ = [
            subj.load_data(str(path).format(set_id=self.set_id, id=subj.id))
            for subj in self.fmri_maps
        ]

    def load_loading_data(self, path: Path):
        loading_ddf = dd.read_csv(path).set_index("Id")
        self.loadings = loading_ddf.loc[self.ids]

    def load_icn(self, path: Path):
        icn = pd.read_csv(path)
        self.icn = icn.values


@dataclass
class TrainingResults:
    model_version: str = None
    model: Model = None
    scores: List[float] = None
    weighted_score: float = None

    def print_score_results(self):
        log.info(f"Training scores for model {self.model_version}")
        log.info("#########################################")
        log.info("MAE: ", self.scores)
        log.info("Weighted Score: ", self.weighted_score)
