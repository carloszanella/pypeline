import h5py
import dask.array as da
from trends_ni.structure import structure


class SubjectFMRI:
    def __init__(self, id: int, set_id: str = "train"):
        self.id = id
        self.set_id = set_id
        self.fmri_map = None

    def load_data(self):
        fmri_path = str(structure.raw.fmri_map).format(set_id=self.set_id, id=self.id)
        f = h5py.File(fmri_path, "r")
        self.fmri_map = da.array(f["SM_feature"])