from trends_ni.entities import SubjectFMRI, RawData
import pytest

from trends_ni.structure import structure


@pytest.fixture()
def sample_ids():
    return [10001, 10002]


def test_subject_fmri_load_data(sample_ids):
    subj = SubjectFMRI(id=10001)
    subj.load_data(str(structure.raw.fmri_map).format(set_id="train", id=10001))
    assert subj.fmri_map.compute().any()


def test_subject_fmri_compute(sample_ids):
    subj = SubjectFMRI(id=10001)
    subj.load_data(str(structure.raw.fmri_map).format(set_id="train", id=10001))
    assert subj.compute().any()


def test_raw_data_init(sample_ids):
    raw = RawData(sample_ids)
    assert raw.ids


def test_raw_data_load_data(sample_ids):
    raw = RawData(sample_ids)
    raw.load_data_in_memory(correlations=True)
    assert raw.correlations.any()
    assert raw.y.any()
    assert not raw.icn.any()
    assert not raw.fmri_maps.any()


def test_raw_load_y(sample_ids, tiny_files_structure):
    raw = RawData(sample_ids)
    raw.load_y(tiny_files_structure.raw.y_train)
    assert raw.y.all().all()


def test_raw_load_correlations(sample_ids, tiny_files_structure):
    raw = RawData(sample_ids)
    raw.load_correlations(tiny_files_structure.raw.correlations)
    assert raw.correlations.compute().any().all()


def test_raw_load_loading_data(sample_ids):
    raw = RawData(sample_ids)
    raw.load_loading_data()
    assert raw.loadings.any()


def test_raw_load_fmri_map(sample_ids):
    raw = RawData(sample_ids)
    raw.load_fmri(structure.raw.fmri_map)
    assert raw.fmri_maps
    assert raw.fmri_maps[0].compute().any()


def test_raw_load_icn(sample_ids):
    raw = RawData(sample_ids)
    raw.load_icn()
    assert raw.icn.any()
