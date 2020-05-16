from pathlib import Path
from unittest.mock import Mock

from trends_ni.processing.datasets import (
    FMRIDataset,
    SimpleCorrelationsDataset,
    BenchmarkDataset,
    Dataset,
    SimpleLoadingDataset, PCAWrapper,
)
from trends_ni.structure import structure


def test_dataset_abc_class():
    class DS(Dataset):
        def __init__(self):
            super().__init__("none")

        def build_dataset(self, raw, out_path, save=False):
            pass

        def load_data(self, ids, set_id: str, fs):
            pass

    ds = DS()

    assert ds.version


def test_fmri_dataset_instantiation():
    fmri_ds = FMRIDataset()
    assert fmri_ds.version
    assert fmri_ds.n_maps


def test_fmri_build_dataset_calls(raw_data_sample):
    fmri_ds = FMRIDataset()
    fmri_ds.make_fmri_features = Mock(spec=fmri_ds.make_fmri_features)
    fmri_ds.build_dataset(raw_data_sample, Path())
    fmri_ds.make_fmri_features.assert_called_once()


def test_fmri_make_fmri_features(raw_data_sample):
    fmri_ds = FMRIDataset()
    raw_data_sample.load_data_in_memory(fmri_path=structure.raw.fmri_map)
    ddf = fmri_ds.make_fmri_features(raw_data_sample.fmri_maps)
    assert ddf.compute().any().any()
    assert ddf.shape[1] == fmri_ds.n_maps * 2


def test_fmri_load_data(sample_ids):
    fmri_ds = FMRIDataset()
    assert (
        fmri_ds.load_data(sample_ids[:2], "train", structure)
        .fmri_maps[0]
        .fmri_map.compute()
        .any()
    )


def test_simple_corr_build_dataset(raw_data_sample, tiny_files_structure):
    simple_corr_ds = SimpleCorrelationsDataset()
    raw_data_sample.load_data_in_memory(
        correlations_path=tiny_files_structure.raw.correlations
    )
    ddf = simple_corr_ds.build_dataset(raw_data_sample, Path("test_path"))
    assert ddf.compute().any().any()


def test_simple_corr_load_data(tiny_files_structure, sample_ids):
    simple_corr_ds = SimpleCorrelationsDataset()
    simple_corr_ds.structure = tiny_files_structure
    assert (
        simple_corr_ds.load_data(sample_ids, "test", tiny_files_structure)
        .correlations.compute()
        .any()
        .any()
    )


def test_benchmark_model_ds(tiny_files_structure, raw_data_sample, sample_ids):
    bm_ds = BenchmarkDataset()
    raw = bm_ds.load_data(sample_ids, "train", tiny_files_structure)
    df = bm_ds.build_dataset(raw, Path())
    assert not df.compute().any().any()
    assert raw.y.any().any()


def test_simple_loading_build_dataset(raw_data_sample, tiny_files_structure):
    simple_loading_ds = SimpleLoadingDataset()
    raw_data_sample.load_data_in_memory(loadings_path=tiny_files_structure.raw.loading)
    ddf = simple_loading_ds.build_dataset(raw_data_sample, Path("test_path"))
    assert ddf.compute().any().any()


def test_simple_loading_load_data(tiny_files_structure, sample_ids):
    simple_loading_ds = SimpleLoadingDataset()
    simple_loading_ds.structure = tiny_files_structure
    assert (
        simple_loading_ds.load_data(sample_ids, "test", tiny_files_structure)
        .loadings.compute()
        .any()
        .any()
    )


def test_pca_wrapper(raw_data_sample, tiny_files_structure):
    simple_loading_ds = SimpleLoadingDataset()
    raw_data_sample.load_data_in_memory(loadings_path=tiny_files_structure.raw.loading)
    pca_wrapper = PCAWrapper(simple_loading_ds, 2)
    pca_wrapper.pca = Mock(spec=pca_wrapper.pca)
    pca_wrapper.make_pca_ddf = Mock(spec=pca_wrapper.make_pca_ddf)

    pca_ds = pca_wrapper.build_dataset(raw_data_sample, Path("test"))

    pca_wrapper.pca.fit_transform.assert_called_once()
    pca_wrapper.make_pca_ddf.assert_called_once()


def test_pca_make_ddf(raw_data_sample, tiny_files_structure, X):
    simple_loading_ds = SimpleLoadingDataset()
    raw_data_sample.load_data_in_memory(loadings_path=tiny_files_structure.raw.loading)
    pca_wrapper = PCAWrapper(simple_loading_ds, 2)

    pca_ddf = pca_wrapper.make_pca_ddf(X)
    assert (pca_ddf.columns == ["pca-0", "pca-1"]).all()