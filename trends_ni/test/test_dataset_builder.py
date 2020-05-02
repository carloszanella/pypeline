from pathlib import Path

from trends_ni.dataset.dataset_builder import DatasetBuilder


def test_dataset_builder_instantiation(raw_data_sample):
    ds_builder = DatasetBuilder()
    ds_builder.build_dataset = lambda *args: 1
    raw_data_sample.load_data_in_memory()
    ds_builder.maybe_build_dataset(raw_data_sample, Path("test"))
    assert ds_builder.process_target(raw_data_sample).any().all()

