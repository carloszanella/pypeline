from pathlib import Path

from trends_ni.structure import structure
import pytest


ASSETS_DIR = Path("/Users/carloszanella/dev/study/kaggle/trends/trends_ni/test/assets")


@pytest.fixture()
def tiny_files_structure():
    test_structure = structure
    test_structure.ROOT = ASSETS_DIR
    return test_structure
