import os
import pytest
from .context import omc3
from omc3.tfs_files import tfs_pandas


CURRENT_DIR = os.path.dirname(__file__)


def test_tfs_write_read(_tfs_file, _test_file):
    original = tfs_pandas.read_tfs(_tfs_file)
    tfs_pandas.write_tfs(_test_file, original)
    new = tfs_pandas.read_tfs(_test_file)
    assert original.headers == new.headers
    assert all(original.columns == new.columns)
    for column in original:
        assert all(original.loc[:, column] == new.loc[:, column])


@pytest.fixture()
def _tfs_file():
    return os.path.join(CURRENT_DIR, "..", "inputs",
                        "harmonic_results", "flat_60_15cm_b1",
                        "on_mom_file1.sdds.linx")


@pytest.fixture()
def _test_file():
    test_file = os.path.join(CURRENT_DIR, "test_file.tfs")
    try:
        yield test_file
    finally:
        if os.path.isfile(test_file):
            os.remove(test_file)
