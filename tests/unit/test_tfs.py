import os
import pytest
from .context import omc3
from omc3.tfs import read_tfs, write_tfs


CURRENT_DIR = os.path.dirname(__file__)


def test_tfs_write_read(_tfs_file, _test_file):
    original = read_tfs(_tfs_file)
    write_tfs(_test_file, original)
    new = read_tfs(_test_file)
    assert original.headers == new.headers
    assert all(original.columns == new.columns)
    for column in original:
        assert all(original.loc[:, column] == new.loc[:, column])


@pytest.fixture()
def _tfs_file():
    return os.path.join(CURRENT_DIR, "..", "inputs", "test_file.tfs")


@pytest.fixture()
def _test_file():
    test_file = os.path.join(CURRENT_DIR, "test_file.tfs")
    try:
        yield test_file
    finally:
        if os.path.isfile(test_file):
            os.remove(test_file)
