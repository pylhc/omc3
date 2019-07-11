import os
import pytest
import numpy as np
from . import context
import lhc_handler as tbt

CURRENT_DIR = os.path.dirname(__file__)
PLANES = ('X', 'Y')


def test_tbt_write_read_binary(_sdds_file, _test_file):
    origin = tbt.read_tbt(_sdds_file)
    tbt.write_tbt(_test_file, origin)
    new = tbt.read_tbt(_test_file)
    _compare_tbt(origin, new, False)


def test_tbt_write_read_ascii(_sdds_file, _test_file):
    origin = tbt.read_tbt(_sdds_file)
    tbt.write_tbt(_test_file, origin, no_binary=True)
    new = tbt.read_tbt(_test_file)
    _compare_tbt(origin, new, True)


def _compare_tbt(origin, new, no_binary):
    assert new.nturns == origin.nturns
    assert new.nbunches == origin.nbunches
    assert new.bunch_ids == origin.bunch_ids
    for index in range(origin.nbunches):
        for plane in PLANES:
            assert np.all(new.matrices[index][plane].index == origin.matrices[index][plane].index)
            origin_mat = origin.matrices[index][plane].values
            new_mat = new.matrices[index][plane].values
            if no_binary:
                ascii_precision = 0.5 / np.power(10, tbt.handler.PRINT_PRECISION)
                assert np.max(np.abs(origin_mat - new_mat)) < ascii_precision
            else:
                assert np.all(origin_mat == new_mat)


@pytest.fixture()
def _sdds_file():
    return os.path.join(CURRENT_DIR, os.pardir, "inputs", "test_file.sdds")


@pytest.fixture()
def _test_file():
    test_file = os.path.join(CURRENT_DIR, "test_file.sdds")
    try:
        yield test_file
    finally:
        if os.path.isfile(test_file):
            os.remove(test_file)

