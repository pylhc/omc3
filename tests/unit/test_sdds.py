import os
import pytest
import numpy as np
from .context import omc3
from omc3.sdds import read_sdds, write_sdds


CURRENT_DIR = os.path.dirname(__file__)


def test_sdds_write_read(_sdds_file, _test_file):
    original = read_sdds(_sdds_file)
    write_sdds(original, _test_file)
    new = read_sdds(_test_file)
    for param_name in original.get_parameters():
        assert (original.get_parameters()[param_name].value ==
                new.get_parameters()[param_name].value)
    for array_name in original.get_arrays():
        assert np.all(original.get_arrays()[array_name].values ==
                      new.get_arrays()[array_name].values)


@pytest.fixture()
def _sdds_file():
    return os.path.join(CURRENT_DIR, "..", "inputs",
                        "tbt_files", "flat_beam1_3d.sdds")


@pytest.fixture()
def _test_file():
    test_file = os.path.join(CURRENT_DIR, "test_file.sdds")
    try:
        yield test_file
    finally:
        if os.path.isfile(test_file):
            os.remove(test_file)
