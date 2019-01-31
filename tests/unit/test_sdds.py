import os
import io
import pytest
import numpy as np
from . import context
from sdds.reader import (
    read_sdds,
    _gen_words,
    _get_def_as_dict,
    _read_header,
    _sort_definitions,
)
from sdds.writer import write_sdds, _sdds_def_as_str
from sdds.classes import Parameter, Column, Array


CURRENT_DIR = os.path.dirname(__file__)


def test_sdds_write_read(_sdds_file, _test_file):
    original = read_sdds(_sdds_file)
    write_sdds(original, _test_file)
    new = read_sdds(_test_file)
    for definition, value in original:
        new_def, new_val = new[definition.name]
        assert new_def.name == definition.name
        assert new_def.type == definition.type
        assert np.all(value == new_val)


def test_read_word():
    test_str = b"""   test1\ntest2 test3,\ttest4
    """
    word_gen = _gen_words(io.BytesIO(test_str))
    assert next(word_gen) == "test1"
    assert next(word_gen) == "test2"
    assert next(word_gen) == "test3,"
    assert next(word_gen) == "test4"


def test_read_no_more():
    test_str = b""
    with pytest.raises(StopIteration):
        next(_gen_words(io.BytesIO(test_str)))


def test_def_as_dict():
    test_str = (b"test1=value1,  test2= value2, \n"
                b"test3=value3, &end")
    word_gen = _gen_words(io.BytesIO(test_str))
    def_dict = _get_def_as_dict(word_gen)
    assert def_dict["test1"] == "value1"
    assert def_dict["test2"] == "value2"
    assert def_dict["test3"] == "value3"


def test_sort_defs():
    param1 = Parameter(name="param1", type_="long")
    param2 = Parameter(name="param2", type_="long")
    array1 = Array(name="array1", type_="long")
    array2 = Array(name="array2", type_="long")
    col1 = Column(name="col1", type_="long")
    col2 = Column(name="col2", type_="long")
    unsorted = [array1, col1, param1, param2, array2, col2]
    sorted_ = [param1, param2, array1, array2, col1, col2]
    assert sorted_ == _sort_definitions(unsorted)


def test_read_header():
    test_head = b"""
SDDS1
!# big-endian
&parameter name=acqStamp, type=double, &end
&parameter name=nbOfCapTurns, type=long, &end
&array name=horPositionsConcentratedAndSorted, type=float, &end
&array
    name=verBunchId,
    type=long,
    field_length=3,
&end
&data mode=binary, &end
"""
    test_data = {"acqStamp": "double", "nbOfCapTurns": "long",
                 "horPositionsConcentratedAndSorted": "float",
                 "verBunchId": "long"}
    version, definitions, _, data = _read_header(io.BytesIO(test_head))
    assert version == "SDDS1"
    assert data.mode == "binary"
    for definition in definitions:
        assert definition.type == test_data[definition.name]


def _write_read_header():
    original = Parameter(name="param1", type="str")
    encoded = _sdds_def_as_str(original)
    word_gen = _gen_words(io.BytesIO(encoded))
    def_dict = _get_def_as_dict(word_gen)
    assert def_dict["name"] == original.name
    assert def_dict["type"] == original.type


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
