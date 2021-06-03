from pathlib import Path

import pandas as pd
import numpy as np

from omc3.correction.response_io import write_fullresponse, read_fullresponse, write_varmap, read_varmap
import pytest


INPUT_DIR = Path(__file__).parent.parent / "inputs"
INJ_BEAM1_MODEL = INPUT_DIR / "models" / "inj_beam1"
FULLRESPONSE_PATH = INJ_BEAM1_MODEL / "fullresponse.h5"
VARMAP_PATH = INJ_BEAM1_MODEL / "varmap_lhcb1_MQY.h5"


# Fullresponse -----------------------------------------------------------------


@pytest.mark.basic
def test_fullresponse_read_all():
    response = read_fullresponse(FULLRESPONSE_PATH)
    assert len(response) == 13
    for key, df in response.items():
        assert isinstance(key, str)
        assert isinstance(df, pd.DataFrame)
        assert len(df)


@pytest.mark.basic
def test_fullresponse_read_some():
    optics_parameters = ["BETX", "BETY", "F1001I"]
    response = read_fullresponse(FULLRESPONSE_PATH, optics_parameters=optics_parameters)
    assert len(response) == len(optics_parameters)
    for key in optics_parameters:
        assert key in response
        assert isinstance(response[key], pd.DataFrame)


@pytest.mark.basic
def test_fullresponse_read_write(tmp_path):
    params = ["A", "B", "C"]
    matrix_dict = {p: pd.DataFrame(np.random.random([10*i, 17*i])) for i, p in enumerate(params, start=1)}
    out_path = tmp_path / "responsetest.h5"
    write_fullresponse(out_path, matrix_dict)
    new_matrix_dict = read_fullresponse(out_path)

    for p in params:
        assert matrix_dict[p].equals(new_matrix_dict[p])


@pytest.mark.basic
def test_fullresponse_read_fail():
    with pytest.raises(IOError):
        read_fullresponse(Path('notaresponse.h5'))


@pytest.mark.basic
def test_fullresponse_parameter_fail():
    with pytest.raises(ValueError) as e:
        read_fullresponse(VARMAP_PATH, optics_parameters=["BETX", "SOME", "OTHER"])
    assert "SOME" in str(e.value)
    assert "OTHER" in str(e.value)


# Varmap -----------------------------------------------------------------------


@pytest.mark.basic
def test_varmap_read_all():
    varmap = read_varmap(VARMAP_PATH)
    assert len(varmap) == 2
    for key, df in varmap.items():
        assert isinstance(key, str)
        for subkey in varmap[key]:
            assert isinstance(subkey, str)
            assert isinstance(varmap[key][subkey], pd.Series)


@pytest.mark.basic
def test_varmap_read_some():
    k_values = ["K1", "K1L"]
    varmap = read_varmap(VARMAP_PATH, k_values=k_values)
    assert len(varmap) == len(k_values)
    for key in k_values:
        assert key in varmap
        for subkey in varmap[key]:
            assert isinstance(varmap[key][subkey], pd.Series)


@pytest.mark.basic
def test_varmap_read_write(tmp_path):
    params = ["A", "B", "C"]
    sub_params = ['1', '2', '3']
    matrix_dict = {p: {sp: pd.Series(np.random.random([13*i])) for i, sp in enumerate(sub_params, start=1)} for p in params}
    out_path = tmp_path / "varmaptest.h5"
    write_varmap(out_path, matrix_dict)
    new_matrix_dict = read_varmap(out_path)

    for p in params:
        for sp in sub_params:
            assert matrix_dict[p][sp].equals(new_matrix_dict[p][sp])


@pytest.mark.basic
def test_varmap_read_fail():
    with pytest.raises(IOError):
        read_varmap(Path('notavarmap.h5'))


@pytest.mark.basic
def test_varmap_kvalues_fail():
    with pytest.raises(ValueError) as e:
        read_varmap(VARMAP_PATH, k_values=["K1S", "K0", "K1L"])
    assert "K0" in str(e.value)
    assert "K1S" in str(e.value)

