from pathlib import Path

import numpy as np
import pytest
import turn_by_turn as tbt

from omc3.definitions.constants import PLANES
from omc3.tbt_converter import converter_entrypoint

INPUTS_DIR = Path(__file__).parent.parent / "inputs"
ASCII_PRECISION = 0.5 / np.power(10, tbt.constants.PRINT_PRECISION)


@pytest.mark.basic
def test_converter_one_file(_sdds_file, _test_file):
    converter_entrypoint(files=[_sdds_file], outputdir=_test_file.parent)
    origin = tbt.read_tbt(_sdds_file)
    new = tbt.read_tbt(f"{_test_file}.sdds")
    _compare_tbt(origin, new, False)


@pytest.mark.basic
@pytest.mark.parametrize("dropped_elements", [["BPMSX.4R2.B"], ["BPMSX.4L2.B1", "BPMSW.1R2.B1"]])
def test_converter_drop_elements(_sdds_file, _test_file, dropped_elements):
    converter_entrypoint(
        files=[_sdds_file],
        outputdir=_test_file.parent,
        drop_elements=dropped_elements,
    )
    new = tbt.read_tbt(f"{_test_file}.sdds")
    for transverse_data in new.matrices:
        for dataframe in (transverse_data.X, transverse_data.Y):
            for element in dropped_elements:
                assert element not in dataframe.index


@pytest.mark.basic
@pytest.mark.parametrize("unknown_element", ["NOT_IN_DATA", "QBX.P4.T1", "INVALID"])
def test_converter_warns_on_not_found_drop_elements(
    _sdds_file, _test_file, unknown_element, caplog
):
    converter_entrypoint(
        files=[_sdds_file],
        outputdir=_test_file.parent,
        drop_elements=[unknown_element],
    )
    assert f"Element '{unknown_element}' could not be found, skipped" in caplog.text


@pytest.mark.basic
def test_converter_one_file_with_noise(_sdds_file, _test_file):
    np.random.seed(2019)
    noiselevel = 0.0005
    converter_entrypoint(files=[_sdds_file], outputdir=_test_file.parent, noise_levels=[noiselevel])
    origin = tbt.read_tbt(_sdds_file)
    new = tbt.read_tbt(f"{_test_file}_n{noiselevel}.sdds")
    _compare_tbt(origin, new, True, noiselevel * 10)


@pytest.mark.basic
def test_converter_more_files(_sdds_file, _test_file):
    rep = 2
    converter_entrypoint(files=[_sdds_file], outputdir=_test_file.parent, realizations=rep)
    origin = tbt.read_tbt(_sdds_file)
    for i in range(rep):
        new = tbt.read_tbt(f"{_test_file}_r{i}.sdds")
        _compare_tbt(origin, new, False)


@pytest.mark.basic
def test_converter_more_files_with_noise(_sdds_file, _test_file):
    np.random.seed(2019)
    rep = 2
    noiselevel = 0.0005
    converter_entrypoint(
        files=[_sdds_file], outputdir=_test_file.parent, realizations=rep, noise_levels=[noiselevel]
    )
    origin = tbt.read_tbt(_sdds_file)
    for i in range(rep):
        new = tbt.read_tbt(f"{_test_file}_n{noiselevel}_r{i}.sdds")
        _compare_tbt(origin, new, True, noiselevel * 10)


def _compare_tbt(
    origin: tbt.TbtData, new: tbt.TbtData, no_binary: bool, max_deviation=ASCII_PRECISION
) -> None:
    assert new.nturns == origin.nturns
    assert new.nbunches == origin.nbunches
    assert new.bunch_ids == origin.bunch_ids
    for index in range(origin.nbunches):
        for plane in PLANES:
            assert np.all(new.matrices[index][plane].index == origin.matrices[index][plane].index)
            origin_mat = origin.matrices[index][plane].to_numpy()
            new_mat = new.matrices[index][plane].to_numpy()
            if no_binary:
                assert np.max(np.abs(origin_mat - new_mat)) < max_deviation
            else:
                assert np.all(origin_mat == new_mat)


@pytest.fixture()
def _sdds_file() -> Path:
    return INPUTS_DIR / "test_file.sdds"


@pytest.fixture()
def _test_file(tmp_path) -> Path:
    yield tmp_path / "test_file"
