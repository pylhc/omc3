from pathlib import Path

import pytest
import tfs

from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import (
    AMP_BETA_NAME,
    BETA_NAME,
    DELTA,
    DISPERSION_NAME,
    ERR,
    MDL,
    NORM_DISP_NAME,
    ORBIT_NAME,
    PHASE_NAME,
    TOTAL_PHASE_NAME,
)
from omc3.scripts.betabeatsrc_output_converter import converter_entrypoint
from tests.conftest import cli_args

INPUT_DIR = Path(__file__).parent.parent / "inputs"
BBRSC_OUTPUTS = INPUT_DIR / "bbsrc_output_converter"


# ----- Tests ----- #


@pytest.mark.basic
@pytest.mark.parametrize("suffix", ["", "_free", "_free2"])
def test_betabeatsrc_output_converter(tmp_path, suffix):
    converter_entrypoint(inputdir=str(BBRSC_OUTPUTS), outputdir=str(tmp_path), suffix=suffix)
    _assert_correct_files_are_present(tmp_path)

    for plane in PLANES:
        _assert_correct_beta_amp_columns(tmp_path, plane)
        _assert_correct_beta_phase_columns(tmp_path, plane)
        _assert_correct_phase_columns(tmp_path, plane)
        _assert_correct_total_phase_columns(tmp_path, plane)
        _assert_correct_closed_orbit_columns(tmp_path, plane)
        _assert_correct_dispersion_columns(tmp_path, "X")
    _assert_correct_normalized_dispersion_columns(tmp_path, "X")  # no norm disp in Y plane

    for rdt in ["1001", "1010"]:
        _assert_correct_coupling_columns(tmp_path, rdt)


@pytest.mark.basic
@pytest.mark.parametrize("suffix", ["", "_free", "_free2"])
def test_betabeatsrc_output_converter_commandline(tmp_path, suffix):
    with cli_args(
        "--inputdir", str(BBRSC_OUTPUTS.absolute()), "--outputdir", str(tmp_path), "--suffix", suffix
    ):
        converter_entrypoint()
    _assert_correct_files_are_present(tmp_path)

    for plane in PLANES:
        _assert_correct_beta_amp_columns(tmp_path, plane)
        _assert_correct_beta_phase_columns(tmp_path, plane)
        _assert_correct_phase_columns(tmp_path, plane)
        _assert_correct_total_phase_columns(tmp_path, plane)
        _assert_correct_closed_orbit_columns(tmp_path, plane)
        _assert_correct_dispersion_columns(tmp_path, "X")
    _assert_correct_normalized_dispersion_columns(tmp_path, "X")  # no norm disp in Y plane

    for rdt in ["1001", "1010"]:
        _assert_correct_coupling_columns(tmp_path, rdt)


# ----- Helpers ----- #

def _assert_correct_files_are_present(outputdir: Path) -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    for plane in PLANES:
        assert (outputdir / f"{AMP_BETA_NAME}{plane.lower()}.tfs").is_file()
        assert (outputdir / f"{BETA_NAME}{plane.lower()}.tfs").is_file()
        assert (outputdir / f"{PHASE_NAME}{plane.lower()}.tfs").is_file()
        assert (outputdir / f"{TOTAL_PHASE_NAME}{plane.lower()}.tfs").is_file()
        assert (outputdir / f"{ORBIT_NAME}{plane.lower()}.tfs").is_file()
        assert (outputdir / f"{DISPERSION_NAME}x.tfs").is_file()
    assert (outputdir / f"{NORM_DISP_NAME}x.tfs").is_file()  # no norm disp in Y plane

    for rdt in ["1001", "1010"]:
        assert (outputdir / f"coupling_f{rdt}.tfs").is_file()


def _assert_correct_beta_amp_columns(outputdir: Path, plane: str) -> None:
    """Checks the expected columns are present in the beta from amplitude file in outputdir"""
    dframe = tfs.read(outputdir / f"{AMP_BETA_NAME}{plane.lower()}.tfs")
    expected_created_columns = [f"{DELTA}BET{plane}", f"{ERR}{DELTA}BET{plane}"]  # new
    expected_renamed_columns = [f"BET{plane}STD", f"BET{plane}STDRES"]  # disappeared

    for col in expected_created_columns:
        assert col in dframe.columns
    for col in expected_renamed_columns:
        assert col not in dframe.columns


def _assert_correct_beta_phase_columns(outputdir: Path, plane: str) -> None:
    """Checks the expected columns are present in the beta from phase file in outputdir"""
    dframe = tfs.read(outputdir / f"{BETA_NAME}{plane.lower()}.tfs")
    expected_maybe_dropped_columns = [f"STATBET{plane}", f"SYSBET{plane}", "CORR_ALFABETA",
                                      f"STATAL", f"F{plane}", f"SYSALF{plane}"]  # disappeared
    expected_created_columns = [f"{DELTA}BET{plane}", f"{ERR}{DELTA}BET{plane}",
                                f"{DELTA}ALF{plane}", f"{ERR}{DELTA}ALF{plane}"]  # new
    expected_renamed_columns = []  # disappeared

    if "CORR_ALFABETA" in dframe.columns:
        for col in expected_maybe_dropped_columns:
            assert col not in dframe.columns
    for col in expected_created_columns:
        assert col in dframe.columns
    for col in expected_renamed_columns:
        assert col not in dframe.columns


def _assert_correct_phase_columns(outputdir: Path, plane: str) -> None:
    """Checks the expected columns are present in the phase file in outputdir"""
    dframe = tfs.read(outputdir / f"{PHASE_NAME}{plane.lower()}.tfs")
    expected_converted_columns = [f"{ERR}PHASE{plane}", f"PHASE{plane}{MDL}", "S2"]  # replace renamed
    expected_created_columns = [f"{DELTA}PHASE{plane}", f"{ERR}{DELTA}PHASE{plane}"]  # new
    expected_renamed_columns = [f"STDPH{plane}", f"PH{plane}{MDL}", "S1"]  # disappeared

    for col in expected_converted_columns:
        assert col in dframe.columns
    for col in expected_created_columns:
        assert col in dframe.columns
    for col in expected_renamed_columns:
        assert col not in dframe.columns


def _assert_correct_total_phase_columns(outputdir: Path, plane: str) -> None:
    """Checks the expected columns are present in the total phase file in outputdir"""
    dframe = tfs.read(outputdir / f"{TOTAL_PHASE_NAME}{plane.lower()}.tfs")
    expected_converted_columns = [f"{ERR}PHASE{plane}", f"PHASE{plane}{MDL}", "S2"]  # replace renamed
    expected_created_columns = [f"{DELTA}PHASE{plane}", f"{ERR}{DELTA}PHASE{plane}"]  # new
    expected_renamed_columns = [f"STDPH{plane}", f"PH{plane}{MDL}", "S1"]  # disappeared

    for col in expected_converted_columns:
        assert col in dframe.columns
    for col in expected_created_columns:
        assert col in dframe.columns
    for col in expected_renamed_columns:
        assert col not in dframe.columns


def _assert_correct_closed_orbit_columns(outputdir: Path, plane: str) -> None:
    """Checks the expected columns are present in the closed orbit file in outputdir"""
    dframe = tfs.read(outputdir / f"{ORBIT_NAME}{plane.lower()}.tfs")
    expected_converted_columns = [f"{ERR}{plane}"]    # replace renamed
    expected_created_columns = [f"{DELTA}{plane}", f"{ERR}{DELTA}{plane}"]  # new
    expected_renamed_columns = [f"STD{plane}"]  # disappeared

    for col in expected_converted_columns:
        assert col in dframe.columns
    for col in expected_created_columns:
        assert col in dframe.columns
    for col in expected_renamed_columns:
        assert col not in dframe.columns


def _assert_correct_dispersion_columns(outputdir: Path, plane: str) -> None:
    """Checks the expected columns are present in the dispersion file in outputdir"""
    dframe = tfs.read(outputdir / f"{DISPERSION_NAME}{plane.lower()}.tfs")
    expected_converted_columns = [f"{ERR}D{plane}"]    # replace renamed
    expected_created_columns = [f"{DELTA}D{plane}", f"{ERR}{DELTA}D{plane}"]  # new
    expected_renamed_columns = [f"STDD{plane}"]  # disappeared

    for col in expected_converted_columns:
        assert col in dframe.columns
    for col in expected_created_columns:
        assert col in dframe.columns
    for col in expected_renamed_columns:
        assert col not in dframe.columns


def _assert_correct_normalized_dispersion_columns(outputdir: Path, plane: str) -> None:
    """Checks the expected columns are present in the normalized dispersion file in outputdir"""
    dframe = tfs.read(outputdir / f"{NORM_DISP_NAME}{plane.lower()}.tfs")
    expected_converted_columns = [f"{ERR}ND{plane}"]    # replace renamed
    expected_created_columns = [f"{DELTA}ND{plane}", f"{ERR}{DELTA}ND{plane}"]  # new
    expected_renamed_columns = [f"STDND{plane}"]  # disappeared

    for col in expected_converted_columns:
        assert col in dframe.columns
    for col in expected_created_columns:
        assert col in dframe.columns
    for col in expected_renamed_columns:
        assert col not in dframe.columns


def _assert_correct_coupling_columns(outputdir: Path, rdt: str) -> None:
    """Checks the expected columns are present in the normalized dispersion file in outputdir"""
    dframe = tfs.read(outputdir / f"coupling_f{rdt}.tfs")
    expected_converted_columns = ["AMP", f"{ERR}AMP", "PHASE", f"{ERR}PHASE", "REAL", "IMAG", "MDLREAL", "MDLIMAG"]    # replace renamed
    expected_renamed_columns = [f"F{rdt}W", f"Q{rdt}", f"Q{rdt}STD", f"F{rdt}R", f"F{rdt}I"]  # disappeared

    for col in expected_converted_columns:
        assert col in dframe.columns
    for col in expected_renamed_columns:
        assert col not in dframe.columns
