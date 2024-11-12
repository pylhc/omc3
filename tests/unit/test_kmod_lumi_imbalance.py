from pathlib import Path

import pytest
import tfs

from omc3.optics_measurements.constants import (
    AVERAGED_BETASTAR_FILENAME,
    EFFECTIVE_BETAS_FILENAME,
    EXT,
)
from omc3.scripts.kmod_lumi_imbalance import calculate_lumi_imbalance
from tests.conftest import assert_tfsdataframe_equal
from tests.unit.test_kmod_averaging import REFERENCE_DIR, get_reference_dir

# Tests ---

@pytest.mark.basic
def test_kmod_lumi_imbalance(tmp_path):
    beta = 0.22
    path_beta_ip1 = _get_input_path(1, beta)
    path_beta_ip5 = _get_input_path(5, beta)
    calculate_lumi_imbalance(ip1=path_beta_ip1, ip5=path_beta_ip5, betastar=[beta], output_dir=tmp_path)
    _assert_correct_files_are_present(tmp_path, beta)

    eff_betas = tfs.read(tmp_path / _get_effbetas_filename(beta))
    eff_betas_ref = tfs.read(REFERENCE_DIR / _get_effbetas_filename(beta))
    assert_tfsdataframe_equal(eff_betas_ref, eff_betas, check_like=True)


# Helper ---

def _assert_correct_files_are_present(outputdir: Path, beta: float) -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    assert (outputdir / _get_effbetas_filename(beta)).is_file()


def _get_input_path(ip: int, beta: float) -> Path:
    return get_reference_dir(ip=ip, n_files=2) / f"{AVERAGED_BETASTAR_FILENAME.format(betastar_x=beta, betastar_y=beta, ip=ip)}{EXT}"


def _get_effbetas_filename(beta: float) -> str:
    return f"{EFFECTIVE_BETAS_FILENAME.format(betastar_x=beta, betastar_y=beta)}{EXT}"


# ---------------- FOR UPDATING THE REFERENCES ------------------------------- #

def update_reference_files():
    """ Helper function to update the reference files. """
    REFERENCE_DIR.mkdir(exist_ok=True, parents=True)
    beta = 0.22
    path_beta_ip1 = _get_input_path(1, beta)
    path_beta_ip5 = _get_input_path(5, beta)
    calculate_lumi_imbalance(ip1=path_beta_ip1, ip5=path_beta_ip5, output_dir=REFERENCE_DIR, betastar=[beta, beta])
    for ini_file in REFERENCE_DIR.glob("*.ini"):
        ini_file.unlink()