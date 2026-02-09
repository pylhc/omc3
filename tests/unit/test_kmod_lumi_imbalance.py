from pathlib import Path
from typing import Any

import pytest
import tfs
from tfs.testing import assert_tfs_frame_equal

from omc3.optics_measurements.constants import (
    AVERAGED_BETASTAR_FILENAME,
    EFFECTIVE_BETAS_FILENAME,
    EXT,
)
from omc3.scripts.kmod_lumi_imbalance import calculate_lumi_imbalance
from tests.unit.test_kmod_averaging import REFERENCE_DIR, get_reference_dir

# Tests ---

@pytest.mark.basic
def test_kmod_lumi_imbalance(tmp_path):
    betas = [0.22, 0.22]
    path_beta_ip1 = _get_input_path(1, betas)
    path_beta_ip5 = _get_input_path(5, betas)
    calculate_lumi_imbalance(ip1=path_beta_ip1, ip5=path_beta_ip5, betastar=betas, output_dir=tmp_path)
    _assert_correct_files_are_present(tmp_path, betas, 1, 5)

    eff_betas = tfs.read(tmp_path / _get_effbetas_filename(betas, 1, 5))
    eff_betas_ref = tfs.read(REFERENCE_DIR / _get_effbetas_filename(betas, 1, 5))
    assert_tfs_frame_equal(eff_betas_ref, eff_betas, check_like=True)


# Helper ---

def _assert_correct_files_are_present(outputdir: Path, betas: list[float], ip_a: Any, ip_b: Any) -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    assert (outputdir / _get_effbetas_filename(betas, ip_a, ip_b)).is_file()


def _get_input_path(ip: int, betas: list[float]) -> Path:
    return get_reference_dir(ip=ip, n_files=2) / f"{AVERAGED_BETASTAR_FILENAME.format(betastar_x=betas[0], betastar_y=betas[1], ip=ip)}{EXT}"


def _get_effbetas_filename(betas: list[float], ip_a: Any, ip_b: Any) -> str:
    return f"{EFFECTIVE_BETAS_FILENAME.format(ip_a=ip_a, ip_b=ip_b, betastar_x=betas[0], betastar_y=betas[1])}{EXT}"


# ---------------- FOR UPDATING THE REFERENCES ------------------------------- #

def update_reference_files():
    """ Helper function to update the reference files. """
    REFERENCE_DIR.mkdir(exist_ok=True, parents=True)
    betas = [0.22, 0.22]
    path_beta_ip1 = _get_input_path(1, betas)
    path_beta_ip5 = _get_input_path(5, betas)
    calculate_lumi_imbalance(ip1=path_beta_ip1, ip5=path_beta_ip5, output_dir=REFERENCE_DIR, betastar=betas)
    for ini_file in REFERENCE_DIR.glob("*.ini"):
        ini_file.unlink()
