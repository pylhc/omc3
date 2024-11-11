from pathlib import Path

import pandas.testing as pdt
import pytest
import tfs

from omc3.scripts.kmod_average import (
    AVERAGED_BETASTAR_FILENAME,
    AVERAGED_BPM_FILENAME,
    EXT,
    average_kmod_results,
)
from omc3.plotting.plot_kmod_results import PARAM_BETA, PARAM_WAIST
from tests.conftest import INPUTS, ids_str

KMOD_INPUT_DIR = INPUTS / "kmod"
REFERENCE_DIR = KMOD_INPUT_DIR / "references"

@pytest.mark.basic
@pytest.mark.parametrize("ip", [1, 5], ids=ids_str("ip{}"))
@pytest.mark.parametrize("n_files", [1, 2], ids=ids_str("{}files"))
def test_kmod_averaging(tmp_path, ip, n_files):
    beta = 0.22
    meas_paths = [_get_measurement_dir(ip, i+1) for i in range(n_files)]
    ref_output_dir = _get_reference_dir(ip, n_files)

    average_kmod_results(
        meas_paths=meas_paths, 
        output_dir=tmp_path,
        ip=ip, 
        betastar=[beta],
        plot=True
     )
    _assert_correct_files_are_present(tmp_path, ip, beta)

    for out_name in _get_all_tfs_filenames(ip, beta):
        out_file = tfs.read(tmp_path / out_name)
        ref_file = tfs.read(ref_output_dir / out_name)
        pdt.assert_frame_equal(out_file, ref_file, check_like=True)


def _assert_correct_files_are_present(outputdir: Path, ip: int, beta: float) -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    all_files = _get_all_tfs_filenames(ip, beta) + [f"ip{ip}_{PARAM_WAIST}.pdf", f"ip{ip}_{PARAM_BETA}.pdf"]
    for file_name in all_files:
        assert (outputdir / file_name).is_file()


def _get_measurement_dir(ip: int, i_meas: int) -> Path:
    return KMOD_INPUT_DIR / f"ip{ip}_meas{i_meas}"


def _get_reference_dir(ip: int, n_files: int) -> Path:
    return REFERENCE_DIR / f"ip{ip}_averaged_{n_files}files"


def _get_all_tfs_filenames(ip: int, beta:float) -> list[str]:
    return [
        f"{AVERAGED_BPM_FILENAME.format(betastar_x=beta, betastar_y=beta, ip=ip, beam=1)}{EXT}",
        f"{AVERAGED_BPM_FILENAME.format(betastar_x=beta, betastar_y=beta, ip=ip, beam=2)}{EXT}",
        f"{AVERAGED_BETASTAR_FILENAME.format(betastar_x=beta, betastar_y=beta, ip=ip)}{EXT}",
    ]


# ---------------- FOR UPDATING THE REFERENCES ------------------------------- #

def update_reference_files():
    """ Helper function to update the reference files. """
    REFERENCE_DIR.mkdir(exist_ok=True, parents=True)
    beta = 0.22
    for ip in (1, 5):
        for n_files in (1, 2):
            meas_paths = [_get_measurement_dir(ip, i+1) for i in range(n_files)]
            output_dir = _get_reference_dir(ip, n_files)
            average_kmod_results(
                meas_paths=meas_paths, 
                output_dir=output_dir,
                ip=ip, 
                betastar=[beta],
                plot=False
            )
            for ini_file in output_dir.glob("*.ini"):
                ini_file.unlink()
