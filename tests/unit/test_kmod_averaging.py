import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest
import tfs

from omc3.optics_measurements.constants import BEAM, BEAM_DIR, BETA, NAME
from omc3.plotting.plot_kmod_results import PARAM_BETA, PARAM_BETABEAT, PARAM_WAIST
from omc3.scripts.kmod_average import (
    AVERAGED_BETASTAR_FILENAME,
    AVERAGED_BPM_FILENAME,
    EXT,
    average_kmod_results,
)
from tests.conftest import INPUTS, assert_tfsdataframe_equal, ids_str

KMOD_INPUT_DIR = INPUTS / "kmod"
REFERENCE_DIR = KMOD_INPUT_DIR / "references"


# Tests ----

@pytest.mark.basic
@pytest.mark.parametrize("ip", [1, 5], ids=ids_str("ip{}"))
@pytest.mark.parametrize("n_files", [1, 2], ids=ids_str("{}files"))
def test_kmod_averaging(tmp_path, ip, n_files):
    betas = get_betastar_values(beam=1, ip=ip)
    meas_paths = [get_measurement_dir(ip, i+1) for i in range(n_files)]
    ref_output_dir = get_reference_dir(ip, n_files)

    average_kmod_results(
        meas_paths=meas_paths, 
        output_dir=tmp_path,
        ip=ip, 
        betastar=betas,
        plot=True
     )
    _assert_correct_files_are_present(tmp_path, ip, betas)

    for out_name in get_all_tfs_filenames(ip, betas):
        out_file = tfs.read(tmp_path / out_name)
        ref_file = tfs.read(ref_output_dir / out_name)
        assert_tfsdataframe_equal(out_file, ref_file, check_like=True)


@pytest.mark.extended
@pytest.mark.parametrize("beam", [1, 2], ids=ids_str("beam{}"))
def test_kmod_averaging_single_beam(tmp_path, beam, caplog):
    ip = 1
    n_files = 2

    betas = get_betastar_values(beam=beam, ip=ip)
    ref_output_dir = get_reference_dir(ip, n_files)

    meas_paths = [get_measurement_dir(ip, i+1)  for i in range(n_files)]
    new_meas_paths = [tmp_path / f"single_beam_meas_{i+1}" for i in range(n_files)]
    
    for tmp_meas, old_meas in zip(new_meas_paths, meas_paths):
        beam_dir = f"{BEAM_DIR}{beam}"
        (tmp_meas / beam_dir).mkdir(parents=True, exist_ok=True)
        for tfs_file in (old_meas / beam_dir).glob("*"):
            shutil.copy(tfs_file, tmp_meas / beam_dir)

    with caplog.at_level(logging.WARNING):
        average_kmod_results(
            meas_paths=new_meas_paths, 
            output_dir=tmp_path,
            ip=ip, 
            betastar=betas,
            plot=True
        )
    assert f"Could not find all results for beam {1 if beam == 2 else 2}" in caplog.text
    _assert_correct_files_are_present(tmp_path, ip, betas, beams=[beam])

    for out_name in get_all_tfs_filenames(ip, betas, beams=[beam]):
        out_file = tfs.read(tmp_path / out_name)
        ref_file = tfs.read(ref_output_dir / out_name)
        if BEAM in ref_file.columns:
            ref_file = ref_file.loc[ref_file[BEAM] == beam, :].reset_index(drop=True)
        assert_tfsdataframe_equal(out_file, ref_file, check_like=True)


# Helper ---

def _assert_correct_files_are_present(outputdir: Path, ip: int, betas: list[float], beams: Sequence[int] = (1, 2)) -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    all_files = (
        get_all_tfs_filenames(ip, betas, beams) + 
        [f"ip{ip}_{PARAM_WAIST}.pdf", f"ip{ip}_{PARAM_BETA}.pdf", f"ip{ip}_{PARAM_BETABEAT}.pdf"]
    )
    for file_name in all_files:
        assert (outputdir / file_name).is_file()


def get_measurement_dir(ip: int, i_meas: int) -> Path:
    return KMOD_INPUT_DIR / f"ip{ip}_meas{i_meas}"


def get_reference_dir(ip: int, n_files: int) -> Path:
    return REFERENCE_DIR / f"ip{ip}_averaged_{n_files}files"


def get_model_path(beam: int) -> Path:
    return KMOD_INPUT_DIR / f"b{beam}_twiss_22cm.dat"


def get_betastar_values(beam: int, ip: int) -> list[float]:
    """Extract the betastar values from the model."""
    model = tfs.read(get_model_path(beam), index=NAME)
    return model.loc[f"IP{ip}", [f"{BETA}X", f"{BETA}Y"]].tolist()


def get_all_tfs_filenames(ip: int, betas: list[float], beams: Sequence[int] = (1, 2)) -> list[str]:
    return [
        f"{AVERAGED_BPM_FILENAME.format(betastar_x=betas[0], betastar_y=betas[1], ip=ip, beam=beam)}{EXT}"
        for beam in beams
    ] + [
        f"{AVERAGED_BETASTAR_FILENAME.format(betastar_x=betas[0], betastar_y=betas[1], ip=ip)}{EXT}",
    ]


# ---------------- FOR UPDATING THE REFERENCES ------------------------------- #

def update_reference_files():
    """ Helper function to update the reference files. """
    REFERENCE_DIR.mkdir(exist_ok=True, parents=True)
    for ip in (1, 5, 2, 8):
        betas = get_betastar_values(beam=1, ip=ip)
        for n_files in (1, 2):
            if ip  % 2 == 0 and n_files == 2:
                continue # Skip the 2nd file for IP 2 and 8
            meas_paths = [get_measurement_dir(ip, i+1) for i in range(n_files)]
            output_dir = get_reference_dir(ip, n_files)
            average_kmod_results(
                meas_paths=meas_paths, 
                output_dir=output_dir,
                ip=ip, 
                betastar=betas,
                plot=False
            )
            for ini_file in output_dir.glob("*.ini"):
                ini_file.unlink()
