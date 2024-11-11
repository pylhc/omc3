from pathlib import Path

import pandas.testing as pdt
import pytest
import tfs

from omc3.scripts.kmod_import import import_kmod_data
from omc3.optics_measurements.constants import (
    AVERAGED_BETASTAR_FILENAME,
    AVERAGED_BPM_FILENAME,
    BETA_KMOD_FILENAME,
    BETA_STAR_FILENAME,
    EXT,
)
from tests.conftest import INPUTS
from tests.unit.test_kmod_averaging import _get_reference_dir as get_averages_dir

INPUT_DIR_KMOD = INPUTS / "kmod"
REFERENCE_DIR = INPUT_DIR_KMOD / "references"
IP1_RESULTS_OUTPUTS = REFERENCE_DIR / "ip1_averaged_2files"
IP5_RESULTS_OUTPUTS = REFERENCE_DIR / "ip5_averaged_2files"

B1_RESULTS_OUTPUTS = INPUT_DIR_KMOD / "b1_imported"
B2_RESULTS_OUTPUTS = INPUT_DIR_KMOD / "b2_imported"

@pytest.mark.basic
@pytest.mark.parametrize('beam', [1, 2])
def test_kmod_import_averaged_folder_beam(tmp_path, beam):
    model = _get_model_path(beam)

    path_bpm_ip1 = get_averages_dir(ip=1, n_files=2)
    path_bpm_ip5 = get_averages_dir(ip=5, n_files=2) 
    
    import_kmod_data(
        measurements=[path_bpm_ip1, path_bpm_ip5],
        model=model, 
        output_dir=tmp_path,
        beam=beam,
    )
    _assert_correct_files_are_present(tmp_path)

    for plane in "xy":
        for ref_path in (_get_bpm_referece_path(beam, plane), _get_betastar_referece_path(beam, plane)):
            beta_ref = tfs.read(ref_path)
            beta_out = tfs.read(tmp_path / ref_path.name)

            # column order might have changed, but that's okay -> check_like=True
            pdt.assert_frame_equal(beta_ref, beta_out, check_like=True)


@pytest.mark.extended
@pytest.mark.parametrize('beam', [1, 2])
@pytest.mark.parametrize('files', ["bpm", "betastar", "bpm-betastar"])
@pytest.mark.parametrize('read', [True, False])
def test_kmod_import_files_beam(tmp_path, beam, files, read):
    model = _get_model_path(beam)
    beta = 0.22

    paths = []
    if "bpm" in files:
        paths += [_get_bpm_input_path(beam, ip, beta) for ip in (1, 5)]
    
    if "betastar" in files:
        paths += [_get_betastar_input_path(beam, ip, beta) for ip in (1, 5)]
    
    if read:
        paths = [tfs.read(path) for path in paths]
    
    import_kmod_data(
        measurements=paths, 
        model=model, 
        beam=beam,
        output_dir=str(tmp_path),
    )
    _assert_correct_files_are_present(tmp_path, which=files)

    for plane in "xy":
        ref_paths = []
        if "bpm" in files:
            ref_paths += [_get_bpm_referece_path(beam, plane)]
        
        if "betastar" in files:
            ref_paths += [_get_betastar_referece_path(beam, plane)]

        for ref_path in ref_paths:
            beta_ref = tfs.read(ref_path)
            beta_out = tfs.read(tmp_path / ref_path.name)

            # column order might have changed, but that's okay -> check_like=True
            pdt.assert_frame_equal(beta_ref, beta_out, check_like=True)


def _assert_correct_files_are_present(outputdir: Path, which: str = "bpm-betastar") -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    for plane in "xy":
        if "bpm" in which:
            assert (outputdir / f"{BETA_KMOD_FILENAME}{plane}{EXT}").is_file()
        if "betastar" in which:
            assert (outputdir / f"{BETA_STAR_FILENAME}{plane}{EXT}").is_file()


def _get_model_path(beam: int) -> Path:
    return INPUT_DIR_KMOD / f"b{beam}_twiss_22cm.dat"


def _get_bpm_input_path(beam: int, ip: int, beta: float) -> Path:
    return get_averages_dir(ip=ip, n_files=2) / f"{AVERAGED_BPM_FILENAME.format(betastar_x=beta, betastar_y=beta, ip=ip, beam=beam)}{EXT}"


def _get_betastar_input_path(beam: int, ip: int, beta: float) -> Path:
    return get_averages_dir(ip=ip, n_files=2) / f"{AVERAGED_BETASTAR_FILENAME.format(betastar_x=beta, betastar_y=beta, ip=ip, beam=beam)}{EXT}"


def _get_bpm_referece_path(beam: int, plane: str) -> Path:
    return REFERENCE_DIR / f"b{beam}_imported" / f"{BETA_KMOD_FILENAME}{plane}{EXT}"


def _get_betastar_referece_path(beam: int, plane: str) -> Path:
    return REFERENCE_DIR / f"b{beam}_imported" / f"{BETA_STAR_FILENAME}{plane}{EXT}"


# ---------------- FOR UPDATING THE REFERENCES ------------------------------- #

def update_reference_files():
    """ Helper function to update the reference files. """
    REFERENCE_DIR.mkdir(exist_ok=True, parents=True)
    beta = 0.22
    for beam in (1, 2):
        output_path = REFERENCE_DIR / f"b{beam}_imported"
        output_path.mkdir(exist_ok=True, parents=True)
        model = _get_model_path(beam)
        path_ip1 = get_averages_dir(ip=1, n_files=2)
        path_ip5 = get_averages_dir(ip=5, n_files=2)

        import_kmod_data(meas_paths=[path_ip1, path_ip5], model=model, output_dir=output_path, beam=beam)
        for ini_file in output_path.glob("*.ini"):
            ini_file.unlink()


if __name__ == "__main__":
    update_reference_files()