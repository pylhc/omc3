from pathlib import Path

import pytest
import tfs
import pandas.testing as pdt
from omc3.kmod_importer import import_kmod_data
from omc3.optics_measurements.constants import EXT, AVERAGED_BPM_FILENAME, BETA_KMOD_FILENAME
from tests.unit.test_kmod_averaging import _get_reference_dir
from tests.conftest import INPUTS

INPUT_DIR_KMOD = INPUTS / "kmod"
REFERENCE_DIR = INPUT_DIR_KMOD / "references"
IP1_RESULTS_OUTPUTS = REFERENCE_DIR / "ip1_averaged_2files"
IP5_RESULTS_OUTPUTS = REFERENCE_DIR / "ip5_averaged_2files"

B1_RESULTS_OUTPUTS = INPUT_DIR_KMOD / "b1_imported"
B2_RESULTS_OUTPUTS = INPUT_DIR_KMOD / "b2_imported"

@pytest.mark.basic
@pytest.mark.parametrize('beam', [1, 2])
def test_kmod_import_beam(tmp_path, beam):
    model = _get_model_path(beam)
    beta = 0.22

    path_bpm_ip1 = _get_input_path(beam, 1, beta)
    path_bpm_ip5 = _get_input_path(beam, 5, beta)
    
    import_kmod_data(meas_paths=[path_bpm_ip1, path_bpm_ip5], model=model, output_dir=str(tmp_path))
    _assert_correct_files_are_present(tmp_path)

    for plane in "xy":
        beta_out = tfs.read(tmp_path / f"{BETA_KMOD_FILENAME}{plane}{EXT}")
        beta_ref = tfs.read(_get_referece_path(beam, plane))

        # column order might have changed, but that's okay -> check_like=True
        pdt.assert_frame_equal(beta_ref, beta_out, check_like=True)


def _assert_correct_files_are_present(outputdir: Path) -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    for plane in "xy":
        assert (outputdir / f"{BETA_KMOD_FILENAME}{plane}{EXT}").is_file()


def _get_model_path(beam: int) -> Path:
    return INPUT_DIR_KMOD / f"b{beam}_twiss_22cm.dat"


def _get_input_path(beam: int, ip: int, beta: float) -> Path:
    return _get_reference_dir(ip=ip, n_files=2) / f"{AVERAGED_BPM_FILENAME.format(betastar_x=beta, betastar_y=beta, ip=ip, beam=beam)}{EXT}"


def _get_referece_path(beam: int, plane: str) -> Path:
    return REFERENCE_DIR / f"b{beam}_imported" / f"{BETA_KMOD_FILENAME}{plane}{EXT}"


# ---------------- FOR UPDATING THE REFERENCES ------------------------------- #

def update_reference_files():
    """ Helper function to update the reference files. """
    REFERENCE_DIR.mkdir(exist_ok=True, parents=True)
    beta = 0.22
    for beam in (1, 2):
        output_path = REFERENCE_DIR / f"b{beam}_imported"
        output_path.mkdir(exist_ok=True, parents=True)
        model = _get_model_path(beam)
        path_bpm_ip1 = _get_input_path(beam, 1, beta)
        path_bpm_ip5 = _get_input_path(beam, 5, beta)

        import_kmod_data(meas_paths=[path_bpm_ip1, path_bpm_ip5], model=model, output_dir=output_path)
        for ini_file in output_path.glob("*.ini"):
            ini_file.unlink()