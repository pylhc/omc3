from pathlib import Path

import pytest
import tfs
import pandas.testing as pdt
from omc3.kmod_import import import_kmod_data, BETA_FILENAME, EXT
from tests.conftest import INPUTS

INPUT_DIR_KMOD = INPUTS / "kmod_averaging"
IP1_RESULTS_OUTPUTS = INPUT_DIR_KMOD / "ip1_averaged"
IP5_RESULTS_OUTPUTS = INPUT_DIR_KMOD / "ip5_averaged"

B1_RESULTS_OUTPUTS = INPUT_DIR_KMOD / "beam1_global_input"
B2_RESULTS_OUTPUTS = INPUT_DIR_KMOD / "beam2_global_input"

MODEL_B1 = INPUT_DIR_KMOD / "b1_twiss.dat"
MODEL_B2 = INPUT_DIR_KMOD / "b2_twiss.dat"


@pytest.mark.basic
@pytest.mark.parametrize('beam', [1, 2])
def test_kmod_import_beam(tmp_path, beam):
    models = {1: MODEL_B1, 2: MODEL_B2}
    results_outputs = {1: B1_RESULTS_OUTPUTS, 2: B2_RESULTS_OUTPUTS}
    beta = 0.22

    path_bpm_ip1 = IP1_RESULTS_OUTPUTS / f"averaged_bpm_beam{beam}_ip1_beta{beta}m.tfs"
    path_bpm_ip5 = IP5_RESULTS_OUTPUTS / f"averaged_bpm_beam{beam}_ip5_beta{beta}m.tfs"
    
    import_kmod_data(meas_paths=[path_bpm_ip1, path_bpm_ip5], model=models[beam], output_dir=str(tmp_path))
    _assert_correct_files_are_present(tmp_path)

    for plane in "xy":
        beta_out = tfs.read(tmp_path / f"{BETA_FILENAME}{plane}{EXT}")
        beta_ref = tfs.read(results_outputs[beam] / f"beta_kmod_{plane}.tfs")

        # column order might have changed, but that's okay -> check_like=True
        pdt.assert_frame_equal(beta_ref, beta_out, check_like=True)


def _assert_correct_files_are_present(outputdir: Path) -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    for plane in "xy":
        assert (outputdir / f"{BETA_FILENAME}{plane}{EXT}").is_file()
