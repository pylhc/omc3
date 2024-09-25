from pathlib import Path

import pytest
import tfs
import pandas.testing as pdt
from omc3.import_kmod import input_kmod_for_global_entrypoint

INPUT_DIR = Path(__file__).parent.parent / "inputs/kmod_averaging"
IP1_RESULTS_OUTPUTS = INPUT_DIR / "ip1_averaged"
IP5_RESULTS_OUTPUTS = INPUT_DIR / "ip5_averaged"

B1_RESULTS_OUTPUTS = INPUT_DIR / "beam1_global_input"
B2_RESULTS_OUTPUTS = INPUT_DIR / "beam2_global_input"

MODEL_B1 = INPUT_DIR / "b1_twiss.dat"
MODEL_B2 = INPUT_DIR / "b2_twiss.dat"


# ----- Tests ----- #
def test_kmod_import_beam(tmp_path):
    models = {1: MODEL_B1, 2: MODEL_B2}
    results_outputs = {1: B1_RESULTS_OUTPUTS, 2: B2_RESULTS_OUTPUTS}
    beta = 0.22

    for beam in [1, 2]:    
        path_bpm_ip1 = IP1_RESULTS_OUTPUTS / f"averaged_bpm_beam{beam}_ip1_beta{beta}m.tfs"
        path_bpm_ip5 = IP5_RESULTS_OUTPUTS / f"averaged_bpm_beam{beam}_ip5_beta{beta}m.tfs"
        
        input_kmod_for_global_entrypoint(meas_paths=[path_bpm_ip1, path_bpm_ip5], model=models[beam], output_dir=str(tmp_path))
        _assert_correct_files_are_present(tmp_path)

        beta_x = tfs.read(tmp_path / f"beta_kmod_x.tfs")
        beta_y = tfs.read(tmp_path / f"beta_kmod_y.tfs")
        
        beta_x_ref = tfs.read(results_outputs[beam] / f"beta_kmod_x.tfs")
        beta_y_ref = tfs.read(results_outputs[beam] / f"beta_kmod_y.tfs")

        pdt.assert_frame_equal(beta_x_ref, beta_x)
        pdt.assert_frame_equal(beta_y_ref, beta_y)


def _assert_correct_files_are_present(outputdir: Path) -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    assert (outputdir / f"beta_kmod_x.tfs").is_file()
    assert (outputdir / f"beta_kmod_y.tfs").is_file()

