from pathlib import Path

import pytest
import tfs
import pandas.testing as pdt
from omc3.kmod_averages import average_kmod_results_entrypoint
from tests.conftest import INPUTS, ids_str

KMOD_INPUT_DIR = INPUTS / "kmod_averaging"

@pytest.mark.parametrize("ip", [1, 5], ids=ids_str("ip{}"))
@pytest.mark.parametrize("n_files", [1, 2], ids=ids_str("{}files"))
def test_kmod_averaging(tmp_path, ip, n_files):
    beta = 0.22
    meas_paths = [KMOD_INPUT_DIR / f"single_ip{ip}_meas{i+1}" for i in range(n_files)]
    ref_output_dir = KMOD_INPUT_DIR / f"ip{ip}_averaged{'_single' if n_files == 1 else ''}"

    average_kmod_results_entrypoint(meas_paths=meas_paths, output_dir=tmp_path, ip=ip, betastar=beta)
    _assert_correct_files_are_present(tmp_path, ip, beta)

    output_files = [
        f"averaged_bpm_beam1_ip{ip}_beta{beta}m.tfs",
        f"averaged_bpm_beam2_ip{ip}_beta{beta}m.tfs",
        f"averaged_ip{ip}_beta{beta}m.tfs",
    ]
    for out_name in output_files:
        out_file = tfs.read(tmp_path / out_name)
        ref_file = tfs.read(ref_output_dir / out_name)
        pdt.assert_frame_equal(out_file, ref_file)


def _assert_correct_files_are_present(outputdir: Path, ip: int, beta: float) -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    assert (outputdir / f"averaged_bpm_beam1_ip{ip}_beta{beta}m.tfs").is_file()
    assert (outputdir / f"averaged_bpm_beam2_ip{ip}_beta{beta}m.tfs").is_file()
    assert (outputdir / f"averaged_ip{ip}_beta{beta}m.tfs").is_file()
    assert (outputdir / f"ip{ip}_betas.png").is_file()
    assert (outputdir / f"ip{ip}_waist.png").is_file()

