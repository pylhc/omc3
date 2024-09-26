from pathlib import Path

import tfs
import pandas.testing as pdt
from omc3.kmod_lumi_imbalance import calculate_lumi_imbalance_entrypoint


INPUT_DIR = Path(__file__).parent.parent / "inputs/kmod_averaging"
IP1_RESULTS_OUTPUTS = INPUT_DIR / "ip1_averaged"
IP5_RESULTS_OUTPUTS = INPUT_DIR / "ip5_averaged"

# ----- Tests ----- #
def test_kmod_lumi_imbalance(tmp_path):
    beta = 0.22 
    path_beta_ip1 = IP1_RESULTS_OUTPUTS / f"averaged_ip1_beta{beta}m.tfs"
    path_beta_ip5 = IP5_RESULTS_OUTPUTS / f"averaged_ip5_beta{beta}m.tfs"
    calculate_lumi_imbalance_entrypoint(ip1=path_beta_ip1, ip5=path_beta_ip5, output_dir=tmp_path)
    _assert_correct_files_are_present(tmp_path, beta)

    eff_betas = tfs.read(tmp_path / f"effective_betas_{beta}m.tfs")
    eff_betas_ref = tfs.read(INPUT_DIR / f"effective_betas_{beta}m.tfs")
    pdt.assert_frame_equal(eff_betas_ref, eff_betas)


def _assert_correct_files_are_present(outputdir: Path, beta: float) -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    assert (outputdir / f"effective_betas_beta{beta}m.tfs").is_file()

