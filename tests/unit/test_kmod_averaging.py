from pathlib import Path

import pytest
import tfs
import pandas.testing as pdt
from omc3.kmod_averages import average_kmod_results_entrypoint

INPUT_DIR = Path(__file__).parent.parent / "inputs/kmod_averaging"
IP1_1_OUTPUTS = INPUT_DIR / "single_ip1_meas1"
IP1_2_OUTPUTS = INPUT_DIR / "single_ip1_meas2"
IP5_1_OUTPUTS = INPUT_DIR / "single_ip5_meas1"
IP5_2_OUTPUTS = INPUT_DIR / "single_ip5_meas2"
IP1_RESULTS_OUTPUTS = INPUT_DIR / "ip1_averaged"
IP5_RESULTS_OUTPUTS = INPUT_DIR / "ip5_averaged"
IP1_SINGLE_RESULTS_OUTPUTS = INPUT_DIR / "ip1_averaged_single"
IP5_SINGLE_RESULTS_OUTPUTS = INPUT_DIR / "ip5_averaged_single"


# ----- Tests ----- #
def test_kmod_averaging_1_file_ip1(tmp_path):
    ip = 1
    beta = 0.22
    average_kmod_results_entrypoint(meas_paths=[IP1_1_OUTPUTS], output_dir=str(tmp_path), ip=ip, beta=beta)
    _assert_correct_files_are_present(tmp_path, ip, beta)

    bpm_b1 = tfs.read(tmp_path / f"averaged_bpm_beam1_ip{ip}_beta{beta}m.tfs")
    bpm_b2 = tfs.read(tmp_path / f"averaged_bpm_beam2_ip{ip}_beta{beta}m.tfs")
    ip_res = tfs.read(tmp_path / f"averaged_ip{ip}_beta{beta}m.tfs")

    bpm_b1_ref = tfs.read(IP1_SINGLE_RESULTS_OUTPUTS / f"averaged_bpm_beam1_ip{ip}_beta{beta}m.tfs")
    bpm_b2_ref = tfs.read(IP1_SINGLE_RESULTS_OUTPUTS / f"averaged_bpm_beam2_ip{ip}_beta{beta}m.tfs")
    ip_res_ref = tfs.read(IP1_SINGLE_RESULTS_OUTPUTS / f"averaged_ip{ip}_beta{beta}m.tfs")

    pdt.assert_frame_equal(bpm_b1_ref, bpm_b1)
    pdt.assert_frame_equal(bpm_b2_ref, bpm_b2)
    pdt.assert_frame_equal(ip_res_ref, ip_res)


def test_kmod_averaging_1_file_ip5(tmp_path):
    ip = 5
    beta = 0.22
    average_kmod_results_entrypoint(meas_paths=[IP5_1_OUTPUTS], output_dir=str(tmp_path), ip=ip, beta=beta)
    _assert_correct_files_are_present(tmp_path, ip, beta)

    bpm_b1 = tfs.read(tmp_path / f"averaged_bpm_beam1_ip{ip}_beta{beta}m.tfs")
    bpm_b2 = tfs.read(tmp_path / f"averaged_bpm_beam2_ip{ip}_beta{beta}m.tfs")
    ip_res = tfs.read(tmp_path / f"averaged_ip{ip}_beta{beta}m.tfs")

    bpm_b1_ref = tfs.read(IP5_SINGLE_RESULTS_OUTPUTS / f"averaged_bpm_beam1_ip{ip}_beta{beta}m.tfs")
    bpm_b2_ref = tfs.read(IP5_SINGLE_RESULTS_OUTPUTS / f"averaged_bpm_beam2_ip{ip}_beta{beta}m.tfs")
    ip_res_ref = tfs.read(IP5_SINGLE_RESULTS_OUTPUTS / f"averaged_ip{ip}_beta{beta}m.tfs")

    pdt.assert_frame_equal(bpm_b1_ref, bpm_b1)
    pdt.assert_frame_equal(bpm_b2_ref, bpm_b2)
    pdt.assert_frame_equal(ip_res_ref, ip_res)


def test_kmod_averaging_2_files_ip1(tmp_path):
    ip = 1
    beta = 0.22
    average_kmod_results_entrypoint(meas_paths=[IP1_1_OUTPUTS, IP1_2_OUTPUTS], output_dir=str(tmp_path), ip=ip, beta=beta)
    _assert_correct_files_are_present(tmp_path, ip, beta)

    bpm_b1 = tfs.read(tmp_path / f"averaged_bpm_beam1_ip{ip}_beta{beta}m.tfs")
    bpm_b2 = tfs.read(tmp_path / f"averaged_bpm_beam2_ip{ip}_beta{beta}m.tfs")
    ip_res = tfs.read(tmp_path / f"averaged_ip{ip}_beta{beta}m.tfs")

    bpm_b1_ref = tfs.read(IP1_RESULTS_OUTPUTS / f"averaged_bpm_beam1_ip{ip}_beta{beta}m.tfs")
    bpm_b2_ref = tfs.read(IP1_RESULTS_OUTPUTS / f"averaged_bpm_beam2_ip{ip}_beta{beta}m.tfs")
    ip_res_ref = tfs.read(IP1_RESULTS_OUTPUTS / f"averaged_ip{ip}_beta{beta}m.tfs")

    pdt.assert_frame_equal(bpm_b1_ref, bpm_b1)
    pdt.assert_frame_equal(bpm_b2_ref, bpm_b2)
    pdt.assert_frame_equal(ip_res_ref, ip_res)


def test_kmod_averaging_2_files_ip5(tmp_path):
    ip = 5
    beta = 0.22
    average_kmod_results_entrypoint(meas_paths=[IP5_1_OUTPUTS, IP5_2_OUTPUTS], output_dir=str(tmp_path), ip=ip, beta=beta)
    _assert_correct_files_are_present(tmp_path, ip, beta)

    bpm_b1 = tfs.read(tmp_path / f"averaged_bpm_beam1_ip{ip}_beta{beta}m.tfs")
    bpm_b2 = tfs.read(tmp_path / f"averaged_bpm_beam2_ip{ip}_beta{beta}m.tfs")
    ip_res = tfs.read(tmp_path / f"averaged_ip{ip}_beta{beta}m.tfs")

    bpm_b1_ref = tfs.read(IP5_RESULTS_OUTPUTS / f"averaged_bpm_beam1_ip{ip}_beta{beta}m.tfs")
    bpm_b2_ref = tfs.read(IP5_RESULTS_OUTPUTS / f"averaged_bpm_beam2_ip{ip}_beta{beta}m.tfs")
    ip_res_ref = tfs.read(IP5_RESULTS_OUTPUTS / f"averaged_ip{ip}_beta{beta}m.tfs")

    pdt.assert_frame_equal(bpm_b1_ref, bpm_b1)
    pdt.assert_frame_equal(bpm_b2_ref, bpm_b2)
    pdt.assert_frame_equal(ip_res_ref, ip_res)



def _assert_correct_files_are_present(outputdir: Path, ip: int, beta: float) -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    assert (outputdir / f"averaged_bpm_beam1_ip{ip}_beta{beta}m.tfs").is_file()
    assert (outputdir / f"averaged_bpm_beam2_ip{ip}_beta{beta}m.tfs").is_file()
    assert (outputdir / f"averaged_ip{ip}_beta{beta}m.tfs").is_file()
    assert (outputdir / f"ip{ip}_betas.png").is_file()
    assert (outputdir / f"ip{ip}_waist.png").is_file()

