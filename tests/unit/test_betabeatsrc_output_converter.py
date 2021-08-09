from pathlib import Path

import numpy as np
import pytest
import tfs

from omc3.scripts import betabeatsrc_output_converter
from tests.conftest import cli_args

INPUT_DIR = Path(__file__).parent.parent / "inputs" / "merge_kmod"


# Full test --------------------------------------------------------------------
@pytest.mark.basic
def test_betabeatsrc_output_converter(tmp_path):
    # paths = [INPUT_DIR / "kmod_ip1", INPUT_DIR / "kmod_ip5"]
    #
    # res_tfs_passed = merge_kmod_results.merge_kmod_results(
    #     kmod_dirs=paths, outputdir=tmp_path
    # )
    # filename = f"{merge_kmod_results.LSA_RESULTS}{merge_kmod_results.EXT}"
    # res_lsa_tfs = tfs.read_tfs(tmp_path / filename, index=merge_kmod_results.NAME)
    # control_tfs = tfs.read_tfs(INPUT_DIR / "lsa_results_merged.tfs", index=merge_kmod_results.NAME)
    #
    # assert_frame_equal(res_lsa_tfs, control_tfs)
    # assert_dict_equal(res_lsa_tfs.headers, control_tfs.headers, compare_keys=True)
    # assert_frame_equal(res_tfs_passed, control_tfs, check_exact=False)
    # assert_dict_equal(res_tfs_passed.headers, control_tfs.headers, compare_keys=True)
    pass


@pytest.mark.basic
def test_betabeatsrc_output_converter_commandline(tmp_path):
    # paths = [str(INPUT_DIR / "kmod_ip1"), str(INPUT_DIR / "kmod_ip5")]
    #
    # with cli_args("--kmod_dirs", *paths, "--outputdir", str(tmp_path)):
    #     merge_kmod_results.merge_kmod_results()
    #
    # filename = f"{merge_kmod_results.LSA_RESULTS}{merge_kmod_results.EXT}"
    # res_lsa_tfs = tfs.read_tfs(tmp_path / filename, index=merge_kmod_results.NAME)
    # control_tfs = tfs.read_tfs(INPUT_DIR / "lsa_results_merged.tfs", index=merge_kmod_results.NAME)
    #
    # assert_frame_equal(res_lsa_tfs, control_tfs)
    # assert_dict_equal(res_lsa_tfs.headers, control_tfs.headers, compare_keys=True)
    pass


# Helper -----------------------------------------------------------------------


@pytest.fixture
def _tfs_file():
    return tfs.read_tfs(INPUT_DIR / "test_imbalance.tfs", index=merge_kmod_results.NAME)
