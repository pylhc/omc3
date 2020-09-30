from pathlib import Path

import numpy as np
import pytest
import tfs
from pandas._testing import assert_dict_equal
from pandas.testing import assert_frame_equal

from omc3.scripts import merge_kmod_results
from tests.conftest import cli_args

INPUT_DIR = Path(__file__).parent.parent / "inputs" / "merge_kmod"
DEBUG = False


# Full test --------------------------------------------------------------------
@pytest.mark.basic
def test_merge_kmod_results(tmp_output_dir):
    paths = [INPUT_DIR / "kmod_ip1", INPUT_DIR / "kmod_ip5"]

    res_tfs_passed = merge_kmod_results.merge_kmod_results(kmod_dirs=paths, outputdir=tmp_output_dir)
    filename = f"{merge_kmod_results.LSA_RESULTS}{merge_kmod_results.EXT}"
    res_lsa_tfs = tfs.read_tfs(tmp_output_dir / filename, index=merge_kmod_results.NAME)
    control_tfs = tfs.read_tfs(INPUT_DIR / "lsa_results_merged.tfs", index=merge_kmod_results.NAME)

    assert_frame_equal(res_lsa_tfs, control_tfs)
    assert_dict_equal(res_lsa_tfs.headers, control_tfs.headers, compare_keys=True)
    assert_frame_equal(res_tfs_passed, control_tfs, check_exact=False)
    assert_dict_equal(res_tfs_passed.headers, control_tfs.headers, compare_keys=True)


@pytest.mark.extended
def test_merge_kmod_results_commandline(tmp_output_dir):
    paths = [str(INPUT_DIR / "kmod_ip1"), str(INPUT_DIR / "kmod_ip5")]

    with cli_args("--kmod_dirs", *paths, "--outputdir", str(tmp_output_dir)):
        merge_kmod_results.merge_kmod_results()

    filename = f"{merge_kmod_results.LSA_RESULTS}{merge_kmod_results.EXT}"
    res_lsa_tfs = tfs.read_tfs(tmp_output_dir / filename, index=merge_kmod_results.NAME)
    control_tfs = tfs.read_tfs(INPUT_DIR / "lsa_results_merged.tfs", index=merge_kmod_results.NAME)

    assert_frame_equal(res_lsa_tfs, control_tfs)
    assert_dict_equal(res_lsa_tfs.headers, control_tfs.headers, compare_keys=True)


# Units ------------------------------------------------------------------------

@pytest.mark.basic
def test_calc_lumi_imbalance(_tfs_file):
    imbalance, beta_ip1, beta_ip5 = merge_kmod_results.get_lumi_imbalance(_tfs_file)
    rtol = 1e-15
    assert np.isclose(imbalance.nominal_value, 0.974139299943968, rtol=rtol)
    assert np.isclose(imbalance.std_dev, 0.0027110906825478205, rtol=rtol)
    assert np.isclose(beta_ip1.nominal_value, 0.3980039936985557, rtol=rtol)
    assert np.isclose(beta_ip1.std_dev, 0.0006143103957639417, rtol=rtol)
    assert np.isclose(beta_ip5.nominal_value, 0.40856989726361387, rtol=rtol)
    assert np.isclose(beta_ip5.std_dev, 0.0009461823749785368, rtol=rtol)


@pytest.mark.basic
def test_wrong_columns(_tfs_file):
    tfs_wrong_columns = _tfs_file.drop(columns=[f"{merge_kmod_results.BETA}Y"])
    res = merge_kmod_results._validate_for_imbalance(tfs_wrong_columns)
    assert not res


@pytest.mark.basic
def test_wrong_names(_tfs_file):
    tfs_wrong_names = _tfs_file.drop(index=[_tfs_file.index[0]])
    res = merge_kmod_results._validate_for_imbalance(tfs_wrong_names)
    assert not res


@pytest.mark.basic
def test_twice_label(tmp_output_dir):
    paths = [INPUT_DIR / "kmod_ip1", INPUT_DIR / "kmod_ip1"]
    with pytest.raises(KeyError) as error:
        merge_kmod_results.merge_kmod_results(kmod_dirs=paths, outputdir=tmp_output_dir)
        assert 'ip1B1' in str(error.value)


@pytest.mark.basic
def test_incorrect_paths():
    paths = [Path("IchBinDerAntonAusTirol"), Path("Pizza4Fromages")]

    with pytest.raises(Exception) as error:
        merge_kmod_results.merge_kmod_results(kmod_dirs=paths)

    assert "Directory IchBinDerAntonAusTirol does not exist" in str(error.value)


# Helper -----------------------------------------------------------------------


@pytest.fixture
def _tfs_file():
    return tfs.read_tfs(INPUT_DIR / "test_imbalance.tfs", index=merge_kmod_results.NAME)
