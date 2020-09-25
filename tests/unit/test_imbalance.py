import os
import shutil
from pathlib import Path

import pytest
import tfs

from omc3.definitions.constants import PLANES
from omc3.scripts import merge_kmod_results
from omc3.scripts.merge_kmod_results import BETASTAR, ERR

CURRENT_DIR = Path(__file__).parent
RESULTS = "results.tfs"
LSA_RESULTS = "lsa_results.tfs"


def test_result_tfs(_tfs_file):
    res = merge_kmod_results.get_lumi_imbalance(_tfs_file)

    assert res["imbalance"] == 0.974139299943968
    assert res["relative_error"] == 0.003859317636164786
    assert res["eff_beta_ip1"] == 0.39800399369855577
    assert res["rel_error_ip1"] == 0.0015434779687894644
    assert res["eff_beta_ip5"] == 0.4085698972636139
    assert res["rel_error_ip5"] == 0.0023158396673753213


def test_wrong_columns(_tfs_wrong_columns):
    res = merge_kmod_results._validate_for_imbalance(_tfs_wrong_columns)
    assert not res


def test_twice_label(_tfs_twice_label):
    with pytest.raises(KeyError) as error:
        merge_kmod_results._validate_for_imbalance(_tfs_twice_label)
    assert "Duplicate label 'ip1B1' in dataframe's columns" in str(error.value)


def test_incorrect_paths():
    paths = [Path("IchBinDerAntonAusTirol"), Path("Pizza4Fromages")]

    with pytest.raises(Exception) as error:
        merge_kmod_results.merge_and_copy_kmod_output({"kmod_dirs": paths, "outputdir": Path(".")})

    assert "Directory IchBinDerAntonAusTirol does not exist" in str(error.value)


def test_lsa_merge(_tmp_dir):
    base = CURRENT_DIR.parent / "inputs" / "merge_kmod"
    paths = [base / "kmod_ip1", base / "kmod_ip5"]

    merge_kmod_results.merge_and_copy_kmod_output({"kmod_dirs": paths, "outputdir": _tmp_dir})

    res_lsa_tfs = tfs.read_tfs(_tmp_dir / LSA_RESULTS)
    control_tfs = tfs.read_tfs(base / LSA_RESULTS)

    assert res_lsa_tfs.equals(control_tfs)


def _get_file(tmp_path, path) -> tfs.TfsDataFrame:
    # Copy the file to a temp directory so that we don't modify the source
    src = CURRENT_DIR.parent / "inputs" / "merge_kmod" / path

    d = tmp_path / "imbalance"
    d.mkdir()
    dst = d / path

    shutil.copyfile(src, dst)
    return tfs.read_tfs(dst)


@pytest.fixture()
def _tmp_dir(tmp_path):
    d = tmp_path / "imbalance"
    d.mkdir()

    return d


@pytest.fixture()
def _tfs_file(tmp_path):
    path = "test_imbalance.tfs"
    return _get_file(tmp_path, path)


@pytest.fixture()
def _tfs_wrong_columns(tmp_path):
    path = "test_wrong_columns.tfs"
    return _get_file(tmp_path, path)


@pytest.fixture()
def _tfs_wrong_label(tmp_path):
    path = "test_wrong_label.tfs"
    return _get_file(tmp_path, path)


@pytest.fixture()
def _tfs_twice_label(tmp_path):
    path = "test_twice_label.tfs"
    return _get_file(tmp_path, path)
