import os
import shutil

from pathlib import Path
import pytest

import tfs

from omc3.definitions.constants import PLANES
from omc3.scripts import luminosity_imbalance
from omc3.scripts.luminosity_imbalance import BETASTAR, ERR

CURRENT_DIR = Path(__file__).parent


def test_result_tfs(_tfs_file):
    res = luminosity_imbalance.main({'tfs': _tfs_file})

    assert res['imbalance'] == '0.974'
    assert res['relative_error'] == '0.004'
    assert res['eff_beta_ip1'] == '0.3980'
    assert res['rel_error_ip1'] == '0.0015'
    assert res['eff_beta_ip5'] == '0.409'
    assert res['rel_error_ip5'] == '0.002'


def test_inplace_tfs(_tfs_file):
    res = luminosity_imbalance.main({'tfs': _tfs_file, 'inplace': True})

    assert res['imbalance'] == '0.974'
    assert res['relative_error'] == '0.004'
    assert res['eff_beta_ip1'] == '0.3980'
    assert res['rel_error_ip1'] == '0.0015'
    assert res['eff_beta_ip5'] == '0.409'
    assert res['rel_error_ip5'] == '0.002'

    t = tfs.read_tfs(_tfs_file)
    assert t.headers['LUMINOSITY_IMBALANCE'] == '0.974'
    assert t.headers['RELATIVE_ERROR'] == '0.004'
    assert t.headers['EFF_BETA_IP1'] == '0.3980'
    assert t.headers['REL_ERROR_IP1'] == '0.0015'
    assert t.headers['EFF_BETA_IP5'] == '0.409'
    assert t.headers['REL_ERROR_IP5'] == '0.002'


def test_wrong_path(_tfs_file):
    with pytest.raises(FileNotFoundError):
        luminosity_imbalance.main({'tfs': 'nanananananaBATMAN'})


def test_wrong_columns(_tfs_wrong_columns):
    with pytest.raises(KeyError) as error:
        luminosity_imbalance.main({'tfs': _tfs_wrong_columns})

    msg = 'Expected columns in the TFS file not found. Expected columns: '
    assert msg in str(error.value)

    columns = [f'{BETASTAR}{p}' for p in PLANES] + \
              [f'{ERR}{BETASTAR}{p}' for p in PLANES]
    for column in columns:
        assert column in str(error.value)


def test_wrong_label(_tfs_wrong_label):
    with pytest.raises(KeyError) as error:
        luminosity_imbalance.main({'tfs': _tfs_wrong_label})

    msg = "The following required labels are not found in dataframe: "\
          "ip5B1"
    assert msg in str(error.value)


def test_twice_label(_tfs_twice_label):
    with pytest.raises(KeyError) as error:
        luminosity_imbalance.main({'tfs': _tfs_twice_label})

    msg = 'Found label ip1B1 several times. Expected only once'
    assert msg in str(error.value)


def _get_file(tmp_path, path):
    # Copy the file to a temp directory so that we don't modify the source
    src = CURRENT_DIR.parent / 'inputs' / 'lumi_imbalance' / path

    d = tmp_path / 'imbalance'
    d.mkdir()
    dst = d / path

    shutil.copyfile(src, dst)
    return str(dst)


@pytest.fixture()
def _tfs_file(tmp_path):
    path = 'test_imbalance.tfs'
    return _get_file(tmp_path, path)


@pytest.fixture()
def _tfs_wrong_columns(tmp_path):
    path = 'test_wrong_columns.tfs'
    return _get_file(tmp_path, path)


@pytest.fixture()
def _tfs_wrong_label(tmp_path):
    path = 'test_wrong_label.tfs'
    return _get_file(tmp_path, path)


@pytest.fixture()
def _tfs_twice_label(tmp_path):
    path = 'test_twice_label.tfs'
    return _get_file(tmp_path, path)
