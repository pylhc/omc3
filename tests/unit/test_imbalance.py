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
    res = luminosity_imbalance.get_imbalance(_tfs_file)

    assert res['imbalance'] == 0.974139299943968
    assert res['relative_error'] == 0.003859317636164786
    assert res['eff_beta_ip1'] == 0.39800399369855577
    assert res['rel_error_ip1'] == 0.0015434779687894644
    assert res['eff_beta_ip5'] == 0.4085698972636139
    assert res['rel_error_ip5'] == 0.0023158396673753213


def test_wrong_columns(_tfs_wrong_columns):
    with pytest.raises(KeyError) as error:
        luminosity_imbalance._validate_tfs(_tfs_wrong_columns)

    msg = 'Expected columns in the TFS file not found. Expected columns: '
    assert msg in str(error.value)

    columns = [f'{BETASTAR}{p}' for p in PLANES] + \
              [f'{ERR}{BETASTAR}{p}' for p in PLANES]
    for column in columns:
        assert column in str(error.value)


def test_wrong_label(_tfs_wrong_label):
    with pytest.raises(KeyError) as error:
        luminosity_imbalance._validate_tfs(_tfs_wrong_label)

    msg = "The following required labels are not found in dataframe: "\
          "ip5B1"
    assert msg in str(error.value)


def test_twice_label(_tfs_twice_label):
    with pytest.raises(KeyError) as error:
        luminosity_imbalance._validate_tfs(_tfs_twice_label)

    msg = 'Found label ip1B1 several times. Expected only once'
    assert msg in str(error.value)


def test_incorrect_paths():
    paths = [Path('IchBinAntonAusTirol'), Path('Pizza4Fromages')]

    with pytest.raises(Exception) as error:
        luminosity_imbalance.merge_and_copy_kmod_output({'kmod_dirs': paths,
                                                         'res_dir': Path('.')})

    msg = 'All directories should account for a total of 4 ipBx directories inside'
    assert msg in str(error.value)


def _get_file(tmp_path, path):
    # Copy the file to a temp directory so that we don't modify the source
    src = CURRENT_DIR.parent / 'inputs' / 'lumi_imbalance' / path

    d = tmp_path / 'imbalance'
    d.mkdir()
    dst = d / path

    shutil.copyfile(src, dst)
    return tfs.read_tfs(dst)


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
