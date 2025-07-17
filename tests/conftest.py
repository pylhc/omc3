"""
Additional tools for pytests.
The name ``conftest.py`` is chosen as it is used by pytest.
Fixtures defined in here are discovered by all tests automatically.

See also https://stackoverflow.com/a/34520971 .
"""
import random
import shutil
import string
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import git
import pytest
from generic_parser import DotDict
from pandas._testing import assert_dict_equal
from pandas.testing import assert_frame_equal

import omc3
from omc3 import model

INPUTS = Path(__file__).parent / 'inputs'
MODELS = INPUTS / "models"
OMC3_DIR = Path(omc3.__file__).parent
MADX_MACROS = Path(model.__file__).parent / "madx_macros"
GITLAB_REPO_ACC_MODELS = "https://gitlab.cern.ch/acc-models/acc-models-{}.git"
INPUTS_MODEL_DIR_FORMAT = "{year}_{tunes}_b{beam}_{beta}{suffix}"


@contextmanager
def cli_args(*args, **kwargs):
    """ Provides context to run an entrypoint like with commandline args.
    Arguments are restored after context.

    Args:
        The Commandline args (excluding the script name)

    Keyword Args:
        script: script-name. Used as first commandline-arg.
                Otherwise it's 'somescript.py'
    """
    script = kwargs.get("script", "somescript.py")
    args_save = sys.argv.copy()
    sys.argv = [script] + list(args)
    yield
    sys.argv = args_save


@contextmanager
def mock_module_import(module, replacement):
    """  Temporarily mock a package with something else.
    Needs to be used before the module is imported,
    so it only works on dynamic imports (which we don't have a lot).
    For already imported packages (e.g. module.package) you can use pytest's:
    monkeypatch.setattr(module, "package", replacement)
    """
    orig = sys.modules.get(module)  # get the original module, if present
    sys.modules[module] = replacement  # patch it
    try:
        yield
    finally:
        if orig is not None:  # if the module was installed, restore patch
            sys.modules[module] = orig
        else:  # if the module never existed, remove the key
            del sys.modules[module]

def random_string(length: int,
                  lower: bool = True,
                  upper: bool = True,
                  digits: bool = True,
                  punctuation: bool = False
                  ) -> str:
    """ Returns a random string. """
    charset = ''
    for chars, switch in ((string.ascii_lowercase, lower),
                          (string.ascii_uppercase, upper),
                          (string.digits, digits),
                          (string.punctuation, punctuation)):
        if switch:
            charset = charset + chars
    return ''.join(random.choice(charset) for _ in range(length))


def ids_str(template: str) -> Callable[[Any], str]:
    """ Function generator that can be used for parametrized fixtures,
    to assign the value to a readable string. """
    def to_string(val: Any):
        return template.format(val)
    return to_string


def assert_tfsdataframe_equal(df1, df2, compare_keys=True, **kwargs):
    """ Wrapper to compare two TfsDataFrames with
    `assert_frame_equal` for the data and `assert_dict_equal` for the headers.

    The `kwargs` are passed to `assert_frame_equal`.
    """
    assert_dict_equal(df1.headers, df2.headers, compare_keys=compare_keys)
    assert_frame_equal(df1, df2, **kwargs)


# Model fixtures from /inputs/models -------------------------------------------

@pytest.fixture(scope="module", params=[1, 2], ids=ids_str("beam{}"))
def model_inj_beams(request, tmp_path_factory):
    """ Fixture for inj model for both beams"""
    return tmp_model(tmp_path_factory, beam=request.param, year="2018", tunes="inj", beta="11m")


@pytest.fixture(scope="module")
def model_inj_beam1(request, tmp_path_factory):
    """ Fixture for inj beam 1 model"""
    return tmp_model(tmp_path_factory, beam=1, year="2018", tunes="inj", beta="11m")


@pytest.fixture(scope="module")
def model_inj_beam2(request, tmp_path_factory):
    """ Fixture for inj beam 2 model"""
    return tmp_model(tmp_path_factory, beam=2, year="2018", tunes="inj", beta="11m")


@pytest.fixture(scope="module", params=[1, 2])
def model_25cm_beams(request, tmp_path_factory):
    """ Fixture for 25cm model for both beams"""
    return tmp_model(tmp_path_factory, beam=request.param, year="2018", tunes="col", beta="25cm")


@pytest.fixture(scope="module")
def model_25cm_beam1(request, tmp_path_factory):
    """ Fixture for 25cm beam 1 model"""
    return tmp_model(tmp_path_factory, beam=1, year="2018", tunes="col", beta="25cm")


@pytest.fixture(scope="module")
def model_25cm_beam2(request, tmp_path_factory):
    """ Fixture for 25cm beam 2 model"""
    return tmp_model(tmp_path_factory, beam=2, year="2018", tunes="col", beta="25cm")


@pytest.fixture(scope="module", params=[1, 2])
def model_30cm_flat_beams(request, tmp_path_factory):
    """ Fixture for inj model for both beams"""
    return tmp_model(tmp_path_factory, beam=request.param, year="2025", tunes="inj", beta="30cm", suffix="_flat")


def tmp_model(factory, year: str, beam: int, tunes: str, beta: str, suffix: str = ""):
    """Creates a temporary model directory based on the input/models/model_inj_beam#
    but with the addition of a macros/ directory containing the macros from
    the omc3/models/madx_macros.

    Args:
        factory: tmp_path_factory
        year (str): Year of the model
        beam (int): Beam to use
        tunes (str): inj or col tunes
        beta (str): beta-star value at IP1/IP5
        suffix (str): other suffixes (e.g. `_adt`)

    Returns:
        A DotDict with the attributes ``path``, the path to the model directory
        and ``settings``, the accelerator class settings for this model.
    """
    model_name = INPUTS_MODEL_DIR_FORMAT.format(year=year, beam=beam, tunes=tunes, beta=beta, suffix=suffix)
    tmp_model_path = factory.mktemp(f"model_{model_name}")
    tmp_model_path.rmdir()  # otherwise copytree will complain

    shutil.copytree(MODELS / model_name, tmp_model_path)  # creates tmp_path dir

    macros_path = tmp_model_path / "macros"
    shutil.copytree(MADX_MACROS, macros_path)

    return DotDict(
        ats=True,
        beam=beam,
        model_dir=tmp_model_path,
        year=year,
        accel="lhc",
        energy=450 if beta == '11m' else 6500,
        driven_excitation=None if beta == '11m' else 'acd'
    )


# Acc-Models Fixtures ---

@pytest.fixture(scope="session")
def acc_models_lhc_2025(tmp_path_factory):
    return clone_acc_models(tmp_path_factory, "lhc", 2025)

@pytest.fixture(scope="session")
def acc_models_lhc_2022(tmp_path_factory):
    return clone_acc_models(tmp_path_factory, "lhc", 2022)

@pytest.fixture(scope="session")
def acc_models_lhc_2018(tmp_path_factory):
    return clone_acc_models(tmp_path_factory, "lhc", 2018)

@pytest.fixture(scope="session")
def acc_models_psb_2021(tmp_path_factory):
    return clone_acc_models(tmp_path_factory, "psb", 2021)

@pytest.fixture(scope="session")
def acc_models_sps_2025(tmp_path_factory):
    return clone_acc_models(tmp_path_factory, "sps", 2025)

@pytest.fixture(scope="session")
def acc_models_ps_2021(tmp_path_factory):
    return clone_acc_models(tmp_path_factory, "ps", 2021)

def clone_acc_models(tmp_path_factory, accel: str, year: int):
    """ Clone the acc-models directory for the specified accelerator from github into a temporary directory. """
    tmp_path = tmp_path_factory.mktemp(f"acc-models-{accel}-{year}")
    git.Repo.clone_from(GITLAB_REPO_ACC_MODELS.format(accel), tmp_path, branch=str(year))
    return tmp_path
