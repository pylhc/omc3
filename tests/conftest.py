"""
Additional tools for pytests.
The name ``conftest.py`` is chosen as it is used by pytest.
Fixtures defined in here are discovered by all tests automatically.

See also https://stackoverflow.com/a/34520971 .
"""
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from generic_parser import DotDict

from omc3 import model

INPUTS = Path(__file__).parent / 'inputs'


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


# Model fixtures from /inputs/models -------------------------------------------
# Hint: Before adding 25cm models, update files (see inj model folders, jdilly 2021)

@pytest.fixture(scope="module", params=[1, 2])
def model_inj_beams(request, tmp_path_factory):
    """ Fixture for inj model for both beams"""
    return tmp_model(get_tmpdir_path(request, tmp_path_factory), beam=request.param, id_='inj')


@pytest.fixture(scope="module")
def model_inj_beam1(request, tmp_path_factory):
    """ Fixture for inj beam 1 model"""
    return tmp_model(get_tmpdir_path(request, tmp_path_factory), beam=1, id_='inj')


@pytest.fixture(scope="module")
def model_inj_beam2(request, tmp_path_factory):
    """ Fixture for inj beam 2 model"""
    return tmp_model(get_tmpdir_path(request, tmp_path_factory), beam=2, id_='inj')


def tmp_model(temp_dir: Path, beam: int, id_: str):
    """Creates a temporary model directory based on the input/models/model_inj_beam#
    but with the addition of a macros/ directory containing the macros from
    the omc3/models/madx_macros.

    Args:
        temp_dir (Path): Base-Temporary Directory, the model will be in a subfolder
        beam (int): Beam to use
        id_ (str): Model identifyier. `inj` or `25cm`

    Returns:
        A DotDict with the attributes ``path``, the path to the model directory
        and ``settings``, the accelerator class settings for this model.
    """
    tmp_model_path = temp_dir / f"model_{id_}_beam{beam}"
    shutil.copytree(INPUTS / "models" / f"{id_}_beam{beam}", tmp_model_path)  # creates tmp_path dir

    macros_path = tmp_model_path / "macros"
    shutil.copytree(Path(model.__file__).parent / "madx_macros", macros_path)

    return DotDict(
        ats=True,
        beam=beam,
        model_dir=tmp_model_path,
        year="2018",
        accel="lhc",
        energy=0.45 if id_ == 'inj' else 6.5,
    )


def get_tmpdir_path(request, factory):
    """ Get's the name from the request node.
    So that there will be different folders for new scopes.
    Similar to factory.mkdir(), but without creating it.
    """
    name = request.node.name
    name = re.sub(r"[\W]", "_", name)
    MAXVAL = 30
    name = name[:MAXVAL]
    return factory.getbasetemp().joinpath(name)
