"""
Additional tools for pytests.
The name ``conftest.py`` is chosen as it is used by pytest.
Fixtures defined in here are discovered by all tests automatically.

See also https://stackoverflow.com/a/34520971 .
"""
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


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
    script = kwargs.get('script', 'somescript.py')
    args_save = sys.argv.copy()
    sys.argv = [script] + list(args)
    yield
    sys.argv = args_save


@pytest.fixture
def tmp_output_dir(tmp_path, request):
    """ Fixture for a temporary directory. If the caller module has a
    global parameter DEBUG set to ``True``, the directory will be created
    as ``temp_ module_name / test_name`` in the current folder and it will
    not be deleted after test-run. """
    # init ---
    debug = getattr(request.module, "DEBUG", False)
    test_name = request.node.name
    module_name = request.module.__name__.split('.')[-1]

    path = tmp_path
    if debug:
        path = Path(f"tmp_{module_name}", test_name)

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

    # yield ---
    yield path

    # cleanup ---
    if not debug:
        shutil.rmtree(path)
