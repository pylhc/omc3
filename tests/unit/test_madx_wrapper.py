from os.path import exists, isfile, join

import pytest

from omc3 import madx_wrapper
from omc3.utils.contexts import silence, temporary_dir

from . import context


def test_with_macro():
    """ Checks:
         - Output_file is created.
         - Macros resolve correctly.
    """
    content = (
        "!@require lhc\n"
        "!@require general.macros.madx\n"
    )
    resolved_lines = [
        'call,file="{}";'.format(join(madx_wrapper.LIB, "lhc.macros.madx")),
        'call,file="{}";'.format(join(madx_wrapper.LIB, "general.macros.madx"))
    ]

    with temporary_dir() as tmpdir:
        outfile = join(tmpdir, "job.with_macro.madx")
        with silence():
            madx_wrapper.resolve_and_run_string(content, output_file=outfile, cwd=tmpdir)
        assert exists(outfile)
        with open(outfile, "r") as of:
            out_lines = of.read().split("\n")
        out_lines = [ol.replace(" ", "") for ol in out_lines]
        assert all([r in out_lines for r in resolved_lines])


def test_with_nonexistent_file():
    """ Checks:
         - Madx crashes when tries to call a non-existent file
         - Logfile is created
         - Error message is read from log
    """
    call_file = "does_not_exist.madx"
    content = "call, file ='{:s}';".format(call_file)
    with temporary_dir() as tmpdir:
        log_file = join(tmpdir, "tmp_log.log")
        with pytest.raises(madx_wrapper.MadxError) as e:
            madx_wrapper.resolve_and_run_string(content, log_file=log_file, cwd=tmpdir)
        assert isfile(log_file)
        assert call_file in str(e.value)
