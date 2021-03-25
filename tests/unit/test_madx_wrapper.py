from pathlib import Path

import pytest

from omc3 import madx_wrapper, model
from omc3.utils.contexts import silence

LIB = Path(model.__file__) / "madx_macros"


@pytest.mark.basic
def test_with_macro(tmp_path):
    """ Checks:
         - Output_file is created.
    """
    content = f"call,file='{str(LIB / 'lhc.macros.madx')}';\ncall,file='{str(LIB / 'general.macros.madx')}';\n"
    outfile = tmp_path / "job.with_macro.madx"
    with silence():
        madx_wrapper.run_string(content, output_file=outfile, cwd=tmp_path)
    assert outfile.exists()
    assert content == outfile.read_text()


@pytest.mark.basic
def test_with_nonexistent_file(tmp_path):
    """ Checks:
         - Madx crashes when tries to call a non-existent file
         - Logfile is created
         - Error message is read from log
    """
    call_file = "does_not_exist.madx"
    content = "call, file ='{:s}';".format(call_file)
    log_file = tmp_path / "tmp_log.log"
    with pytest.raises(madx_wrapper.MadxError) as e:
        madx_wrapper.run_string(content, log_file=log_file, cwd=tmp_path)
    assert log_file.exists()
    assert call_file in str(e.value)
