import tfs
import re
from pathlib import Path
from omc3.hole_in_one import hole_in_one_entrypoint
from numpy import sqrt, mean, square

IN_DIR = Path(__file__).parent.parent / "inputs/coupling"
LIMIT = 1.0e-3


def test_coupling_beam_1(tmpdir):
    _test_coupling_beam_b(tmpdir, 1, "beam1")


def test_coupling_beam_4(tmpdir):
    _test_coupling_beam_b(tmpdir, 2, "beam4")


def _test_coupling_beam_b(tmpdir, beam, input):
    hole_in_one_entrypoint(
        optics=True,
        accel="lhc",
        year="2018",
        beam=beam,
        model_dir=IN_DIR / f"model_b{beam}",
        files=[f"{IN_DIR}/{input}.sdds"],
        compensation="none",
        outputdir=tmpdir,
        only_coupling=True,
    )
    f1001 = tfs.read(tmpdir / "f1001.tfs", index='NAME')
    f1010 = tfs.read(tmpdir / "f1010.tfs", index='NAME')
    coupling_getllm = tfs.read(IN_DIR / f"getllm_{input}.tfs", index='NAME')
    coupling_cmatrix = tfs.read(IN_DIR / f"cmatrix_{input}.tfs", index='NAME')

    assert _rms_arc(f1001["F1001I"] - coupling_getllm["F1001I"]) < LIMIT, "f1001_imag didn't match getllm output"
    assert _rms_arc(f1001["F1001R"] - coupling_getllm["F1001R"]) < LIMIT, "f1001_real didn't match getllm output"
    assert _rms_arc(f1010["F1010I"] - coupling_getllm["F1010I"]) < LIMIT, "f1010_imag didn't match getllm output"
    assert _rms_arc(f1010["F1010R"] - coupling_getllm["F1010R"]) < LIMIT, "f1010_real didn't match getllm output"

    assert _rms_arc(f1001["F1001I"] - coupling_cmatrix["F1001I"]) < LIMIT, "f1001_imag didn't match optics_functions output"
    assert _rms_arc(f1001["F1001R"] - coupling_cmatrix["F1001R"]) < LIMIT, "f1001_real didn't match optics_functions output"
    assert _rms_arc(f1010["F1010I"] - coupling_cmatrix["F1010I"]) < LIMIT, "f1010_imag didn't match optics_functions output"
    assert _rms_arc(f1010["F1010R"] - coupling_cmatrix["F1010R"]) < LIMIT, "f1010_real didn't match optics_functions output"


# --------------------------------------------------------------------------------------------------
# ---- helper functions (could maybe be collected in a testing_utilities module) -------------------
# --------------------------------------------------------------------------------------------------
def _rms_arc(data):
    arc_data = data[_select_arc(data.index)].copy()
    return sqrt(mean(square(arc_data)))


def _select_arc(names):
    bpm_matches = [(name, re.match(r"BPM[^.]*\.(\d+)[LR]\d\.B[12]", name)) for name in names]
    return [name for (name, x) in bpm_matches if x is not None and int(x[1]) > 12]
