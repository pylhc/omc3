import tfs
import re
import pytest
from pathlib import Path
from omc3.hole_in_one import hole_in_one_entrypoint
from numpy import sqrt, mean, square

INPUTS = Path(__file__).parent.parent / 'inputs'
COUPLING_INPUTS = INPUTS / "coupling"
RDT_LIMIT = 1.0e-3


@pytest.mark.basic
def test_coupling_beam_1_against_getllm(tmpdir):
    f1001, f1010 = _run_analysis(tmpdir, 1, "beam1")
    coupling_getllm = tfs.read(COUPLING_INPUTS / "getllm_beam1.tfs", index='NAME')

    assert _rms_arc(f1001["F1001I"] - coupling_getllm["F1001I"]) < RDT_LIMIT, "f1001_imag didn't match getllm output"
    assert _rms_arc(f1001["F1001R"] - coupling_getllm["F1001R"]) < RDT_LIMIT, "f1001_real didn't match getllm output"
    assert _rms_arc(f1010["F1010I"] - coupling_getllm["F1010I"]) < RDT_LIMIT, "f1010_imag didn't match getllm output"
    assert _rms_arc(f1010["F1010R"] - coupling_getllm["F1010R"]) < RDT_LIMIT, "f1010_real didn't match getllm output"


@pytest.mark.basic
def test_coupling_beam_1_against_optics_functions(tmpdir):
    f1001, f1010 = _run_analysis(tmpdir, 1, "beam1")
    coupling_cmatrix = tfs.read(COUPLING_INPUTS / "cmatrix_beam1.tfs", index='NAME')

    assert _rms_arc(f1001["F1001I"] - coupling_cmatrix["F1001I"]) < RDT_LIMIT, "f1001_imag didn't match optics_functions output"
    assert _rms_arc(f1001["F1001R"] - coupling_cmatrix["F1001R"]) < RDT_LIMIT, "f1001_real didn't match optics_functions output"
    assert _rms_arc(f1010["F1010I"] - coupling_cmatrix["F1010I"]) < RDT_LIMIT, "f1010_imag didn't match optics_functions output"
    assert _rms_arc(f1010["F1010R"] - coupling_cmatrix["F1010R"]) < RDT_LIMIT, "f1010_real didn't match optics_functions output"


@pytest.mark.basic
def test_coupling_beam_4_against_getllm(tmpdir):
    f1001, f1010 = _run_analysis(tmpdir, 2, "beam4")
    coupling_getllm = tfs.read(COUPLING_INPUTS / "getllm_beam4.tfs", index='NAME')

    assert _rms_arc(f1001["F1001I"] - coupling_getllm["F1001I"]) < RDT_LIMIT, "f1001_imag didn't match getllm output"
    assert _rms_arc(f1001["F1001R"] - coupling_getllm["F1001R"]) < RDT_LIMIT, "f1001_real didn't match getllm output"
    assert _rms_arc(f1010["F1010I"] - coupling_getllm["F1010I"]) < RDT_LIMIT, "f1010_imag didn't match getllm output"
    assert _rms_arc(f1010["F1010R"] - coupling_getllm["F1010R"]) < RDT_LIMIT, "f1010_real didn't match getllm output"


@pytest.mark.basic
def test_coupling_beam_4_against_optics_functions(tmpdir):
    f1001, f1010 = _run_analysis(tmpdir, 2, "beam4")
    coupling_cmatrix = tfs.read(COUPLING_INPUTS / "cmatrix_beam4.tfs", index='NAME')

    assert _rms_arc(f1001["F1001I"] - coupling_cmatrix["F1001I"]) < RDT_LIMIT, "f1001_imag didn't match optics_functions output"
    assert _rms_arc(f1001["F1001R"] + coupling_cmatrix["F1001R"]) < RDT_LIMIT, "f1001_real didn't match optics_functions output"
    assert _rms_arc(f1010["F1010I"] - coupling_cmatrix["F1010I"]) < RDT_LIMIT, "f1010_imag didn't match optics_functions output"
    assert _rms_arc(f1010["F1010R"] + coupling_cmatrix["F1010R"]) < RDT_LIMIT, "f1010_real didn't match optics_functions output"


def _run_analysis(tmpdir, beam, input):
    hole_in_one_entrypoint(
        optics=True,
        accel="lhc",
        year="2018",
        beam=beam,
        model_dir=COUPLING_INPUTS / f"model_b{beam}",
        files=[f"{COUPLING_INPUTS}/{input}.sdds"],
        compensation="none",
        outputdir=tmpdir,
        only_coupling=True,
    )
    f1001 = tfs.read(tmpdir / "f1001.tfs", index='NAME')
    f1010 = tfs.read(tmpdir / "f1010.tfs", index='NAME')

    return f1001, f1010


# --------------------------------------------------------------------------------------------------
# ---- helper functions (could maybe be collected in a testing_utilities module) -------------------
# --------------------------------------------------------------------------------------------------
def _rms_arc(data):
    arc_data = data[_select_arc(data.index)].copy()
    return sqrt(mean(square(arc_data)))


# select LHC arc BPMs (meaning BPM number > 12) from regular expression.
# the regex filters out the BPM number e.g. 12 from BPM.12R8.B1
def _select_arc(names):
    bpm_matches = [(name, re.match(r"BPM[^.]*\.(\d+)[LR]\d\.B[12]", name)) for name in names]
    return [name for (name, x) in bpm_matches if x is not None and int(x[1]) > 12]
