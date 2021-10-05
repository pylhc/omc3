import re
from pathlib import Path
from typing import List, Union

import pandas as pd
import pytest
import tfs
from numpy import mean, sqrt, square

from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.optics_measurements.constants import F1001, F1010, IMAG, NAME, REAL

INPUTS = Path(__file__).parent.parent / "inputs"
COUPLING_INPUTS = INPUTS / "coupling"
RDT_LIMIT = 1.0e-3


@pytest.mark.basic
def test_coupling_beam_1_against_getllm(tmp_path):
    output_dir = tmp_path / "optics_outputs"
    output_dir.mkdir()
    f1001, f1010 = _run_analysis(output_dir, 1, "beam1")
    coupling_getllm = tfs.read(COUPLING_INPUTS / "getllm_beam1.tfs", index=NAME)

    assert _rms_arc(f1001[IMAG] - coupling_getllm[f"{F1001}I"]) < RDT_LIMIT, "f1001_imag didn't match getllm output"
    assert _rms_arc(f1001[REAL] - coupling_getllm[f"{F1001}R"]) < RDT_LIMIT, "f1001_real didn't match getllm output"
    assert _rms_arc(f1010[IMAG] - coupling_getllm[f"{F1010}I"]) < RDT_LIMIT, "f1010_imag didn't match getllm output"
    assert _rms_arc(f1010[REAL] - coupling_getllm[f"{F1010}R"]) < RDT_LIMIT, "f1010_real didn't match getllm output"


@pytest.mark.basic
def test_coupling_beam_1_against_optics_functions(tmp_path):
    output_dir = tmp_path / "optics_outputs"
    output_dir.mkdir()
    f1001, f1010 = _run_analysis(output_dir, 1, "beam1")
    coupling_cmatrix = tfs.read(COUPLING_INPUTS / "cmatrix_beam1.tfs", index=NAME)

    assert _rms_arc(f1001[IMAG] - coupling_cmatrix[f"{F1001}I"]) < RDT_LIMIT, "f1001_imag didn't match optics_functions output"
    assert _rms_arc(f1001[REAL] - coupling_cmatrix[f"{F1001}R"]) < RDT_LIMIT, "f1001_real didn't match optics_functions output"
    assert _rms_arc(f1010[IMAG] - coupling_cmatrix[f"{F1001}I"]) < RDT_LIMIT, "f1010_imag didn't match optics_functions output"
    assert _rms_arc(f1010[REAL] - coupling_cmatrix[f"{F1010}R"]) < RDT_LIMIT, "f1010_real didn't match optics_functions output"


@pytest.mark.basic
def test_coupling_beam_4_against_getllm(tmp_path):
    output_dir = tmp_path / "optics_outputs"
    output_dir.mkdir()
    f1001, f1010 = _run_analysis(output_dir, 2, "beam4")
    coupling_getllm = tfs.read(COUPLING_INPUTS / "getllm_beam4.tfs", index=NAME)

    assert _rms_arc(f1001[IMAG] - coupling_getllm[f"{F1001}I"]) < RDT_LIMIT, "f1001_imag didn't match getllm output"
    assert _rms_arc(f1001[REAL] - coupling_getllm[f"{F1001}R"]) < RDT_LIMIT, "f1001_real didn't match getllm output"
    assert _rms_arc(f1010[IMAG] - coupling_getllm[f"{F1010}I"]) < RDT_LIMIT, "f1010_imag didn't match getllm output"
    assert _rms_arc(f1010[REAL] - coupling_getllm[f"{F1010}R"]) < RDT_LIMIT, "f1010_real didn't match getllm output"


@pytest.mark.basic
def test_coupling_beam_4_against_optics_functions(tmp_path):
    output_dir = tmp_path / "optics_outputs"
    output_dir.mkdir()
    f1001, f1010 = _run_analysis(output_dir, 2, "beam4")
    coupling_cmatrix = tfs.read(COUPLING_INPUTS / "cmatrix_beam4.tfs", index=NAME)

    # flip of signs in this test because optics-functions is in anti-phase due to different conventions
    assert _rms_arc(f1001[IMAG] - coupling_cmatrix[f"{F1001}I"]) < RDT_LIMIT, "f1001_imag didn't match optics_functions output"
    assert _rms_arc(f1001[REAL] + coupling_cmatrix[f"{F1001}R"]) < RDT_LIMIT, "f1001_real didn't match optics_functions output"
    assert _rms_arc(f1010[IMAG] - coupling_cmatrix[f"{F1010}I"]) < RDT_LIMIT, "f1010_imag didn't match optics_functions output"
    assert _rms_arc(f1010[REAL] + coupling_cmatrix[f"{F1010}R"]) < RDT_LIMIT, "f1010_real didn't match optics_functions output"


# ----- Helpers ----- #


def _run_analysis(output_dir: Union[str, Path], beam: int, sdds_input: str):
    """Run hole_in_one on provided data, return the loaded result coupling files for f1001 and f1010."""
    hole_in_one_entrypoint(
        optics=True,
        accel="lhc",
        year="2018",
        beam=beam,
        model_dir=COUPLING_INPUTS / f"model_b{beam}",
        files=[f"{COUPLING_INPUTS}/{sdds_input}.sdds"],
        compensation="none",
        outputdir=output_dir,
        only_coupling=True,
    )
    f1001 = tfs.read(output_dir / f"{F1001.lower()}.tfs", index=NAME)
    f1010 = tfs.read(output_dir / f"{F1010.lower()}.tfs", index=NAME)
    return f1001, f1010


def _rms_arc(data: pd.DataFrame) -> pd.DataFrame:
    """Get rms of the provided data in LHC arcs BPMs."""
    arc_data = data[_select_arc_bpms(data.index)].copy()
    return sqrt(mean(square(arc_data)))


def _select_arc_bpms(names: pd.Series) -> List[str]:
    """Select LHC arc BPMs (number >12)."""
    bpm_matches = [(name, re.match(r"BPM[^.]*\.(\d+)[LR]\d\.B[12]", name)) for name in names]
    return [name for (name, x) in bpm_matches if x is not None and int(x[1]) > 12]
