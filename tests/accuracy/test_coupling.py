from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
import tfs
from numpy import mean, sqrt, square

from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.optics_measurements.constants import F1001, F1010, IMAG, NAME, REAL
from omc3.optics_measurements.phase import CompensationMode
from tests.conftest import INPUTS

COUPLING_INPUTS: Path = INPUTS / "coupling"
MODELS: Path = INPUTS / "models"
RDT_LIMIT: float = 5.0e-4

BEAM_NO: dict[int, int] = {1: 1, 4: 2}


@pytest.mark.basic
@pytest.mark.parametrize("beam", (1, 4), ids=["Beam1", "Beam4"])
def test_coupling_tracking(tmp_path, beam):
    """ Compares coupling on tracking data with getllm output and optics_functions output. """
    hole_in_one_entrypoint(
        optics=True,
        accel="lhc",
        year="2018",
        beam=BEAM_NO[beam],
        model_dir=MODELS / f"2018_40cm_b{BEAM_NO[beam]}",
        files=[COUPLING_INPUTS / f"beam{beam}.sdds"],
        compensation=CompensationMode.NONE,
        outputdir=tmp_path,
        only_coupling=True,
    )
    f1001 = tfs.read(tmp_path / f"{F1001.lower()}.tfs", index=NAME)
    f1010 = tfs.read(tmp_path / f"{F1010.lower()}.tfs", index=NAME)
    
    coupling_getllm = tfs.read(COUPLING_INPUTS / f"getllm_beam{beam}.tfs", index=NAME)
    _compare_coupling(f1001, f1010, coupling_getllm)

    coupling_cmatrix = tfs.read(COUPLING_INPUTS / f"cmatrix_beam{beam}.tfs", index=NAME)
    _compare_coupling(f1001, f1010, coupling_cmatrix, flip_real = (beam == 4) )



# TODO:This tests does not really work. Not sure why or what the best method is to make it work.
# To be discussed. Data (model, sdds, getllm) available upon request or from BetaBeat folder 
# 2018-06-15, kick at 15:55 (getLLM folder is inside Measurement folder for that kick). jdilly 2024 

# @pytest.mark.extended
# @pytest.mark.parametrize("beam", (1, 2), ids=["Beam1", "Beam2"])
# def test_coupling_lhc_data(tmp_path, beam):
#     """ Compares coupling on lhc data with getllm output and optics_functions output. """

#     getllm_dir = COUPLING_INPUTS / f"getllm_25cm_b{beam}"
#     sdds_files = {
#         1: getllm_dir / "Beam1@Turn@2018_06_15@15_55_58_234.sdds",
#         2: getllm_dir / "Beam2@Turn@2018_06_15@15_55_50_588.sdds",
#     }
#     compensation = CompensationMode.EQUATION
#     suffix = {
#         CompensationMode.NONE: "",
#         CompensationMode.MODEL: "_free2",
#         CompensationMode.EQUATION: "_free",
#     }[compensation]

#     hole_in_one_entrypoint(
#         harpy=True,
#         optics=True,
#         clean=True,
#         unit="mm",
#         output_bits=0,
#         turn_bits=10,
#         to_write=[],
#         first_bpm="BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2",
#         tunes=[0.268, 0.325, 0.0],
#         nattunes=[0.28, 0.31, 0.0],
#         tolerance=0.004,
#         opposite_direction=(beam == 2),
#         # autotunes="transverse",
#         # natdeltas=[+0.012, -0.015, 0.0],
#         accel="lhc",
#         year="2018",
#         beam=beam,
#         model_dir=MODELS / f"2018_inj_b{beam}_25cm",
#         files=[sdds_files[beam]],
#         compensation=compensation,
#         outputdir=tmp_path,
#         only_coupling=True,
#     )
#     f1001 = tfs.read(tmp_path / f"{F1001.lower()}.tfs", index=NAME)
#     f1010 = tfs.read(tmp_path / f"{F1010.lower()}.tfs", index=NAME)
    
#     coupling_getllm = tfs.read(getllm_dir / f"getcouple{suffix}.out", index=NAME)
#     _compare_coupling(f1001, f1010, coupling_getllm)


# ----- Helpers ----- #

def _compare_coupling(f1001: pd.DataFrame, f1010: pd.DataFrame, coupling_compare: pd.DataFrame, flip_real: bool = False):
    """ Compare the coupling between the omc3 values and from the loaded tfs files. """
    real_sign = -1 if flip_real else 1  # due to different conventions in getllm and optics functions

    rms_f1001_i = _rms_arc(f1001[IMAG] -             coupling_compare[f"{F1001}I"])
    rms_f1001_r = _rms_arc(f1001[REAL] - real_sign * coupling_compare[f"{F1001}R"])
    rms_f1010_i = _rms_arc(f1010[IMAG] -             coupling_compare[f"{F1010}I"])
    rms_f1010_r = _rms_arc(f1010[REAL] - real_sign * coupling_compare[f"{F1010}R"])

    assert  rms_f1001_i < RDT_LIMIT, "f1001 imag didn't match"
    assert  rms_f1001_r < RDT_LIMIT, "f1001 real didn't match"
    assert  rms_f1010_i < RDT_LIMIT, "f1010 imag didn't match"
    assert  rms_f1010_r < RDT_LIMIT, "f1010 real didn't match"


def _rms_arc(data: pd.DataFrame) -> pd.DataFrame:
    """Get rms of the provided data in LHC arcs BPMs."""
    arc_data = data[_select_arc_bpms(data.index)].copy()
    return sqrt(mean(square(arc_data)))


def _select_arc_bpms(names: pd.Series) -> list[str]:
    """Select LHC arc BPMs (number >12)."""
    bpm_matches = [(name, re.match(r"BPM[^.]*\.(\d+)[LR]\d\.B[12]", name)) for name in names]
    return [name for (name, x) in bpm_matches if x is not None and int(x[1]) > 12]
