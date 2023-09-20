import pytest
from pathlib import Path

from tfs import TfsDataFrame
from omc3.model.accelerators.accelerator import Accelerator
from omc3.model.accelerators.lhc import Lhc

INPUTS = Path(__file__).parent.parent / 'inputs'
MODEL_INJ_BEAM1 = INPUTS / "models" / "2022_inj_b1_adt"
MODEL_INJ_BEAM2 = INPUTS / "models" / "2022_inj_b2_adt"

# the tests load an existing model, test if the accel class can find the "nearest" BPM,
# removes this BPM, test for the second option
# removes also this, test if it fails

@pytest.mark.basic
def test_lhc_adt_b1():
    accel = Lhc(model_dir=MODEL_INJ_BEAM1, beam=1, year="2022")

    _check_exciter_bpm_detection(accel, "X", "BPMWA.B5L4.B1", "BPMWA.A5L4.B1")
    _check_exciter_bpm_detection(accel, "Y", "BPMWA.B5R4.B1", "BPMWA.A5R4.B1")

@pytest.mark.basic
def test_lhc_adt_b2():
    accel = Lhc(model_dir=MODEL_INJ_BEAM2, beam=2, year="2022")

    _check_exciter_bpm_detection(accel, "X", "BPMWA.B5R4.B2", "BPMWA.A5R4.B2")
    _check_exciter_bpm_detection(accel, "Y", "BPMWA.B5L4.B2", "BPMWA.A5L4.B2")

# ---- Helper function -----------------------------------------------------------------------------

def _check_exciter_bpm_detection(accel: Accelerator, plane: str, nearest_bpm: str, second_bpm: str):
    model: TfsDataFrame = accel.model

    ((_, bpm_name), _) = accel.get_exciter_bpm(plane, model.index.to_list())
    assert bpm_name == nearest_bpm

    model.drop(nearest_bpm, inplace=True)

    ((_, bpm_name), _) = accel.get_exciter_bpm(plane, model.index.to_list())
    assert bpm_name == second_bpm

    model.drop(second_bpm, inplace=True)

    with pytest.raises(KeyError):
        accel.get_exciter_bpm(plane, model.index.to_list())
