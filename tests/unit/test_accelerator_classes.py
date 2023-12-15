import pytest
from pathlib import Path

from tfs import TfsDataFrame
from omc3.model.accelerators.accelerator import Accelerator, AcceleratorDefinitionError
from omc3.model.accelerators.lhc import Lhc
from omc3.model.accelerators.ps import Ps
from omc3.model.accelerators.psbooster import Psbooster
from omc3.model.accelerators.skekb import SKekB

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

@pytest.mark.basic
def test_psbase_best_knowledge():
    # PS and Booster don't have best knowledge models (yet)

    accel = Ps(
            year="2021",
            scenario="lhc",
            cycle_point="0_injection",
            str_file="ps_inj_lhc.str",
            tune_method="qf"
            )

    with pytest.raises(AttributeError):
        _ = accel.get_base_madx_script(best_knowledge=True)

    accel = Psbooster(
            year="2021",
            scenario="lhc",
            cycle_point="0_injection",
            str_file="psb_inj_lhc.str",
            ring=1,
            )

    with pytest.raises(AttributeError):
        _ = accel.get_base_madx_script(best_knowledge=True)

@pytest.mark.basic
def test_skekb():

    accel = SKekB(ring="ler")
    assert accel.ring == "ler"

    # try invalid value
    with pytest.raises(AcceleratorDefinitionError):
        accel.ring = "Beam1"


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
