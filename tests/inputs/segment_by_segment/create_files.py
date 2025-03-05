""" 
Create input files for Segment-by-Segment tests
-----------------------------------------------

This module creates the input files for the Segment-by-Segment tests,
including the lhc-models (with reduced twiss-elements).

Measurements are created directly from the LHC model, disturbed by the errors 
specified in the here written `my_errors.madx` file 
via the `fake_measurement_from_model` module.

If you want to test the segment-by-segment manually (e.g. with the GUI), 
you need to `KEEP_ACC_MODELS` set to `True`, as otherwise
the model creator will not know, that the models were created using acc-models
(in the tests, the acc-models path is recreated manually, but we do not want
to commit this folder to github).
"""
import shutil
from pathlib import Path

import tfs

from omc3.model.accelerators.lhc import Lhc
from omc3.model.constants import TWISS_DAT, TWISS_ELEMENTS_DAT, Fetcher
from omc3.model_creator import create_instance_and_model
from omc3.optics_measurements.constants import NAME
from omc3.scripts.fake_measurement_from_model import ERRORS
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from omc3.utils import logging_tools
from tests.accuracy.test_sbs import (
    INPUT_MODELS,
    INPUT_SBS,
    OPTICS_30CM_FLAT,
    YEAR,
    create_error_file,
)
from tests.conftest import clone_acc_models, INPUTS_MODEL_DIR_FORMAT

LOG = logging_tools.get_logger(__name__)

TMP_ACC_MODELS: Path = INPUT_SBS / "acc-models-tmp"
KEEP_ACC_MODELS: bool = True  # keep for testing


class PathMaker:
    
    @staticmethod
    def mktemp(*args):
        TMP_ACC_MODELS.mkdir(parents=True, exist_ok=True)
        return TMP_ACC_MODELS

    @staticmethod
    def rmpath():
        if not KEEP_ACC_MODELS:
            shutil.rmtree(TMP_ACC_MODELS)


def create_model(path: Path, beam: int, errors: list[Path]):
    model_path = path / INPUTS_MODEL_DIR_FORMAT.format(
        year=YEAR, beam=beam, tunes="inj", beta="30cm", suffix="_flat"
    )
    modifiers = [OPTICS_30CM_FLAT] + errors 
    accel_opt = dict(
        accel="lhc",
        year=YEAR,
        ats=True,
        beam=beam,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6800.0,
        fetch=Fetcher.PATH,
        path=TMP_ACC_MODELS,
        modifiers=modifiers
    )

    # like from the GUI, dump best knowledge on top of nominal
    accel_nominal: Lhc = create_instance_and_model(
        outputdir=model_path, type="nominal", **accel_opt
    )

    # Compress ---
    elements = accel_nominal.elements.loc[accel_nominal.elements.index.str.match("B|M|IP"), :]
    tfs.write(model_path / TWISS_ELEMENTS_DAT, elements, save_index=NAME)
    
    return model_path


def cleanup(nominal_model: Path, error_model: Path):
    shutil.rmtree(error_model)
    shutil.rmtree(nominal_model / "macros")
    (nominal_model / 'error_deffs.txt').unlink()
    for ini in nominal_model.glob("*.ini"):
        ini.unlink()
    
    if not KEEP_ACC_MODELS:
        (nominal_model / 'acc-models-lhc').unlink()


if __name__ == "__main__":
    if not TMP_ACC_MODELS.is_dir():
        clone_acc_models(PathMaker, "lhc", YEAR)

    errors = create_error_file(INPUT_SBS)
    for beam in (1, 2):
        nominal = create_model(INPUT_MODELS, beam, [])
        error_model = create_model(INPUT_SBS, beam, [errors])
        meas_dir = INPUT_SBS / f"measurement_b{beam}"
        fake_measurement(
            twiss=error_model / TWISS_DAT,
            model=nominal / TWISS_DAT,
            outputdir=meas_dir,
            relative_errors=[1e-4],
            randomize=[ERRORS],
        )
        cleanup(nominal, error_model)

    PathMaker.rmpath()

