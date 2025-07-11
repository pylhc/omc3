"""
Create input files for Segment-by-Segment tests
-----------------------------------------------

This module creates the input files for the Segment-by-Segment tests.

Measurements are created directly from the SPS model, disturbed by the errors
specified in the here written `my_errors.madx` file
via the `fake_measurement_from_model` module.
"""
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import tfs

from omc3.model.constants import TWISS_DAT, TWISS_ELEMENTS_DAT, Fetcher
from omc3.model_creator import create_instance_and_model
from omc3.optics_measurements.constants import NAME
from omc3.scripts.fake_measurement_from_model import ERRORS
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from omc3.utils import logging_tools
from tests.conftest import clone_acc_models
from tests.unit.test_sps import (
    Q20_STRENGTHS_FILE,
    SPS_DIR,
    create_error_file,
)

if TYPE_CHECKING:
    from omc3.model.accelerators.lhc import Lhc

LOG = logging_tools.get_logger(__name__)

TMP_ACC_MODELS: Path = SPS_DIR / "acc-models-tmp"
KEEP_ACC_MODELS: bool = False  # keep for testing


class PathMaker:

    @staticmethod
    def mktemp(*args):
        TMP_ACC_MODELS.mkdir(parents=True, exist_ok=True)
        return TMP_ACC_MODELS

    @staticmethod
    def rmpath():
        if not KEEP_ACC_MODELS:
            shutil.rmtree(TMP_ACC_MODELS)


def create_model(name: str, errors: list[Path]):
    model_path = SPS_DIR / name
    modifiers = [Q20_STRENGTHS_FILE] + errors
    accel_opt = dict(
        accel="sps",
        nat_tunes=[20.13, 20.18],
        fetch=Fetcher.PATH,
        path=TMP_ACC_MODELS,
        modifiers=modifiers
    )

    accel_nominal: Lhc = create_instance_and_model(
        outputdir=model_path, type="nominal", **accel_opt
    )

    # Compress ---
    elements = accel_nominal.elements.loc[~accel_nominal.elements["KEYWORD"].str.match("DRIFT"), :]
    tfs.write(model_path / TWISS_ELEMENTS_DAT, elements, save_index=NAME)

    return model_path


def cleanup(nominal_model: Path, error_model: Path):
    shutil.rmtree(error_model)
    for ini in nominal_model.glob("*.ini"):
        ini.unlink()

    if not KEEP_ACC_MODELS:
        (nominal_model / 'acc-models-sps').unlink()


if __name__ == "__main__":
    if not TMP_ACC_MODELS.is_dir():
        clone_acc_models(PathMaker, "sps", 2025)

    errors = create_error_file(SPS_DIR)
    nominal = create_model("model_Q20_noacd", [])
    error_model = create_model("model_Q20_errors", [errors])
    meas_dir = SPS_DIR / "fake_measurement_Q20"
    fake_measurement(
        twiss=error_model / TWISS_DAT,
        model=nominal / TWISS_DAT,
        outputdir=meas_dir,
        relative_errors=[1e-4],
        randomize=[ERRORS],
    )
    cleanup(nominal, error_model)

    PathMaker.rmpath()
