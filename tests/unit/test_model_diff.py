from pathlib import Path

import numpy as np
import tfs
import pytest

from omc3.correction.model_appenders import add_coupling_to_model
from omc3.correction.model_diff import diff_twiss_parameters
from omc3.global_correction import OPTICS_PARAMS_CHOICES
from omc3.optics_measurements.toolbox import ang_diff, ang_sum
from omc3.optics_measurements.constants import (
    NAME, DELTA, BETA, TUNE, NORM_DISPERSION, DISPERSION, PHASE_ADV, PHASE
)

from tests.conftest import INPUTS, MODELS

INPUTS = Path(__file__).parent.parent / 'inputs'

MODEL_INJ_BEAM1 = MODELS / "2018_inj_b1_11m" / "twiss.dat"
MODEL_INJ_BEAM1_MQ_ERR = INPUTS / "correction" / "inj_beam1" / "twiss_quadrupole_error.dat"
MODEL_INJ_BEAM1_MQS_ERR = INPUTS / "correction" / "inj_beam1" / "twiss_skew_quadrupole_error.dat"

EPS = 1e-12  # numerical accuracy, as the inverse calculation will not give the exact starting values


@pytest.mark.parametrize('model_error_path', (MODEL_INJ_BEAM1_MQ_ERR, MODEL_INJ_BEAM1_MQS_ERR), ids=('MQ', 'MQS'))
@pytest.mark.basic
def test_simple_diff(model_error_path):
    """Asserts that the diff_twiss_parameters functions perform the correct
    calculations by applying the respective inverse calculations on
    model, model_errors and delta."""
    model = tfs.read(MODEL_INJ_BEAM1, index=NAME)
    model = add_coupling_to_model(model)

    model_errors = tfs.read(model_error_path, index=NAME)
    model_errors = add_coupling_to_model(model_errors)

    diff = diff_twiss_parameters(model_errors, model, OPTICS_PARAMS_CHOICES)

    for param in OPTICS_PARAMS_CHOICES:
        delta = f"{DELTA}{param}"
        if param[:-1] == BETA:
            check = model[param] * (1 + diff[delta]) - model_errors[param]
        elif param[:-1] == NORM_DISPERSION:
            beta, disp = f"{BETA}{param[-1]}", f"{DISPERSION}{param[-1]}"
            check = model[disp]/np.sqrt(model[beta]) + diff[delta] - model_errors[disp]/np.sqrt(model_errors[beta])
        elif param[:-1] in (PHASE, PHASE_ADV):
            param = f"{PHASE_ADV}{param[-1]}"
            check = ang_diff(ang_sum(np.diff(model[param]), diff[delta][1:]), np.diff(model_errors[param]))
        elif param == TUNE:
            check = [model[f"{param}{i}"] + diff[f"{delta}{i}"] - model_errors[f"{param}{i}"] for i in "12"]
        else:
            check = model[param] + diff[delta] - model_errors[param]

        assert all(np.abs(check) < EPS)

