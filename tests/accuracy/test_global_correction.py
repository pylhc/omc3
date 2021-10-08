import logging
from pathlib import Path

import numpy as np
import pytest
import tfs
from omc3.correction.constants import (BETA, DISP, NORM_DISP, F1001, F1010, TUNE, PHASE, VALUE, ERROR,
                                       ERR, WEIGHT, DELTA)
from omc3.correction.handler import get_measurement_data, _rms
from omc3.correction.model_appenders import add_coupling_to_model
from omc3.correction.model_diff import diff_twiss_parameters
from omc3.global_correction import global_correction_entrypoint as global_correction, OPTICS_PARAMS_CHOICES
from omc3.optics_measurements.constants import NAME, AMPLITUDE, IMAG, REAL
from omc3.scripts.fake_measurement_from_model import VALUES, ERRORS
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)
# LOG = logging_tools.get_logger('__main__', level_console=logging_tools.MADX)

# Paths ---
INPUTS = Path(__file__).parent.parent / 'inputs'
CORRECTION_INPUTS = INPUTS / "correction"

# Correction Input Parameters ---

RMS_TOL_DICT = {
    f"{PHASE}X": 0.001,
    f"{PHASE}Y": 0.001,
    f"{BETA}X": 0.01,
    f"{BETA}Y": 0.01,
    f"{DISP}X": 0.0015,
    f"{DISP}Y": 0.0015,
    f"{NORM_DISP}X": 0.001,
    f"{TUNE}": 0.01,
    f"{F1001}R": 0.0015,
    f"{F1001}I": 0.0015,
    f"{F1010}R": 0.002,
    f"{F1010}I": 0.002,
}


def get_skew_params(beam):
    twiss = CORRECTION_INPUTS / f"inj_beam{beam}" / f"twiss_skew_quadrupole_error.dat"
    optics_params = OPTICS_PARAMS_CHOICES[8:]
    variables = ["MQSl"]
    fullresponse = "fullresponse_MQSl.h5"
    seed = 2234
    return twiss, optics_params, variables, fullresponse, seed


def get_normal_params(beam):
    twiss = CORRECTION_INPUTS / f"inj_beam{beam}" / f"twiss_quadrupole_error.dat"
    optics_params = OPTICS_PARAMS_CHOICES[:6]
    variables = ["MQY"]
    fullresponse = "fullresponse_MQY.h5"
    seed = 12368
    return twiss, optics_params, variables, fullresponse, seed


@pytest.mark.basic
@pytest.mark.parametrize('orientation', ('skew', 'normal'))
def test_lhc_global_correct(tmp_path, model_inj_beams, orientation):
    """Creates a fake measurement from a modfied model-twiss with (skew)
    quadrupole errors and runs global correction on this measurement.
    It is asserted that the resulting model approaches the modified twiss.
    Hint: the `model_inj_beam1` fixture is defined in `conftest.py`."""
    beam = model_inj_beams.beam
    twiss_path, optics_params, variables, fullresponse, seed = get_skew_params(beam) if orientation == 'skew' else get_normal_params(beam)
    iterations = 2

    # create and load fake measurement
    error_val = 0.1
    twiss_df, model_df, meas_dict = _create_fake_measurement(
        tmp_path, model_inj_beams.model_dir, twiss_path, error_val, optics_params, seed
    )

    # Perform global correction
    global_correction(
        **model_inj_beams,
        # correction params
        meas_dir=tmp_path,
        variable_categories=variables,
        fullresponse_path=model_inj_beams.model_dir / fullresponse,
        optics_params=list(optics_params),
        output_dir=tmp_path,
        weights=[1.0] * len(optics_params),
        svd_cut=0.01,
        max_iter=iterations,
    )

    # Test if corrected model is closer to model used to create measurement
    for iter_step in range(iterations+1):
        if iter_step == 0:
            model_iter_df = model_df
        else:
            model_iter_df = tfs.read(tmp_path / f"twiss_{iter_step}.tfs", index=NAME)
            model_iter_df = add_coupling_to_model(model_iter_df)

        diff_df = diff_twiss_parameters(model_iter_df, twiss_df, optics_params)
        if TUNE in optics_params:
            diff_df.headers[f"{DELTA}{TUNE}"] = np.array([diff_df[f"{DELTA}{TUNE}1"], diff_df[f"{DELTA}{TUNE}2"]])
        diff_rms = {param: _rms(diff_df[f"{DELTA}{param}"]) for param in optics_params}

        # ############ FOR DEBUGGING #############
        # print(f"ITERATION {iter_step}")
        # for param in optics_params:
        #     print(f"{param}: {diff_rms[param]}")
        # print(f"Sum: {sum(diff_rms.values())}")
        # print()
        # continue
        # ########################################

        if iter_step > 0:
            # assert RMS after correction smaller than tolerances
            for param in optics_params:
                assert diff_rms[param] < RMS_TOL_DICT[param], (
                    f"RMS for {param} in iteration {iter_step} larger than tolerance: "
                    f"{diff_rms[param]} >= {RMS_TOL_DICT[param]}."
                    )

            # assert total RMS decreases between steps
            # ('skew' is converged after one step, still works with seed 2234)
            assert sum(diff_rms_prev.values()) > sum(diff_rms.values()), (
                f"Total RMS in iteration {iter_step} larger than in previous iteration."
                f"{sum(diff_rms.values())} >= {sum(diff_rms_prev.values())}."
            )

        diff_rms_prev = diff_rms


# Helper -----------------------------------------------------------------------


def _create_fake_measurement(tmp_path, model_path, twiss_path, error_val, optics_params, seed):
    model_df = tfs.read(model_path / "twiss.dat", index=NAME)
    model_df = add_coupling_to_model(model_df)

    twiss_df = tfs.read(twiss_path, index=NAME)
    twiss_df = add_coupling_to_model(twiss_df)

    # create fake measurement data
    fake_measurement(
        model=model_df,
        twiss=twiss_df,
        randomize=[VALUES, ERRORS],
        relative_errors=[error_val],
        seed=seed,
        outputdir=tmp_path,
    )

    # load the fake data into a dict
    _, meas_dict = get_measurement_data(
        optics_params,
        meas_dir=tmp_path,
        beta_file_name='beta_phase_',
    )

    # map to VALUE, ERROR and WEIGHT, similar to filter_measurement
    # but without the filtering
    for col, meas in meas_dict.items():
        if col[:-1] in (F1010, F1001):
            col = {c[0]: c for c in (REAL, IMAG, PHASE, AMPLITUDE)}[col[-1]]

        if col != TUNE:
            meas[VALUE] = meas.loc[:, col].to_numpy()
            meas[ERROR] = meas.loc[:, f"{ERR}{col}"].to_numpy()
        meas[WEIGHT] = 1.
    return twiss_df, model_df, meas_dict
