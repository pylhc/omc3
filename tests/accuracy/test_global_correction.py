from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import tfs
from generic_parser.tools import DotDict

from omc3.correction.constants import ORBIT_DPP
from omc3.correction.model_appenders import add_coupling_to_model
from omc3.correction.model_diff import diff_twiss_parameters
from omc3.global_correction import CORRECTION_DEFAULTS
from omc3.global_correction import global_correction_entrypoint as global_correction
from omc3.model.constants import TWISS_DAT, TWISS_ELEMENTS_DAT
from omc3.optics_measurements.constants import (
    BETA,
    DELTA,
    DISPERSION,
    F1001,
    F1010,
    NAME,
    NORM_DISPERSION,
    PHASE,
    PHASE_ADV,
    TUNE,
)
from omc3.response_creator import create_response_entrypoint as create_response
from omc3.scripts.fake_measurement_from_model import ERRORS, VALUES
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from omc3.utils import logging_tools
from omc3.utils.stats import rms

LOG = logging_tools.get_logger(__name__)
# LOG = logging_tools.get_logger('__main__', level_console=logging_tools.MADX)

# Paths ---
INPUTS = Path(__file__).parent.parent / 'inputs'
CORRECTION_INPUTS = INPUTS / "correction"
CORRECTION_TEST_INPUTS = INPUTS / "correction_test"


RMS_TOL_DICT = {
    f"{PHASE}X": 0.002,
    f"{PHASE}Y": 0.002,
    f"{BETA}X": 0.01,
    f"{BETA}Y": 0.01,
    f"{DISPERSION}X": 0.0015,
    f"{DISPERSION}Y": 0.0015,
    f"{NORM_DISPERSION}X": 0.001,
    f"{TUNE}": 0.01,
    f"{F1001}R": 0.0015,
    f"{F1001}I": 0.0015,
    f"{F1010}R": 0.002,
    f"{F1010}I": 0.002,
}


# Relative Errors for fake measurement
RELATIVE_ERRORS = {
    f"{PHASE}X": 0.002,
    f"{PHASE}Y": 0.002,
    f"{BETA}X": 0.05,
    f"{BETA}Y": 0.05,
    f"{DISPERSION}X": 0.03,
    f"{DISPERSION}Y": 0.03,
    f"{NORM_DISPERSION}X": 0.05,
    f"{F1001}": 0.05,
    f"{F1010}": 0.05,
}



# Correction Input Parameters ---
@dataclass
class CorrectionParameters:
    twiss: Path
    correction_filename: Path
    fullresponse: str
    weights: Sequence[float] | None = None  # will be assigned default values in global_correction
    modelcut: Sequence[float] | None = None  # will be assigned default values in global_correction
    errorcut: Sequence[float] | None = None  # will be assigned default values in global_correction
    variables: Sequence[str] = tuple(CORRECTION_DEFAULTS["variable_categories"])
    optics_params: Sequence[str] = CORRECTION_DEFAULTS["optics_params"]
    arc_by_arc_phase: bool = False
    include_ips_in_arc_by_arc: str | None = None
    seed: int = 0



def get_skew_params(beam):
    return CorrectionParameters(
        twiss=CORRECTION_INPUTS / f"2018_inj_b{beam}_11m" / "twiss_skew_quadrupole_error.dat",
        correction_filename=CORRECTION_TEST_INPUTS / f"changeparameters_injb{beam}_skewquadrupole.madx",
        optics_params=[f"{F1001}R", f"{F1001}I", f"{F1010}R", f"{F1010}I"],
        weights=[1., 1., 1., 1.],
        variables=["MQSl"],
        fullresponse="fullresponse_MQSl.h5",
        seed=2234,  # iteration test might not work with other seeds (converges too fast)
    )


def get_normal_params(beam):
    return CorrectionParameters(
        twiss=CORRECTION_INPUTS / f"2018_inj_b{beam}_11m" / "twiss_quadrupole_error.dat",
        correction_filename=CORRECTION_TEST_INPUTS / f"changeparameters_injb{beam}_quadrupole.madx",
        optics_params=[f"{PHASE}X", f"{PHASE}Y", f"{BETA}X", f"{BETA}Y", f"{NORM_DISPERSION}X", TUNE],
        weights=[1., 1., 1., 1., 1., 1.],
        variables=["MQY_Q4"],
        fullresponse="fullresponse_MQY.h5",
        seed=12362,  # iteration test might not work with other seeds (converges too fast)
    )


def get_arc_by_arc_params(beam):
    return CorrectionParameters(
        twiss=CORRECTION_INPUTS / f"2018_inj_b{beam}_11m" / "twiss_mqt_quadrupole_error.dat",
        correction_filename=CORRECTION_TEST_INPUTS / f"changeparameters_injb{beam}_mqt_quadrupole.madx",
        optics_params=[f"{PHASE}X", f"{PHASE}Y"],
        modelcut=[1., 1.],  # no cut
        errorcut=[1., 1.],  # no cut
        weights=[1., 1.],
        variables=["kqtf.a23b", "kqtf.a34b", "kqtd.a67b", "kqtf.a78b"],  # only arcs with errors in the model
        fullresponse="fullresponse_MQT.h5",
        arc_by_arc_phase=True,
        seed=1267,
    )


# Tests -----------------------------------------------------------------------

@pytest.mark.basic
@pytest.mark.parametrize('correction_type', ('skew', 'normal'))
def test_lhc_global_correct(tmp_path: Path, model_inj_beams: DotDict, correction_type: Literal['skew', 'normal', 'arc_by_arc']):
    """Creates a fake measurement from a modfied model-twiss with (skew)
    quadrupole errors and runs global correction on this measurement.
    It is asserted that the resulting model approaches the modified twiss.
    In principle one could also check the last model, build from the final
    correction (as this correction is not plugged in to MAD-X again),
    but this is kind-of done with the correction test.
    Hint: the `model_inj_beam1` fixture is defined in `conftest.py`."""
    beam = model_inj_beams.beam
    param_map = {
        "skew": get_skew_params,
        "normal": get_normal_params,
        "arc_by_arc": get_arc_by_arc_params,  # created seperate test below for now
    }

    correction_params = param_map[correction_type](beam)
    iterations = 3   # '3' tests a single correction + one iteration, as the last (3rd) correction is not tested itself.

    # create and load model and twiss-with-errors
    model_df = tfs.read(model_inj_beams.model_dir / TWISS_DAT, index=NAME)
    model_df = add_coupling_to_model(model_df)

    twiss_errors_df = tfs.read(correction_params.twiss, index=NAME)
    twiss_errors_df = add_coupling_to_model(twiss_errors_df)

    # create fake measurement data
    params, errors = zip(
        *[(k, v) for k, v in RELATIVE_ERRORS.items()
        if k in correction_params.optics_params or f"{k}R" in correction_params.optics_params]
    )

    randomize = [VALUES, ERRORS]

    fake_measurement(
        model=model_df,
        twiss=twiss_errors_df,
        randomize=randomize,
        parameters=list(params),
        relative_errors=list(errors),
        seed=correction_params.seed,
        outputdir=tmp_path,
    )

    # Perform global correction
    global_correction(
        **model_inj_beams,
        # correction params
        meas_dir=tmp_path,
        output_dir=tmp_path,
        svd_cut=0.01,
        iterations=iterations,
        variable_categories=correction_params.variables,
        fullresponse_path=model_inj_beams.model_dir / correction_params.fullresponse,
        optics_params=correction_params.optics_params,
        weights=correction_params.weights,
        arc_by_arc_phase=correction_params.arc_by_arc_phase,
        modelcut=correction_params.modelcut,
        errorcut=correction_params.errorcut,
    )

    models = {  # gather models for plotting at the end (debugging)
        "model": tfs.read(model_inj_beams.model_dir / TWISS_ELEMENTS_DAT, index=NAME),
        "errors": twiss_errors_df,
    }

    # Test if corrected model is closer to model used to create measurement
    diff_rms_prev = None
    for iter_step in range(iterations):
        if iter_step == 0:
            model_iter_df = model_df
        else:
            model_iter_df = tfs.read(tmp_path / f"twiss_{iter_step}.tfs", index=NAME)
            model_iter_df = add_coupling_to_model(model_iter_df)
            models[f"iter{iter_step}"] = model_iter_df


        diff_df = diff_twiss_parameters(model_iter_df, twiss_errors_df, correction_params.optics_params)
        if TUNE in correction_params.optics_params:
            diff_df.headers[f"{DELTA}{TUNE}"] = np.array([diff_df[f"{DELTA}{TUNE}1"], diff_df[f"{DELTA}{TUNE}2"]])
        diff_rms = {param: rms(diff_df[f"{DELTA}{param}"] * weight)
                    for param, weight in zip(correction_params.optics_params, correction_params.weights)}

        ############ FOR DEBUGGING #############
        # Iteration 0 == fake uncorrected model
        # print()
        # print(f"ITERATION {iter_step}")
        # for param in correction_params.optics_params:
        #     print(f"{param}: {diff_rms[param]}")
        # print(f"Weighted Sum: {sum(diff_rms.values())}")
        # print()
        # continue
        # ########################################

        if diff_rms_prev is not None:
            # assert RMS after correction smaller than tolerances
            for param in correction_params.optics_params:
                tolerance = RMS_TOL_DICT[param]
                assert diff_rms[param] < tolerance, (
                    f"RMS for {param} in iteration {iter_step} larger than tolerance: "
                    f"{diff_rms[param]} >= {tolerance}."
                    )

            # assert total (weighted) RMS decreases between steps
            # ('skew' is converged after one step, still works with seed 2234)
            assert sum(diff_rms_prev.values()) > sum(diff_rms.values()), (
                f"Total RMS in iteration {iter_step} larger than in previous iteration."
                f"{sum(diff_rms.values())} >= {sum(diff_rms_prev.values())}."
            )

        diff_rms_prev = diff_rms



@pytest.mark.basic
@pytest.mark.parametrize('correction_type', ('arc_by_arc',))
def test_lhc_global_correct_arc_by_arc(tmp_path: Path, model_inj_beams: DotDict, correction_type: Literal['skew', 'normal', 'arc_by_arc']):
    """ This test is similar to `test_lhc_global_correct` above,
    but as the arc-by-arc correction does not work yet as intended (see https://github.com/pylhc/omc3/issues/480),
    checking the improvement of the phase-advance does not work very well.
    So no randomizing errors here and only one iteration step.
    And the used correctors are only the ones in the arcs with errors.

    TODO: (after #480 has been fixed)
     - Remove which correctors are used, should be automatic in the global correction
     - Compare RMS phase advance to model over all arcs (getting smaller and being below tolerance)
     - Maybe merge with `test_lhc_global_correct`.


    It is questionable, if this functionality SHOULD be used as is in the first place, though.
    (jdilly, Jan 2025)
    """
    beam = model_inj_beams.beam
    param_map = {
        "arc_by_arc": get_arc_by_arc_params,
    }

    correction_params = param_map[correction_type](beam)
    iterations = 2   # tests a single correction the last (2nd) correction is not tested itself.

    # create and load model and twiss-with-errors
    model_df = tfs.read(model_inj_beams.model_dir / TWISS_DAT, index=NAME)
    model_df = add_coupling_to_model(model_df)

    twiss_errors_df = tfs.read(correction_params.twiss, index=NAME)
    twiss_errors_df = add_coupling_to_model(twiss_errors_df)

    # create fake measurement data
    params, errors = zip(
        *[(k, v) for k, v in RELATIVE_ERRORS.items()
        if k in correction_params.optics_params or f"{k}R" in correction_params.optics_params]
    )

    randomize = []

    fake_measurement(
        model=model_df,
        twiss=twiss_errors_df,
        randomize=randomize,
        parameters=list(params),
        relative_errors=list(errors),
        seed=correction_params.seed,
        outputdir=tmp_path,
    )

    # Perform global correction
    global_correction(
        **model_inj_beams,
        # correction params
        meas_dir=tmp_path,
        output_dir=tmp_path,
        svd_cut=0.01,
        iterations=iterations,
        variable_categories=[f"{var}{beam}" for var in correction_params.variables],
        fullresponse_path=model_inj_beams.model_dir / correction_params.fullresponse,
        optics_params=correction_params.optics_params,
        weights=correction_params.weights,
        arc_by_arc_phase=correction_params.arc_by_arc_phase,
        modelcut=correction_params.modelcut,
        errorcut=correction_params.errorcut,
    )

    models = {  # gather models for plotting at the end (debugging)
        "model": tfs.read(model_inj_beams.model_dir / TWISS_ELEMENTS_DAT, index=NAME),
        "errors": twiss_errors_df,
    }

    # Test if corrected model is closer to model used to create measurement
    model_iter_df = tfs.read(tmp_path / "twiss_1.tfs", index=NAME)
    models['iter1'] = model_iter_df

    for plane in ("X", "Y"):
        index = twiss_errors_df.index.intersection(model_df.index)
        phase_column = f"{PHASE_ADV}{plane}"
        diff_mu_errors = twiss_errors_df.loc[index, phase_column] - model_df.loc[index, phase_column]
        diff_mu_iter = model_iter_df.loc[index, phase_column] - model_df.loc[index, phase_column]
        corrected_mu = diff_mu_errors - diff_mu_iter
        # for arc in ['12', '23', '34', '45', '56', '67', '78', '81']:
        for arc in ['23', '34', '67', '78']:  # check only arcs with errors for now
            bpm_start  = f'BPM.8R{arc[0]}.B{beam}'
            bpm_end  = f'BPM.8L{arc[1]}.B{beam}'
            abs_phase_diff_arc_after = abs(corrected_mu.loc[bpm_end] - corrected_mu.loc[bpm_start])
            abs_phase_diff_arc_before = abs(diff_mu_errors.loc[bpm_end] - diff_mu_errors.loc[bpm_start])

            assert abs_phase_diff_arc_after < abs_phase_diff_arc_before

            ############ FOR DEBUGGING #############
            # print(plane, arc, abs_phase_diff_arc_after < abs_phase_diff_arc_before, abs_phase_diff_arc_after, abs_phase_diff_arc_before)
            #########################################

    # ############ FOR DEBUGGING #############
    # # _plot_arc_by_arc(beam, **models)
    # #########################################


@pytest.mark.basic
@pytest.mark.parametrize("dpp", (2.5e-4, -1e-4))
def test_lhc_global_correct_dpp(tmp_path: Path, model_inj_beams: DotDict, dpp: float):
    response_path = tmp_path / "full_response_dpp.h5"
    beam = model_inj_beams.beam

    # Create response
    response_dict = create_response(
        outfile_path=response_path,
        variable_categories=[ORBIT_DPP, f"kq10.l1b{beam}", f"kq10.l2b{beam}"],
        delta_k=2e-5,
        **model_inj_beams,
    )

    # Verify response creation
    assert all(ORBIT_DPP in response_dict[key].columns for key in response_dict)

    # Create fake measurement
    dpp_path = CORRECTION_INPUTS / "deltap" / f"twiss_dpp_{dpp:.1e}_B{beam}.dat"
    model_df = tfs.read(dpp_path, index=NAME)
    fake_measurement(
        twiss=model_df,
        parameters=[f"{PHASE}X", f"{PHASE}Y"],
        outputdir=tmp_path,
    )

    # Test global correction with and without response update
    for update_response in [True, False]:
        previous_diff = np.inf
        for iteration in range(1, 4):
            global_correction(
                meas_dir=tmp_path,
                output_dir=tmp_path,
                fullresponse_path=response_path,
                variable_categories=[ORBIT_DPP, f"kq10.l1b{beam}"],
                optics_params=[f"{PHASE}X", f"{PHASE}Y"],
                iterations=iteration,
                update_response=update_response,
                **model_inj_beams,
            )
            result = tfs.read(tmp_path / "changeparameters_iter.tfs", index=NAME)
            current_dpp = -result[DELTA][ORBIT_DPP]

            # Check output accuracy
            rtol = 2e-2
            assert np.isclose(dpp, current_dpp, rtol=rtol), f"Expected {dpp}, got {current_dpp}, diff: {dpp - current_dpp}, iteration: {iteration}"

            # Check convergence
            # For some reason after updating the input files, I can expect a convergence off by 5e-3 instead of 1e-3 (jgray 2025)
            current_diff = np.abs(dpp - current_dpp) / np.abs(dpp)
            assert previous_diff > current_diff or np.isclose(
                previous_diff, current_diff, atol=5e-3
            ), (
                f"Convergence not reached, diff: {previous_diff} <= {current_diff}, iteration: {iteration}"
            )
            previous_diff = current_diff


def _plot_arc_by_arc(beam, **kwargs):
    """ Plot the arc-by-arc phase advance.

    Inputs should be data-frames with columns ``S``, ``MUX``, ``MUY``

    This function is here for debugging purposes.
    """
    from matplotlib import pyplot as plt
    from matplotlib import transforms

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    df_model = kwargs["model"]
    df_error = kwargs["errors"]
    df_iter1 = kwargs["iter1"]

    for ax, plane in zip(axs, ["X", "Y"]):
        for ip in df_model.index[df_model.index.str.startswith("IP")]:
            ax.axvline(df_model.loc[ip, "S"], color="k", linestyle="--")
            ax.text(
                df_model.loc[ip, "S"], 1.05, ip,
                transform=transforms.blended_transform_factory(ax.transData, ax.transAxes)
            )

        for name, df in kwargs.items():
            ax.plot(df["S"],df[f"MU{plane}"] - df_model.loc[df.index, f"MU{plane}"], label=name)

        ax.plot(df_error["S"],df_error[f"MU{plane}"] - df_iter1.loc[df_error.index, f"MU{plane}"], label='predicted corrected')

        ax.legend()
    plt.show()
