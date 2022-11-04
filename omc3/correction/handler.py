"""
Handler
-------

This module contains high-level functions to manage most functionality of the corrections calculations.
"""
import datetime
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tfs
from generic_parser import DotDict
from sklearn.linear_model import OrthogonalMatchingPursuit

import omc3.madx_wrapper as madx_wrapper
from omc3.correction import filters, model_appenders, response_twiss
from omc3.correction.constants import (BETA, DELTA, DIFF, DISP, ERROR, F1001,
                                       F1010, NORM_DISP, PHASE, TUNE,
                                       VALUE, WEIGHT)
from omc3.correction.model_appenders import add_coupling_to_model
from omc3.correction.response_io import read_fullresponse
from omc3.model.accelerators.accelerator import Accelerator
from omc3.optics_measurements.constants import (DISPERSION_NAME, EXT, REAL, IMAG,
                                                NORM_DISP_NAME, PHASE_NAME, NAME)
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


def correct(accel_inst: Accelerator, opt: DotDict) -> None:
    """ Perform global correction as described in :mod:`omc3.global_correction`.

    Args:
        accel_inst (Accelerator): Accelerator Instance
        opt (DotDict): Correction options,
                       see :mod:`omc3.global_correction` for details.

    """
    method_options = opt.get_subdict(["svd_cut", "n_correctors"])
    # read data from files
    vars_list = _get_varlist(accel_inst, opt.variable_categories)
    optics_params, meas_dict = get_measurement_data(
        opt.optics_params,
        opt.meas_dir,
        opt.beta_file_name,
        opt.weights,
    )

    if opt.fullresponse_path is not None:
        resp_dict = _load_fullresponse(opt.fullresponse_path, vars_list)
    else:
        resp_dict = response_twiss.create_response(accel_inst, opt.variable_categories, optics_params)

    # the model in accel_inst is modified later, so save nominal model here to variables
    nominal_model = _maybe_add_coupling_to_model(accel_inst.model, optics_params)
    # apply filters to data
    meas_dict = filters.filter_measurement(optics_params, meas_dict, nominal_model, opt)
    meas_dict = model_appenders.add_differences_to_model_to_measurements(nominal_model, meas_dict)

    resp_dict = filters.filter_response_index(resp_dict, meas_dict, optics_params)
    resp_matrix = _join_responses(resp_dict, optics_params, vars_list)
    delta = tfs.TfsDataFrame(0., index=vars_list, columns=[DELTA])

    # ######### Iteration Phase ######### #
    for iteration in range(opt.max_iter + 1):
        LOG.info(f"Correction Iteration {iteration} of {opt.max_iter}.")

        # ######### Update Model and Response ######### #
        if iteration > 0:
            LOG.debug("Updating model via MADX.")
            corr_model_path = opt.output_dir / f"twiss_{iteration}{EXT}"

            corr_model_elements = _create_corrected_model(corr_model_path, opt.change_params_path, accel_inst)
            corr_model_elements = _maybe_add_coupling_to_model(corr_model_elements, optics_params)

            bpms_index_mask = accel_inst.get_element_types_mask(corr_model_elements.index, types=["bpm"])
            corr_model = corr_model_elements.loc[bpms_index_mask, :]

            meas_dict = model_appenders.add_differences_to_model_to_measurements(corr_model, meas_dict)

            if opt.update_response:
                LOG.debug("Updating response.")
                # please look away for the next two lines.
                accel_inst._model = corr_model
                accel_inst._elements = corr_model_elements
                resp_dict = response_twiss.create_response(accel_inst, opt.variable_categories, optics_params)
                resp_dict = filters.filter_response_index(resp_dict, meas_dict, optics_params)
                resp_matrix = _join_responses(resp_dict, optics_params, vars_list)

        # ######### Actual optimization ######### #
        delta += _calculate_delta(resp_matrix, meas_dict, optics_params, vars_list, opt.method, method_options)

        # remove unused correctors from vars_list
        delta, resp_matrix, vars_list = _filter_by_strength(delta, resp_matrix, opt.min_corrector_strength)

        writeparams(opt.change_params_path, delta)
        writeparams(opt.change_params_correct_path, -delta)
        LOG.debug(f"Cumulative delta: {np.sum(np.abs(delta.loc[:, DELTA].to_numpy())):.5e}")
    write_knob(opt.knob_path, delta)
    LOG.info("Finished Iterative Global Correction.")


# Input ------------------------------------------------------------------------


def read_measurement_file(meas_dir: Path, filename: str) -> tfs.TfsDataFrame:
    return tfs.read(meas_dir / filename, index="NAME")


def get_measurement_data(
        keys: Sequence[str],
        meas_dir: Path,
        beta_file_name: str,
        w_dict: Dict[str, float] = None,
) -> Tuple[List[str], Dict[str, tfs.TfsDataFrame]]:
    """ Loads all measurements defined by `keys` into a dictionary. """
    measurement = {}
    filtered_keys = keys
    if w_dict is not None:
        filtered_keys = [key for key in keys if w_dict[key] != 0]
        if not len(filtered_keys):
            raise ValueError(
                "All given Parameters have been discarded due to all-zero weights. "
                "Check given weights and weight default values."
            )

    for key in filtered_keys:
        if key.startswith(f"{PHASE}"):
            measurement[key] = read_measurement_file(meas_dir, f"{PHASE_NAME}{key[-1].lower()}{EXT}")

        elif key.startswith(f"{DISP}"):
            measurement[key] = read_measurement_file(meas_dir, f"{DISPERSION_NAME}{key[-1].lower()}{EXT}")

        elif key == f"{NORM_DISP}X":
            measurement[key] = read_measurement_file(meas_dir, f"{NORM_DISP_NAME}{key[-1].lower()}{EXT}")

        elif key[:5] in (F1010, F1001):
            measurement[key] = read_measurement_file(meas_dir, f"{key[:5].lower()}{EXT}")

        elif key == f"{TUNE}":
            measurement[key] = pd.DataFrame(
                {  # Just fractional tunes:
                    VALUE: np.remainder([read_measurement_file(meas_dir, f"{PHASE_NAME}x{EXT}")[f"{TUNE}1"],
                                         read_measurement_file(meas_dir, f"{PHASE_NAME}x{EXT}")[f"{TUNE}2"]],
                                        [1, 1]),
                    ERROR: np.array([0.001, 0.001])  # TODO measured errors not in the file
                },
                index=[f"{TUNE}1", f"{TUNE}2"],
            )

        elif key.startswith(f"{BETA}"):
            measurement[key] = read_measurement_file(meas_dir, f"{beta_file_name}{key[-1].lower()}{EXT}")
    return filtered_keys, measurement


def _load_fullresponse(full_response_path: Path, variables: Sequence[str]) -> dict:
    """
    Full response is dictionary of optics-parameter gradients upon
    a change of a single quadrupole strength
    """
    LOG.debug("Starting loading Full Response optics")
    full_response_data = read_fullresponse(full_response_path)

    # There is a check in read_fullresponse but there all variables need to be present.
    # Here only some. So I leave it like that (jdilly 2021-06-03)
    loaded_variables = [var for resp in full_response_data.values() for var in resp]
    if not any([var in loaded_variables for var in variables]):
        raise ValueError(
            "None of the given variables found in response matrix. Are you using the right categories?"
        )
    return full_response_data


# Data handling ----------------------------------------------------------------


def _get_varlist(accel_cls: Accelerator, variables: Sequence[str]):  # TODO: Virtual?
    varlist = np.array(accel_cls.get_variables(classes=variables))
    if len(varlist) == 0:
        raise ValueError("No variables found! Make sure your categories are valid!")
    return varlist


def _maybe_add_coupling_to_model(model: tfs.TfsDataFrame, keys: Sequence[str]) -> tfs.TfsDataFrame:
    """ Add coupling to the model, if terms corresponding to coupling RDTs are
    found in the provided keys.

    Args:
        model (tfs.TfsDataFrame): Twiss dataframe.
        keys (Sequence[str]):

    Returns:
        A TfsDataFrame with the added columns.
    """
    if any([key for key in keys if key.startswith("F1")]):
        return add_coupling_to_model(model)
    return model


def _create_corrected_model(twiss_out: str, change_params, accel_inst: Accelerator) -> tfs.TfsDataFrame:
    """ Use the calculated deltas in changeparameters.madx to create a corrected model """
    madx_script: str = accel_inst.get_update_correction_script(twiss_out, change_params)
    twiss_out_path = Path(twiss_out)
    madx_script = f"! Based on model '{accel_inst.model_dir}'\n" + madx_script
    madx_wrapper.run_string(
        madx_script,
        output_file=twiss_out_path.parent / f"job.create_{twiss_out_path.stem}.madx",
        log_file=twiss_out_path.parent / f"job.create_{twiss_out_path.stem}.log",
        cwd=accel_inst.model_dir,  # models are always run from there
    )
    return tfs.read(twiss_out, index=NAME)


def _join_responses(resp, keys, varslist):
    """ Returns matrix #BPMs * #Parameters x #variables """
    return (
        pd.concat(
            [resp[k] for k in keys],  # dataframes
            axis="index",  # axis to join along
            join="outer",  # =[pd.Index(varslist)]
            # other axes to use (pd Index obj required)
        ).reindex(columns=varslist).fillna(0.0)
    )


def _join_columns(col, meas, keys):
    """ Retuns vector: N= #BPMs * #Parameters (BBX, MUX etc.) """
    return np.concatenate([meas[key].loc[:, col].to_numpy() for key in keys], axis=0)


def _filter_by_strength(delta: pd.DataFrame, resp_matrix: pd.DataFrame, min_strength: float = 0):
    """ Remove too small correctors """
    delta = delta.loc[delta[DELTA].abs() > min_strength]
    return delta, resp_matrix.loc[:, delta.index], delta.index.to_numpy()


# Optimization -----------------------------------------------------------------


def _get_method_fun(method: str) -> Callable:
    method_to_function_dict = {"pinv": _pseudo_inverse, "omp": _orthogonal_matching_pursuit}
    return method_to_function_dict[method]


def _pseudo_inverse(response_mat: pd.DataFrame, diff_vec, opt: DotDict):
    """ Calculates the pseudo-inverse of the response via svd. (numpy) """
    if opt.svd_cut is None:
        raise ValueError("svd_cut setting needed for pseudo inverse method.")
    return np.dot(np.linalg.pinv(response_mat, opt.svd_cut), diff_vec)


def _orthogonal_matching_pursuit(response_mat: pd.DataFrame, diff_vec, opt: DotDict):
    """ Calculated n_correctors via orthogonal matching pursuit"""
    if opt.n_correctors is None:
        raise ValueError("n_correctors setting needed for orthogonal matching pursuit.")

    res = OrthogonalMatchingPursuit(opt.n_correctors).fit(response_mat, diff_vec)
    coef = res.coef_
    LOG.debug(f"Orthogonal Matching Pursuit Results: \n"
              f"  Chosen variables: {response_mat.columns.to_numpy()[coef.nonzero()]}\n"
              f"  Score: {res.score(response_mat, diff_vec)}")
    return coef


def _calculate_delta(
        resp_matrix: pd.DataFrame,
        meas_dict: dict,
        keys: Sequence[str],
        vars_list: Sequence[str],
        method: str,
        meth_opt
):
    """Get the deltas for the variables.

    Output is Dataframe with one column 'DELTA' and vars_list index."""
    weight_vector = _join_columns(f"{WEIGHT}", meas_dict, keys)
    diff_vector = _join_columns(f"{DIFF}", meas_dict, keys)

    resp_weighted = resp_matrix.mul(weight_vector, axis="index")
    diff_weighted = diff_vector * weight_vector

    delta = _get_method_fun(method)(resp_weighted, diff_weighted, meth_opt)
    delta = tfs.TfsDataFrame(delta, index=vars_list, columns=[DELTA])

    # check calculations
    update = np.dot(resp_weighted, delta[DELTA])
    _print_rms(meas_dict, diff_weighted, update)

    return delta


# Print ------------------------------------------------------------------------


def _print_rms(meas: dict, diff_w, r_delta_w) -> None:
    """ Prints current RMS status """
    f_str = "{:>20s} : {:.5e}"
    LOG.debug("RMS Measure - Model (before correction, w/o weigths):")
    for key in meas:
        LOG.debug(f_str.format(key, _rms(meas[key].loc[:, DIFF].to_numpy())))

    LOG.info("RMS Measure - Model (before correction, w/ weigths):")
    for key in meas:
        LOG.info(f_str.format(key, _rms(meas[key].loc[:, DIFF].to_numpy() * meas[key].loc[:, WEIGHT].to_numpy())))

    LOG.info(f_str.format("All", _rms(diff_w)))
    LOG.debug(f_str.format("R * delta", _rms(r_delta_w)))
    LOG.debug("(Measure - Model) - (R * delta)   ")
    LOG.debug(f_str.format("", _rms(diff_w - r_delta_w)))


def _rms(a):
    """Calculate root mean square error of a number or an ndarray."""
    return np.sqrt(np.mean(np.square(a)))


# Output -----------------------------------------------------------------------


def write_knob(knob_path: Path, delta: pd.DataFrame) -> None:
    a = datetime.datetime.fromtimestamp(time.time())
    delta_out = -delta.loc[:, [DELTA]]
    delta_out.headers["PATH"] = str(knob_path.parent)
    delta_out.headers["DATE"] = str(a.ctime())
    tfs.write(knob_path, delta_out, save_index="NAME")


def writeparams(path_to_file: Path, delta: pd.DataFrame) -> None:
    with open(path_to_file, "w") as madx_script:
        for var in delta.index.to_numpy():
            value = delta.loc[var, DELTA]
            madx_script.write(f"{var} = {var} {value:+e};\n")
