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
from omc3.correction.handler import get_measurement_data, _maybe_add_coupling_to_model, _create_corrected_model
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
    # read data from files
    optics_params, meas_dict = get_measurement_data(
        opt.optics_params,
        opt.meas_dir,
        opt.beta_file_name,
    )

    corr_model_path = opt.output_dir / f"twiss_matched{EXT}"

    corr_model_elements = _create_corrected_model(corr_model_path, opt.change_params_path, accel_inst)
    corr_model_elements = _maybe_add_coupling_to_model(corr_model_elements, optics_params)

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

        writeparams(opt.change_params_path, delta, "! Values to match model to measurement.")
        writeparams(opt.change_params_correct_path, -delta, "! Values to correct the measurement.")
        LOG.debug(f"Cumulative delta: {np.sum(np.abs(delta.loc[:, DELTA].to_numpy())):.5e}")
    write_knob(opt.knob_path, delta)
    LOG.info("Finished Iterative Global Correction.")
