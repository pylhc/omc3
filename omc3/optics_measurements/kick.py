"""
Kick
----

This module contains kick functionality of ``optics_measurements``.
It provides functions to compute kick actions.
"""
from __future__ import annotations
from contextlib import suppress
from os.path import join

import numpy as np
import pandas as pd
import tfs

from omc3.definitions.constants import PLANE_TO_NUM
from omc3.model.accelerators.accelerator import AccElementTypes
from omc3.optics_measurements.constants import (ACTION, AMPLITUDE, BETA, DPP,
                                                DPPAMP, ERR, EXT, KICK_NAME,
                                                NAT_TUNE, PEAK2PEAK,
                                                RES,
                                                RESCALE_FACTOR, RMS,
                                                SQRT_ACTION, TIME, TUNE, S, CLOSED_ORBIT)
from omc3.utils.stats import weighted_mean, weighted_error

from typing import TYPE_CHECKING

if TYPE_CHECKING: 
    from generic_parser import DotDict
    from omc3.optics_measurements.data_models import InputFiles


def calculate(measure_input: DotDict, input_files: InputFiles, scale, header_dict, plane):
    """

    Args:
        measure_input: `OpticsInput` object.
        input_files: Stores the input files tfs.
        scale: measured beta functions.
        header_dict: `OrderedDict` containing information about the analysis.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        `TfsDataFrame` containing actions and their errors.
    """
    try:
        kick_frame = _get_kick(measure_input, input_files, plane)
    except IndexError:  # occurs if either no x or no y files exist
        return pd.DataFrame
    kick_frame = _rescale_actions(kick_frame, scale, plane)
    header = _get_header(header_dict, plane, scale)
    tfs.write(join(measure_input.outputdir, f"{KICK_NAME}{plane.lower()}{EXT}"), kick_frame, header)
    return kick_frame.loc[:, [f"{SQRT_ACTION}{plane}", f"{ERR}{SQRT_ACTION}{plane}"]].to_numpy()


def _get_header(header_dict, plane, scale):
    header = header_dict.copy()
    header[f"{RESCALE_FACTOR}{plane}"] = scale
    return header


def _rescale_actions(df, scaling_factor, plane):
    for col in (f"{SQRT_ACTION}{plane}", f"{ERR}{SQRT_ACTION}{plane}", f"{ACTION}{plane}", f"{ERR}{ACTION}{plane}"):
        df[f"{col}{RES}"] = df.loc[:, col].to_numpy() * scaling_factor
    return df


def _get_kick(measure_input, files, plane):
    load_columns, calc_columns, column_types = _get_column_mapping(plane)
    kick_frame = pd.DataFrame(data=0.,
                              index=range(len(files[plane])),
                              columns=list(column_types.keys()))
    kick_frame = kick_frame.astype(column_types)

    for i, df in enumerate(files[plane]):
        # load data directly from file
        for col, src in load_columns.items():
            with suppress(KeyError):
                kick_frame.loc[i, col] = df[src]

        # calculate data from measurement
        kick_frame.loc[i, calc_columns] = _get_action(measure_input, df, plane)
    kick_frame = kick_frame.astype(column_types)
    return kick_frame


def _get_action(meas_input, lin: pd.DataFrame, plane: str) -> np.ndarray:
    """
    Calculates action (2J and sqrt(2J)) and its errors from BPM data in lin-df.
    Takes either PK2PK/2 for kicker excitation or AMP for AC-dipole excitation,
    as the amplitude of the oscillations for single kicks falls off over turns,
    and hence the amplitude of the main line does not represent the initial kick,
    whereas it is constant for the driven excitation.
    Reminder: A = sqrt(2J \beta) .

    TODO (jdilly 07.09.2022):
          beta_phase instead of beta_model as stated below Eq. (11) in
          PHYS. REV. ACCEL. BEAMS 23, 042801 (2020)

    Returns:
        sqrt(2J), error sqrt(2J), 2J, error 2J as  (1x4) array
    """
    frame = pd.merge(_get_model_arc_betas(meas_input, plane), lin,
                     how='inner', left_index=True, right_index=True)

    if meas_input.accelerator.excitation:
        amps = frame.loc[:, f"{AMPLITUDE}{plane}"].to_numpy()
        try:  # only created when using cleaning in harpy
            err_amps = frame.loc[:, f"{ERR}{AMPLITUDE}{plane}"].to_numpy()
        except KeyError:
            err_amps = np.zeros_like(amps)
    else:
        amps = frame.loc[:, PEAK2PEAK].to_numpy() / 2.0
        try:
            err_amps = frame.loc[:, f"{CLOSED_ORBIT}{RMS}"].to_numpy()
        except KeyError:
            err_amps = np.zeros_like(amps)

    # sqrt(2J) ---
    sqrt_beta = np.sqrt(frame.loc[:, f"{BETA}{plane}"].to_numpy())

    actions_sqrt2j = amps / sqrt_beta
    errors_sqrt2j = err_amps / sqrt_beta

    mean_sqrt2j = weighted_mean(data=actions_sqrt2j, errors=errors_sqrt2j)
    err_sqrt2j = weighted_error(data=actions_sqrt2j, errors=errors_sqrt2j)

    # 2J ---
    actions_2j = np.square(amps) / frame.loc[:, f"{BETA}{plane}"].to_numpy()
    errors_2j = 2 * amps * err_amps / frame.loc[:, f"{BETA}{plane}"].to_numpy()

    mean_2j = weighted_mean(data=actions_2j, errors=errors_2j)
    err_2j = weighted_error(data=actions_2j, errors=errors_2j)

    return np.array([mean_sqrt2j, err_sqrt2j, mean_2j, err_2j])


def _get_model_arc_betas(measure_input, plane):
    accel = measure_input.accelerator
    model = (accel.model_driven if accel.excitation else accel.model)
    return model.loc[:, [S, f"{BETA}{plane}"]].loc[
           accel.get_element_types_mask(model.index, [AccElementTypes.ARC_BPMS]), :]


def _get_column_mapping(plane):
    plane_number = PLANE_TO_NUM[plane]
    load_columns = dict([
        (TIME,                      "TIME"),
        (DPP,                       DPP),
        (DPPAMP,                    DPPAMP),
        (f"{TUNE}{plane}",          f"{TUNE}{plane_number}"),
        (f"{ERR}{TUNE}{plane}",     f"{TUNE}{plane_number}{RMS}"),
        (f"{NAT_TUNE}{plane}",      f"{NAT_TUNE}{plane_number}"),
        (f"{ERR}{NAT_TUNE}{plane}", f"{NAT_TUNE}{plane_number}{RMS}"),
    ])
    calc_columns = [f"{SQRT_ACTION}{plane}", f"{ERR}{SQRT_ACTION}{plane}",
                    f"{ACTION}{plane}", f"{ERR}{ACTION}{plane}"]

    column_types = {TIME: str}
    column_types.update({k: float for k in list(load_columns.keys())[1:] + calc_columns})
    return load_columns, calc_columns, column_types
