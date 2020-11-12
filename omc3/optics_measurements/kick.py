"""
Kick
----

This module contains kick functionality of ``optics_measurements``.
It provides functions to compute kick actions.
"""
from collections import OrderedDict
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
                                                SQRT_ACTION, TIME, TUNE, S)


def calculate(measure_input, input_files, scale, header_dict, plane):
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
        kick_frame = _getkick(measure_input, input_files, plane)
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


def _getkick(measure_input, files, plane):
    load_columns, calc_columns, column_types = _get_column_mapping(plane)
    kick_frame = pd.DataFrame(data=0.,
                              index=range(len(files[plane])),
                              columns=list(load_columns.keys()) + calc_columns)

    for i, df in enumerate(files[plane]):
        # load data directly from file
        for col, src in load_columns.items():
            with suppress(KeyError):
                kick_frame.loc[i, col] = df[src]

        # calculate data from measurement
        kick_frame.loc[i, calc_columns] = _gen_kick_calc(measure_input, df, plane)
    kick_frame = kick_frame.astype(column_types)
    return kick_frame


def _gen_kick_calc(meas_input, lin, plane):
    """
    Takes either PK2PK/2 for kicker excitation or AMP for AC-dipole excitation
    """
    frame = pd.merge(_get_model_arc_betas(meas_input, plane), lin.loc[:, [f"{AMPLITUDE}{plane}", PEAK2PEAK]],
                     how='inner', left_index=True, right_index=True)
    amps = (frame.loc[:, f"{AMPLITUDE}{plane}"].to_numpy() if meas_input.accelerator.excitation
            else frame.loc[:, PEAK2PEAK].to_numpy() / 2.0)
    meansqrt2j = amps / np.sqrt(frame.loc[:, f"{BETA}{plane}"].to_numpy())
    mean2j = np.square(amps) / frame.loc[:, f"{BETA}{plane}"].to_numpy()
    return np.array([np.mean(meansqrt2j), np.std(meansqrt2j), np.mean(mean2j), np.std(mean2j)])


def _get_model_arc_betas(measure_input, plane):
    accel = measure_input.accelerator
    model = (accel.model_driven if accel.excitation else accel.model)
    return model.loc[:, [S, f"{BETA}{plane}"]].loc[
           accel.get_element_types_mask(model.index, [AccElementTypes.ARC_BPMS]), :]


def _get_column_mapping(plane):
    plane_number = PLANE_TO_NUM[plane]
    load_columns = OrderedDict([
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
