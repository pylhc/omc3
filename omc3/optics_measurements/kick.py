"""
Kick
----

This module contains kick functionality of ``optics_measurements``.
It provides functions to compute kick actions.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs

from omc3.definitions.constants import PLANE_TO_NUM
from omc3.model.accelerators.accelerator import AccElementTypes
from omc3.optics_measurements.constants import (
    ACTION,
    AMPLITUDE,
    BETA,
    CLOSED_ORBIT,
    DPP,
    DPPAMP,
    ERR,
    EXT,
    KICK_NAME,
    NAT_TUNE,
    PEAK2PEAK,
    RES,
    RESCALE_FACTOR,
    RMS,
    SQRT_ACTION,
    TIME,
    TUNE,
    S,
)
from omc3.utils.stats import weighted_error, weighted_mean

if TYPE_CHECKING:
    from generic_parser import DotDict

    from omc3.optics_measurements.data_models import InputFiles


def calculate(
    measure_input: DotDict, input_files: InputFiles, scale: float, header_dict: dict, plane: str
) -> np.ndarray:
    """

    Args:
        measure_input: `OpticsInput` object.
        input_files: Stores the input files tfs.
        scale: measured beta functions.
        header_dict: `dict` containing information about the analysis.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        A 2D numpy array containing the sqrt(2J) and associated errors
        for the given plane.
    """
    try:
        kick_frame = _get_kick(measure_input, input_files, plane)
    except IndexError:  # occurs if either no x or no y files exist
        return pd.DataFrame
    kick_frame = _rescale_actions(kick_frame, scale, plane)
    header = _get_header(header_dict, plane, scale)
    tfs.write(
        Path(measure_input.outputdir) / f"{KICK_NAME}{plane.lower()}{EXT}",
        kick_frame,
        header,
    )
    return kick_frame.loc[:, [f"{SQRT_ACTION}{plane}", f"{ERR}{SQRT_ACTION}{plane}"]].to_numpy()


def _get_header(header_dict: dict, plane: str, scale: float) -> dict:
    header = header_dict.copy()
    header[f"{RESCALE_FACTOR}{plane}"] = scale
    return header


def _rescale_actions(df: pd.DataFrame, scaling_factor: float, plane: str) -> pd.DataFrame:
    """
    Apply computed rescaling factor to action columns, for the given plane.

    Note
    ----
    This function applies the scaling factor to the sqrt(2J) column and
    uses the result to compute the rescaled 2J column afterwards. In the
    past the same rescaling was applied to both columns which is incorrect.

    Args:
        df (pd.DataFrame): the kick file dataframe.
        scaling_factor (float): computed scaling to apply to the sqrt(2J) column.
        plane (str): the plane to apply for.

    Returns:
        A pandas.DataFrame with the relevant columns rescaled, consistently.
    """
    # Directly rescale sqrt(2J) and its error with the provided factor
    df[f"{SQRT_ACTION}{plane}{RES}"] = (df.loc[:, f"{SQRT_ACTION}{plane}"].to_numpy() * scaling_factor)
    df[f"{ERR}{SQRT_ACTION}{plane}{RES}"] = df.loc[:, f"{ERR}{SQRT_ACTION}{plane}"].to_numpy() * abs(scaling_factor)
    # Infer the rescaled 2J values from the rescaled sqrt(2J) values, same for errors
    df[f"{ACTION}{plane}{RES}"] = df.loc[:, f"{SQRT_ACTION}{plane}{RES}"].to_numpy() ** 2
    df[f"{ERR}{ACTION}{plane}{RES}"] = abs(2 * df.loc[:, f"{SQRT_ACTION}{plane}{RES}"].to_numpy() * df.loc[:, f"{ERR}{SQRT_ACTION}{plane}{RES}"].to_numpy())
    return df


def _get_kick(measure_input: DotDict, files: InputFiles, plane: str) -> pd.DataFrame:
    load_columns, calc_columns, column_types = _get_column_mapping(plane)
    kick_frame = pd.DataFrame(
        data=0.0, index=range(len(files[plane])), columns=list(column_types.keys())
    )
    kick_frame = kick_frame.astype(column_types)

    for i, df in enumerate(files[plane]):
        # load data directly from file
        for col, src in load_columns.items():
            with suppress(KeyError):
                kick_frame.loc[i, col] = df[src]

        # calculate data from measurement
        kick_frame.loc[i, calc_columns] = _get_action(measure_input, df, plane)
    return kick_frame.astype(column_types)


def _get_action(meas_input: DotDict, lin: pd.DataFrame, plane: str) -> np.ndarray:
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
    frame: pd.DataFrame = pd.merge(
        _get_model_arc_betas(meas_input, plane),
        lin,
        how="inner",
        left_index=True,
        right_index=True,
    )

    if meas_input.accelerator.excitation:
        amps: np.ndarray = frame.loc[:, f"{AMPLITUDE}{plane}"].to_numpy()
        try:  # only created when using cleaning in harpy
            err_amps: np.ndarray = frame.loc[:, f"{ERR}{AMPLITUDE}{plane}"].to_numpy()
        except KeyError:
            err_amps: np.ndarray = np.zeros_like(amps)
    else:
        amps: np.ndarray = frame.loc[:, PEAK2PEAK].to_numpy() / 2.0
        try:
            err_amps: np.ndarray = frame.loc[:, f"{CLOSED_ORBIT}{RMS}"].to_numpy()
        except KeyError:
            err_amps: np.ndarray = np.zeros_like(amps)

    # ----- Compute the sqrt(2J) and its error ----- #
    sqrt_beta: np.ndarray = np.sqrt(frame.loc[:, f"{BETA}{plane}"].to_numpy())

    actions_sqrt2j: np.ndarray = amps / sqrt_beta
    errors_sqrt2j: np.ndarray = err_amps / sqrt_beta

    mean_sqrt2j: np.ndarray = weighted_mean(data=actions_sqrt2j, errors=errors_sqrt2j)
    err_sqrt2j: np.ndarray = weighted_error(data=actions_sqrt2j, errors=errors_sqrt2j)

    # ----- Derive 2J from sqrt(2J) for consistency ----- #
    mean_2j: np.ndarray = mean_sqrt2j**2
    err_2j: np.ndarray = abs(2 * mean_sqrt2j * err_sqrt2j)

    return np.array([mean_sqrt2j, err_sqrt2j, mean_2j, err_2j])


def _get_model_arc_betas(measure_input: DotDict, plane: str) -> pd.DataFrame:
    accel = measure_input.accelerator
    model = accel.model_driven if accel.excitation else accel.model
    return model.loc[:, [S, f"{BETA}{plane}"]].loc[
        accel.get_element_types_mask(model.index, [AccElementTypes.ARC_BPMS]), :
    ]


def _get_column_mapping(plane: str) -> tuple[dict[str, str], list[str], dict[str, type]]:
    plane_number: int = PLANE_TO_NUM[plane]
    load_columns: dict[str, str] = {
        TIME: "TIME",
        DPP: DPP,
        DPPAMP: DPPAMP,
        f"{TUNE}{plane}": f"{TUNE}{plane_number}",
        f"{ERR}{TUNE}{plane}": f"{TUNE}{plane_number}{RMS}",
        f"{NAT_TUNE}{plane}": f"{NAT_TUNE}{plane_number}",
        f"{ERR}{NAT_TUNE}{plane}": f"{NAT_TUNE}{plane_number}{RMS}",
    }
    calc_columns: list[str] = [
        f"{SQRT_ACTION}{plane}",
        f"{ERR}{SQRT_ACTION}{plane}",
        f"{ACTION}{plane}",
        f"{ERR}{ACTION}{plane}",
    ]
    column_types: dict[str, type] = {TIME: str}
    column_types.update(dict.fromkeys(list(load_columns.keys())[1:] + calc_columns, float))
    return load_columns, calc_columns, column_types
