from __future__ import annotations

import re
from typing import Any, TypeAlias
from collections.abc import Sequence
import numpy as np
import pandas as pd
import tfs
from omc3.definitions.constants import PLANES, UNIT_IN_METERS
from omc3.harpy import clean
from omc3.utils import logging_tools
from omc3.optics_measurements.constants import BETA, PHASE_ADV, S
import turn_by_turn as tbt
from numpy._typing._array_like import _DualArrayLike
from omc3.utils import outliers


LOG = logging_tools.get_logger(__name__)

LENGTH: str = "LENGTH"
QX: str = "Q1"
QY: str = "Q2"

AVERAGED: str = "AV"
MAX: str = "MAX"
MIN: str = "MIN"
NORMALIZED: str = "NORM"

ArrayLikeStr: TypeAlias = _DualArrayLike[
        np.dtype[Any],
        str,
    ]

ArrayLikeNumber: TypeAlias = _DualArrayLike[
        np.dtype[Any],
        bool | int | float | complex,
    ]

ArrayLikeBool: TypeAlias = _DualArrayLike[
        np.dtype[Any],
        bool,
    ]


BoolOrSeries: TypeAlias = bool | pd.Series[bool]

class ActionPhaseData:
    
    def __init__(self):
        pass

    def use_bpm(self, name: ArrayLikeStr) -> BoolOrSeries:
        """ Function to check if the given elements should be used for action and phase calculations. """
        raise NotImplementedError


class LHCActionPhaseData(ActionPhaseData):
    
    def __init__(self):
        pass


    def use_bpm(self, name: ArrayLikeStr) -> BoolOrSeries:
        """ In the LHC only even-numbered BPMs are used."""
        pattern = r"B.+\.(\d+)[LR]\d\."

        if isinstance(name, str):
            match = re.match(pattern, name)
            if match is None:
                LOG.warning(f"Could not parse BPM name '{name}'")
                return False

            return int(match.group(1)) % 2 == 0
        
        return name.str.extract(pattern).astype(int) % 2 == 0


def cycle_model(model: tfs.TfsDataFrame, start_element: str) -> tfs.TfsDataFrame:
    """ Cycles the model so that the first element is the given start_element. 

    Args:
        model (tfs.TfsDataFrame): The model to cycle.
        start_element (str): The name of the element to start with.

    Returns:
        tfs.TfsDataFrame: The cycled model.
    """
    if start_element not in model.index:
        raise ValueError(f"Start element '{start_element}' not in model.")
    
    if start_element == model.index[0]:
        LOG.debug("Start element is already first element. Returning model.")

    LOG.debug(f"Cycling model so that '{start_element}' is first element.")
    cycled_model = np.roll(model, -model.index.get_loc(start_element), axis=0)
    cycled_model = tfs.TfsDataFrame(cycled_model, headers=model.headers)

    start_values = cycled_model.loc[start_element, :]

    cycled_model[S] = (cycled_model[S] - start_values[S]) % cycled_model.headers[LENGTH]
    for q, plane in zip((QX, QY), PLANES):
        cycled_model[f"{PHASE_ADV}{plane}"] = (
            cycled_model[f"{PHASE_ADV}{plane}"] - start_values[f"{PHASE_ADV}{plane}"]
        ) % cycled_model.headers[q]
    return cycled_model


def get_action_phase_from_two_bpms(
    df_data: pd.DataFrame, 
    plane: str, 
    bpm1: ArrayLikeStr, 
    bpm2: ArrayLikeStr,
    ) -> tuple[ArrayLikeNumber, ArrayLikeNumber]:
    """ Returns the action and phase from orbit, beta and phase of two BPMs, assuming no magnetic fields in between them.
    
    Implementation of Eq. 13 and Eq. 14 in [CardonaLinearNonlinearMagnetic2009]_ 
    with the missing square-roots over the betas as explained in [CardonaErratumLinearNonlinear2010]_ .

    Args:
        df_data (pd.DataFrame): Dataframe with BPMs and their beta, phase and orbit.
        bpm1 (ArrayLikeStr): First BPM name.
        bpm2 (ArrayLikeStr): Second BPM name.
        plane (str): 'X' or 'Y'.

    Returns:
        A tuple of floats or arrays of floats (depending on input) with the action and phase.

    """
    z1 = df_data.loc[bpm1, plane] / np.sqrt(df_data.loc[bpm1, f"{BETA}{plane}"])
    z2 = df_data.loc[bpm2, plane] / np.sqrt(df_data.loc[bpm2, f"{BETA}{plane}"])
    psi1 = 2 * np.pi * df_data.loc[bpm1, f"{PHASE_ADV}{plane}"]
    psi2 = 2 * np.pi * df_data.loc[bpm2, f"{PHASE_ADV}{plane}"]

    # Equation (13)
    action = (0.5*(z1**2 + z2**2) - z1 * z2 * np.cos(psi2 - psi1)) / (np.sin(psi2 - psi1)**2)

    # # Equation (14)
    # sin_diff = (z1 * np.sin(psi2) - z2 * np.sin(psi1))
    # cos_diff = (z1 * np.cos(psi2) - z2 * np.cos(psi1)) 
    # phase = np.arctan(sin_diff/cos_diff)

    # Equation (14) as implemented by J. Cardona in code. 
    # Equivalent, but with well defined phase sign.
    denom = np.sin(psi1) * np.cos(psi2) - np.cos(psi1) * np.sin(psi2)
    sin_delta = (z1 * np.sin(psi2) - z2 * np.sin(psi1)) / denom
    cos_delta = (z1 * np.cos(psi2) - z2 * np.cos(psi1)) / denom
    phase = np.arctan2(sin_delta, cos_delta)
    return action, phase
    

def remove_closed_orbit(tbt_data: tbt.TbtData) -> tbt.TbtData:
    """ Removes the average orbit per element, i.e. orbit offset/closed orbit, 
    from the tbt-data.  
    This is returned as a new TbtData object.

    Args:
        tbt_data (tbt.TbtData): TbtData to remove the average orbit from.

    Returns:
        tbt.TbtData: New TbtData instance with the average orbit removed.
     """
    matrices = [ 
        tbt.TransverseData(
            X=m.X - m.X.mean(axis=1), 
            Y=m.Y - m.Y.mean(axis=1)
        ) for m in tbt_data.matrices
    ] 
    return tbt.TbtData(
        matrices=matrices, 
        date=tbt_data.date, 
        bunch_ids=tbt_data.bunch_ids, 
        nturns=tbt_data.nturns
    )


def shift_and_scale_tbt_data(
    tbt_data: tbt.TbtData, 
    model: tfs.TfsDataFrame, 
    opposite_direction: bool,
    first_bpm: str = None, 
    ac_dipole: str = None, 
    unit: str = "m",
    ) -> tbt.TbtData:
    """ Shift the TBT data to start at the same point as the model.
    If the ac-dipole element is given, the model is shifted to start at the AC-Dipole location first.
    The data is returned as a new TbtData object.

    Args:
        tbt_data (tbt.TbtData): TBT data to shift.
        model (tfs.TfsDataFrame): Twiss model to use for the shift.
        opposite_direction (bool): Indiciates if the beam in the machine is circulating in the 
                                   opposite direction from the direction given in the model.
        first_bpm (str): Name of the first BPM after injection.
        ac_dipole (str): Name of the AC-dipole element.
        unit (str): Unit of the data.

    Returns:
        tbt.TbtData: New TbtData instance with the shifted data and its unit in meters.
    """
    LOG.debug("Shifting TBT data to start at the same point as the model.")
    if ac_dipole is not None:
        model = cycle_model(model, ac_dipole)

    matrices = [
        tbt.TransverseData(
            X = clean.resync_bpms(first_bpm, m.X, model, opposite_direction) * UNIT_IN_METERS[unit],
            Y = clean.resync_bpms(first_bpm, m.Y, model, opposite_direction) * UNIT_IN_METERS[unit],
        ) for m in tbt_data.matrices
    ]
    return tbt.TbtData(
        matrices=matrices, 
        date=tbt_data.date, 
        bunch_ids=tbt_data.bunch_ids, 
        nturns=tbt_data.nturns
    )
    

def get_average_orbit(tbt_data: tbt.TbtData, bpm: str, threshold: float = 0) -> pd.DataFrame:
    """ Return the average orbit, based on the phase at the given BPM.
    
    This is described in [CardonaLocalCorrectionQuadrupole2017]_ Section V B. 
    In particular Eq. 32 and Eq. 34 with Omega = pi/2 (if threshold = 0), i.e. all turns 
    with positive sign at the given BPM, as described.

    To make use of all available information, we also flip the sign for all BPMs of all 
    turns  with phases > pi/2, i.e. all turns with negative sign at the given BPM.
    They can then also be added to the average.

    The calculation is done per plane, resulting in a max-orbit at the plane with 
    positive orbit sign and a min-orbit in the other plane.

    Args:
        tbt_data (tbt.TbtData): TbtData to calculate the average orbit from.
        bpm (str): BPM name used as phase reference.
        threshold (float): Fraction of the max-value at the given BPM to include.
                           I.e. a threshold of 0 includes all data, a threshold of 
                           0.5 only data with absolute value higher than half of the maximum, etc.

    Returns:
        pd.DataFrame: With the average orbits per plane, separated into MIN and MAX orbits.

    """
    LOG.debug(f"Creating average orbit with reference BPM {bpm} and threshold {threshold}.")
    if len(tbt_data.matrices) != 1:
        raise NotImplementedError("Average orbit only implemented for TBT data with one bunch!")
    
    averaged_df = pd.DataFrame(index=tbt_data.matrices[0].X.index.union(tbt_data.matrices[0].Y.index))

    for plane in PLANES:
        data = getattr(tbt_data.matrices[0], plane)

        max_threshold = data.loc[bpm, :].max() * threshold
        min_threshold = data.loc[bpm, :].min() * threshold

        # Array with +1 positive amp > threshold, -1 negative amp > threshold, 0 for others
        sign_mask = (data.loc[bpm, :] > max_threshold) - (data.loc[bpm, :] < min_threshold)

        # The average orbit in this plane is its max-orbit
        averaged_df.loc[:, f"{AVERAGED}{MAX}{plane}"] = (sign_mask[None, :] * data).mean(axis=1)
        
        # The average orbit in the other plane is its min-orbit 
        other_plane = PLANES[PLANES.index(plane) + 1 % len(PLANES)]
        data_other_plane = getattr(tbt_data.matrices[0], other_plane)
        averaged_df.loc[:, f"{AVERAGED}{MIN}{other_plane}"] = (sign_mask[None, :] * data_other_plane).mean(axis=1)
    
    return averaged_df


def calculate_error_kick_strengths(
    action_before: float, 
    phase_before: float, 
    action_after: float, 
    phase_after: float, 
    beta: float, 
    phase_advance: float
    ) -> float:
    """ 
    Calculate the kick strengths of an error, given action and phase before and after the error location and beta at the error location.
    This is based on Eq. 3 and 4 in [CardonaLocalCorrectionQuadrupole2017]_

    Args:
        action_before (float): Action before the error location.
        phase_before (float): Phase (delta) before the error location.
        action_after (float): Action after the error location.
        phase_after (float): Phase (delta) after the error location.
        beta (float): Beta at the error location.
        phase_advance (float): Phase advance at the error location.
    
    Returns:
        float: Kick strength of the error.
    """
    # Calculate magnitude via Eq. 3
    numerator = (
        2 * action_before + 
        2 * action_after -
        4 * np.sqrt(action_before * action_after) * np.cos(phase_after - phase_before)
    )

    if numerator < 0:
        LOG.warning("Negative numerator in kick strength calculation! Assigning zero kick strength.")
        return 0

    magnitude = np.sqrt(numerator / beta)    
    
    # Calculate sign via Eq. 4    
    sign = (
        np.sqrt(2 * action_before / beta) * 
        np.sin(phase_before - phase_after) / 
        np.sin(phase_after - 2 * np.pi * phase_advance)
    )
    return np.sign(sign) * magnitude


def filter_data(data: Sequence[float], n_sigma: float = 2) -> Sequence[float]:
    """ Filter data within the n-sigma range around the mean. 
    Before sigma calculation, the data is also filtered for outliers.

    Args:
        data (Sequence[float]): Data to filter.
        n_sigma (float, optional): Number of sigmas to filter. Defaults to 2.

    Returns:
        Sequence[float]: Filtered data.
    """
    mask = outliers.get_filter_mask(data)
    mean, std = data[mask].mean(), data[mask].std()
    return data[(data < mean + n_sigma * std) & (data > mean - n_sigma * std)]


def estimate_orbit(
    model: tfs.TfsDataFrame, 
    location: ArrayLikeStr, 
    reference: ArrayLikeStr, 
    phase: float, 
    plane: str
    ) -> ArrayLikeNumber:
    """ Estimate the orbit at a certain location, given lattice functions and orbit close by.
    This is Eq. 14 in [CardonaCalibrationBeamPosition2021]_ , of which the simplified form is also 
    given as Eq. 10 in [CardonaLocalCorrectionQuadrupole2017]_ .

    Args:
        model (tfs.TfsDataFrame): TfsDataFrame with lattice functions.
        location (ArrayLikeStr): Location to estimate orbit at.
        reference (ArrayLikeStr): Location to use as reference orbit.
        phase (float): Phase (delta) at the location.
        plane (str): 'X' or 'Y'.

    Returns:
        ArrayLikeNumber: Estimated orbit.    

    """
    beta_ratio = (
        model.loc[location, f"{BETA}{plane}"] / 
        model.loc[reference, f"{BETA}{plane}"]
    )
    sin_ratio = (
        np.sin(2 * np.pi * model.loc[location, f"{PHASE_ADV}{plane}" - phase]) /
        np.sin(2 * np.pi * model.loc[reference, f"{PHASE_ADV}{plane}" - phase])
    )
    return model.loc[location, plane] * np.sqrt(beta_ratio) * sin_ratio


def name_tbd(orbit_data: pd.DataFrame, model: tfs.TfsDataFrame) -> pd.DataFrame:
    

    for plane in PLANES:
        data = model.loc[orbit_data.index, [S, f"{BETA}{plane}", f"{PHASE_ADV}{plane}"]]
        data.loc[:, plane] = orbit_data[f"{AVERAGED}{MAX}{plane}"]

        action, phase = get_action_phase_from_two_bpms(data, plane, data.index[:-1], data.index[1:])

        # sort into left and right BPMs, use only BPMs allowed (lhc = even bpms)
        # calculate action and phases
        # calculate average action and average phase on filtered data 



def action_phase_jumps(accel, beam, ip):
    pass
    
    
    # build average and average max tbt data
    #