"""
Interaction Point
-----------------

This module contains IP properties calculations related functionality of ``optics_measurements``.
It provides functions to compute beta* from phase.

TODO:
    - Put columns into the constants

"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import tfs

from omc3.definitions.constants import PI2
from omc3.optics_measurements.constants import EXT, IP_NAME, S, MODEL, MEASUREMENT, ERR 
from omc3.utils import logging_tools

from typing import TYPE_CHECKING, Any 

if TYPE_CHECKING: 
    from generic_parser import DotDict 
    from omc3.optics_measurements import phase


LOGGER = logging_tools.get_logger(__name__)
COLUMNS = ("IP", "BETASTAR", "ERRBETASTAR", "PHASEADV", "ERRPHASEADV", "PHASEDVMDL", "LSTAR")


def betastar_from_phase(meas_input: DotDict, phase_d: phase.PhaseDict) -> pd.DataFrame:
    """
    Calculate beta* and l* from the phase advance of the IP-BPMs.

    Arguments:
        meas_input: `Measurement_input` object.
        phase_d: PhaseDict output of the phasecalculation.

    Returns:
        A DataFrame with the beta* and l* as well as the phases used for the calculation
        as columns and the IP names as index.
    """
    accel = meas_input.accelerator
    model = accel.model
    try:
        ips = list(accel.get_ips())
    except AttributeError:
        LOGGER.debug(f"Accelerator {type(accel).__name__} has not get_ips method.")
        return None
    rows = []
    for ip_name, bpml, bpmr in ips:
        try:
            phaseadv, ephaseadv, mdlphadv = _get_meas_phase(bpml, bpmr, phase_d)
        except KeyError:
            LOGGER.debug(f"{bpml} {bpmr} not in phases dataframe.")
            continue  # Measurement on one of the BPMs not present.
        lstar = _get_lstar(bpml, bpmr, model)
        betastar, ebestar = phase_to_betastar(lstar, PI2 * phaseadv, PI2 * ephaseadv)
        rows.append([ip_name, betastar, ebestar, phaseadv, ephaseadv, mdlphadv, lstar])
    return pd.DataFrame(columns=COLUMNS, data=rows)


def write(df_ips: pd.DataFrame, headers: dict[str, Any], output_dir: str|Path, plane: str):
    """ Write the interaction point data to disk. 
    Empty DataFrames are skipped on write.
    
    Args:
        df_ips (pd.DataFrame): The interaction point data.
        headers (dict[str, Any]): The headers for the tfs file.
        output_dir (str|Path): The path to the output directory.
        plane (str): The plane of the interaction point data.
    """
    if df_ips is not None:
        output_path = Path(output_dir) / f"{IP_NAME}{plane.lower()}{EXT}"
        tfs.write(output_path, df_ips, headers_dict=headers)


def phase_to_betastar(lstar: float, phase: float, errphase: float) -> tuple[float, float]:
    """
    Return the betastar and its error given the phase advance across the IP.

    This function computes the betastar using the phase advance between the BPMs around the IP
    and their distance (lstar). The phase and error in the phase must be given in radians.

    Args:
        lstar (float): The distance between the BPMs and the IR.
        phase (float): The phase advance between the BPMs at each side of the IP. Must be given in radians.
        errphase (float): The error in phase. Must be given in radians.

    Returns:
        tuple[float, float]: betastar and its error.
    """
    return (
        _phase_to_betastar_value(lstar, phase),
        _phase_to_betastar_error(lstar, phase, errphase),
    )


def _phase_to_betastar_value(lstar, phase):
    tan_phase = np.tan(phase)
    return (lstar * (1 - np.sqrt(tan_phase ** 2 + 1))) / tan_phase


def _phase_to_betastar_error(lstar: float, phase: float, errphase: float):
    return abs((errphase * lstar * (abs(np.cos(phase)) - 1)) / (np.sin(phase) ** 2))


def _get_meas_phase(bpm_left, bpm_right, phases_df):
    return (
        phases_df[MEASUREMENT].loc[bpm_left, bpm_right],
        phases_df[f"{ERR}{MEASUREMENT}"].loc[bpm_left, bpm_right],
        phases_df[MODEL].loc[bpm_left, bpm_right],
    )


def _get_lstar(bpm_left, bpm_right, model):
    return abs(model.loc[bpm_left, S] - model.loc[bpm_right, S]) / 2.
