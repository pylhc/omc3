"""
Interaction Point
-----------------

This module contains IP properties calculations related functionality of ``optics_measurements``.
It provides functions to compute beta* from phase.
"""
from os.path import join

import numpy as np
import pandas as pd
import tfs

from omc3.definitions.constants import PI2
from omc3.optics_measurements.constants import EXT, IP_NAME, S
from omc3.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)
COLUMNS = ("IP", "BETASTAR", "ERRBETASTAR", "PHASEADV", "ERRPHASEADV", "PHASEDVMDL", "LSTAR")


def betastar_from_phase(meas_input, phase_d):
    """
    Writes the **getIP** files with the betastar computed using phase advance.

    Arguments:
        meas_input: `Measurement_input` object.
        phase_d: Output of calculation.

    Returns:
        A nested `dict` with the same structure as the ``phase_d`` `dict`.
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


def write(ips_d, headers, output_dir, plane):
    if ips_d is not None:
        tfs.write(join(output_dir, f"{IP_NAME}{plane.lower()}{EXT}"), ips_d, headers_dict=headers,)


def phase_to_betastar(lstar, phase, errphase):
    """
    Return the betastar and its error given the phase advance across the IP.

    This function computes the betastar using the phase advance between the BPMs around the IP
    and their distance (lstar). The phase and error in the phase must be given in radians.

    Args:
        lstar: The distance between the BPMs and the IR.
        phase: The phase advance between the BPMs at each side of the IP. Must be given in radians.
        errphase: The error in phase. Must be given in radians.
    """
    return (_phase_to_betastar_value(lstar, phase),
            _phase_to_betastar_error(lstar, phase, errphase))


def _phase_to_betastar_value(l, ph):
    tph = np.tan(ph)
    return (l * (1 - np.sqrt(tph ** 2 + 1))) / tph


def _phase_to_betastar_error(l, ph, eph):
    return abs((eph * l * (abs(np.cos(ph)) - 1)) / (np.sin(ph) ** 2))


def _get_meas_phase(bpml, bpmr, phases_df):
    return (phases_df["MEAS"].loc[bpml, bpmr],
            phases_df["ERRMEAS"].loc[bpml, bpmr],
            phases_df["MODEL"].loc[bpml, bpmr])


def _get_lstar(bpml, bpmr, model):
    return abs(model.loc[bpml, S] - model.loc[bpmr, S]) / 2.
