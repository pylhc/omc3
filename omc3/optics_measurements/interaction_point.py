"""
.. module: interaction_point

Created on 13/06/18

:author: Jaime Coello de Portugal

It computes beta* from phase.
"""
from os.path import join
import numpy as np
import pandas as pd
import tfs
from utils import logging_tools


LOGGER = logging_tools.get_logger(__name__)
PI2 = 2 * np.pi
COLUMNS = ("IP", "BETASTAR", "EBETASTAR", "PHASEADV", "EPHASEADV",
           "MDLPHADV", "LSTAR")
PLANES = ("X", "Y")
MODE_TO_SUFFIX = {"D": "getIP{}.out",
                  "F": "getIP{}_free.out",
                  "F2": "getIP{}_free2.out"}
MODES = ("D", "F", "F2")


def betastar_from_phase(accel, phase_d, model):
    """Writes the getIP files with the betastar computed using phase advance.

    Arguments:
        accel: The accelerator class to be used.
        phase_d: The GetLLM phase_d object, output of calculate_phase.
        model: tfs instance with a model of the machine.
    Returns:
        A nested dict with the same structure as the phase_d dict.
    """
    try:
        ips = list(accel.get_ips())
    except AttributeError:
        LOGGER.debug("Accelerator {accel} has not get_ips method."
                     .format(accel=accel.__name__))
        return None
    ip_dict = dict(zip(PLANES, ({}, {})))
    for plane in PLANES:
        for mode in MODES:
            phases_df = phase_d[plane][mode]
            if phases_df is None:
                continue
            rows = []
            for ip_name, bpml, bpmr in ips:
                try:
                    phaseadv, ephaseadv, mdlphadv = _get_meas_phase(
                        bpml, bpmr, phases_df
                    )
                except KeyError:
                    LOGGER.debug("{0} {1} not in phases ({2}) dataframe."
                                 .format(bpml, bpmr, mode))
                    continue  # Measurement on one of the BPMs not present.
                lstar = _get_lstar(bpml, bpmr, model)
                betastar, ebestar = phase_to_betastar(
                    lstar, PI2 * phaseadv, PI2 * ephaseadv
                )
                rows.append([ip_name, betastar, ebestar,
                            phaseadv, ephaseadv, mdlphadv,
                            lstar])
            ip_dict[plane][mode] = pd.DataFrame(columns=COLUMNS, data=rows)
    return ip_dict


def write_betastar_from_phase(ips_d, headers, output_dir):
    """ TODO
    """
    assert not any(
        tfs.write(
            join(output_dir, MODE_TO_SUFFIX[mode].format(plane.lower())),
            ips_d[plane][mode],
            headers_dict=headers,
        )
        for plane in PLANES for mode in ips_d[plane]
    )


def phase_to_betastar(lstar, phase, errphase):
    """Return the betastar and its error given the phase advance across the IP.

    This function computes the betastar using the phase advance between the
    BPMs around the IP and their distance (lstar). The phase and error in the
    phase must be given in radians.

    Arguments:
        lstar: The distance between the BPMs and the IR.
        phase: The phase advance between the BPMs at each side of the IP. Must
            be given in radians.
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
    return abs(model.S[model.indx[bpml]] - model.S[model.indx[bpmr]]) / 2.
