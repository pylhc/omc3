"""
.. module: beta_from_amplitude

Created on 27/05/2013

:author: ?

Stores helper functions to compensate the AC dipole effect based on analytic formulae (R. Miyamoto).
"""

import numpy as np
from utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def get_lambda(driven_tune, free_tune):
    """
    Tunes are fractional in units of 2PI
    """
    return np.sin(np.pi * (driven_tune - free_tune)) / np.sin(np.pi * (driven_tune + free_tune))


def phase_ac2bpm(df_idx_by_bpms, driven_tune, free_tune, plane, accelerator):
    """Returns the necessary values for the exciter compensation.

    See: doi:10.1103/PhysRevSTAB.11.084002

    Args:
        df_idx_by_bpms (pandas.DataFrame): commonbpms (see GetLLM._get_commonbpms)
        driven_tune: Driven fractional tunes.
        free_tune: Natural fractional tunes.
        plane (char): X,Y
        accelerator: accelerator class instance.

    Returns tupel(a,b,c,d):
        a (string): name of the nearest BPM.
        b (float): compensated phase advance between the exciter and the nearest BPM.
        c (int): k of the nearest BPM.
        d (string): name of the exciter element.
    """
    model = accelerator.get_elements_tfs()
    r = get_lambda(driven_tune % 1.0, free_tune % 1.0)
    [k, bpmac1], exciter = accelerator.get_exciter_bpm(plane, df_idx_by_bpms.index)
    psi = model.loc[bpmac1, "MU" + plane] - model.loc[exciter, "MU" + plane]
    psi = np.arctan((1+r)/(1-r) * np.tan(2 * np.pi * psi + np.pi * free_tune)) % np.pi - np.pi * driven_tune
    psi = psi / (2 * np.pi)
    return bpmac1, psi, k, exciter


def get_kick_from_bpm_list_w_acdipole(model_ac, bpm_list, measurements, plane):
    """
    @author: F Carlier
    Function calculates kick from measurements with AC dipole using the amplitude of the main line.
    The main line amplitude is is normalized with the model beta-function.

    Input:
        bpm_list:     Can be any list of bpms. Preferably only arc bpms for kick calculations,
                        but other bad bpms may be included as well.
        measurements: List of measurements when analyzing multiple measurements at once
        plane:        Either X or Y
    Output:
        actions_sqrt:       array containing the actions of each measurement. Notice this is the
                            square root of the action, so sqrt(2J_x/y)
        actions_sqrt_err:   array containing the errors for sqrt(2J_x/y) for each measurement.
    """
    betmdl = model_ac.loc[bpm_list.index, "BET" + plane].values
    actions_sqrt, actions_sqrt_err = [], []
    for meas in measurements[plane]:
        amp = 2 * meas.loc[bpm_list.index, "AMP" + plane].values
        actions_sqrt.append(np.average(amp / np.sqrt(betmdl)))
        actions_sqrt_err.append(np.std(amp / np.sqrt(betmdl)))
    return np.array(actions_sqrt), np.array(actions_sqrt_err)
