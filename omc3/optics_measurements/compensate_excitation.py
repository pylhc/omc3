"""
.. module: beta_from_amplitude

Created on 27/05/2013

:author: ?

Stores helper functions to compensate the AC dipole effect based on analytic formulae (R. Miyamoto).
"""

import numpy as np
from utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


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
