"""
Kicker
------

This module contains phase correction functionality of ``harpy``.
It provides tools to correct phases of a main spectral line in a case where damped (exponentially
decaying) oscillations are analysed.
"""
import numpy as np

from omc3.utils import logging_tools
from omc3.harpy.constants import COL_TUNE, COL_MU
from omc3.definitions.constants import PI2

LOGGER = logging_tools.getLogger(__name__)


def phase_correction(bpm_data_orig, lin_frame, plane):
    """
    Corrects phase of main spectral line assuming exponentially decaying oscillations.

    Args:
        bpm_data_orig: matrix of original `TbtData`.
        lin_frame: DataFrame in which to correct the results.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        A `DataFrame` with corrected phases.
    """
    bpm_data = bpm_data_orig.loc[lin_frame.index, :]
    damp, dstd = _get_damping(bpm_data)
    LOGGER.debug(f"Damping factor X: {damp:2.2e} +- {dstd:2.2e}")
    int_range = np.arange(0.0, bpm_data.shape[1])
    amp = lin_frame.loc[:, 'PK2PK'].to_numpy() / 2
    tune = lin_frame.loc[:, f"{COL_TUNE}{plane}"].to_numpy() * PI2
    phase = lin_frame.loc[:, f"{COL_MU}{plane}"].to_numpy() * PI2
    damp_range = damp * int_range
    phase_range = np.outer(tune, int_range - bpm_data.shape[1] / 2) + np.outer(phase, np.ones(bpm_data.shape[1]))

    e1 = np.sum(np.exp(2 * damp_range) * np.sin(2 * phase_range), axis=1) * amp/2
    e2 = np.sum(bpm_data * np.exp(damp_range) * np.sin(phase_range), axis=1)
    e3 = np.sum(bpm_data * np.exp(damp_range) * np.cos(phase_range), axis=1)
    e4 = np.sum(np.exp(2 * damp_range) * np.cos(2 * phase_range), axis=1) * amp
    cor = (e1 - e2) / ((e3 - e4) * PI2)
    lin_frame[f"{COL_MU}{plane}"] = lin_frame.loc[:, f"{COL_MU}{plane}"].to_numpy() + cor
    return lin_frame


def _get_damping(bpm_data):
    coefs = np.polyfit(np.arange(bpm_data.shape[1]),
                       np.maximum.accumulate(np.log(np.abs(bpm_data[::-1]))).T, 1)
    return np.mean(coefs[0, :]), np.std(coefs[0, :])
