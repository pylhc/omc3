"""
Module harpy.kicker
-------------------

Corrects phase of main spectral line in a case,
when damped (exponentially decaying) oscillations are analysed.
"""
import numpy as np
from utils import logging_tools

LOGGER = logging_tools.getLogger(__name__)


def phase_correction(bpm_data_orig, lin_frame, plane):
    """
    Corrects phase of main spectral line assuming exponentially decaying oscillations

    Args:
        bpm_data_orig: matrix of original TbtData
        lin_frame: DataFrame in which to correct the results
        plane: "X" or "Y"

    Returns:
        DataFrame with corrected phases
    """
    bpm_data = bpm_data_orig.loc[lin_frame.index, :]
    damp, dstd = _get_damping(bpm_data)
    LOGGER.debug(f"Damping factor X: {damp:2.2e} +- {dstd:2.2e}")
    int_range = np.arange(0.0, bpm_data.shape[1])
    amp = lin_frame.loc[:, 'PK2PK'].values / 2
    tune = lin_frame.loc[:, f"TUNE{plane}"].values * 2 * np.pi
    phase = lin_frame.loc[:, f"MU{plane}"].values * 2 * np.pi
    damp_range = damp * int_range
    phase_range = np.outer(tune, int_range - bpm_data.shape[1] / 2) + np.outer(phase, np.ones(bpm_data.shape[1]))

    e1 = np.sum(np.exp(2 * damp_range) * np.sin(2 * phase_range), axis=1) * amp/2
    e2 = np.sum(bpm_data * np.exp(damp_range) * np.sin(phase_range), axis=1)
    e3 = np.sum(bpm_data * np.exp(damp_range) * np.cos(phase_range), axis=1)
    e4 = np.sum(np.exp(2 * damp_range) * np.cos(2 * phase_range), axis=1) * amp
    cor = (e1 - e2) / ((e3 - e4) * 2 * np.pi)
    lin_frame[f"MU{plane}"] = lin_frame.loc[:, f"MU{plane}"].values + cor
    return lin_frame


def _get_damping(bpm_data):
    coefs = np.polyfit(np.arange(bpm_data.shape[1]),
                       np.maximum.accumulate(np.log(np.abs(bpm_data[::-1]))).T, 1)
    return np.mean(coefs[0, :]), np.std(coefs[0, :])
