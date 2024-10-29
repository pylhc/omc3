"""
Dpp
---

This module contains deltap over p calculations related functionality of ``optics_measurements``.
It provides functions to computes and arrange dp over p.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Sequence
import logging

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import tfs
    from generic_parser import DotDict
    from omc3.optics_measurements.data_models import InputFiles

DPP_BIN_TOLERANCE: float = 1e-4
DPP_TOLERANCE: float = 1e-5  # not sure if these should be different! (jdilly)

LOGGER = logging.getLogger(__name__)


def arrange_dpps(dpps: Sequence[float], tolerance: float = DPP_BIN_TOLERANCE):
    """
    Grouping of dpp-values and averaging them in the bins, also zeroes the bin closest to zero.
    """
    closest_to_zero = np.argmin(np.abs(dpps))
    ranges = _compute_ranges(dpps, tolerance)
    zero_offset = np.mean(_values_in_range(_find_range_with_element(ranges, closest_to_zero), dpps))
    LOGGER.debug(f"dp/p closest to zero is {dpps[closest_to_zero]}")
    if np.abs(zero_offset) > tolerance:
        LOGGER.warning(f"Analysed files have large momentum deviation {zero_offset}. "
                       f"Optics parameters might be wrong.")
    LOGGER.debug(f"Detected dpp differences, aranging as: {ranges}, zero offset: {zero_offset}.")
    arranged_dpps = []
    for idx in range(len(dpps)):
        dpps_from_range = _values_in_range(_find_range_with_element(ranges, idx), dpps)
        arranged_dpps.append(np.mean(dpps_from_range) - zero_offset)
    return np.array(arranged_dpps)


def _values_in_range(range_to_use, dpp_values):
    return [dpp_values[idx] for idx in range_to_use]


def _find_range_with_element(ranges, element):
    return [dpp_range for dpp_range in ranges if element in dpp_range][0]


def _compute_ranges(dpps: Sequence[float], tolerance: float) -> list[list[int]]:
    """ Groups the indices of dpps in bins of tolerance. 
    The function first sorts the indices by their dpp value 
    adds a new group whenever the dpp of the next index is larger than the first value in the group.

    Works for now, but we could improve by using the mean dpp of the group to check against,
    i.e. `abs(np.mean(dpps[ranges[-1]]) - dpps[idx]) < tolerance`.
    Only neccessary if we have some weird outliers. (jdilly)
    """
    ordered_ids = np.argsort(dpps)
    ranges = [[ordered_ids[0]]]
    if len(ordered_ids) == 1:
        return ranges
    for idx in ordered_ids[1:]:
        if abs(dpps[ranges[-1][0]] - dpps[idx]) < tolerance:
            ranges[-1].append(idx)
        else:
            ranges.append([idx])
    return ranges


def append_amp_dpp(list_of_tfs: Sequence[tfs.TfsFile], dpp_values: Sequence[float]):
    """ Add the dpp values to the DPP-header of the tfs files, if larger than the DPP-tolerance, otherwise set to zero.
    This is intended to the DPP value for on-momentum files to zero. """
    for i, dpp in enumerate(dpp_values):
        list_of_tfs[i].headers["DPPAMP"] = dpp if (not np.isnan(dpp) and np.abs(dpp) > DPP_TOLERANCE) else 0.0
    return list_of_tfs


def append_dpp(list_of_tfs: Sequence[tfs.TfsFile], dpp_values: Sequence[float]):
    """ Add the dpp values to the DPP-header of the tfs files. """
    for i, dpp in enumerate(dpp_values):
        list_of_tfs[i].headers["DPP"] = dpp
    return list_of_tfs


def calculate_dpoverp(input_files: InputFiles, meas_input: DotDict): 
    df_orbit = pd.DataFrame(meas_input.accelerator.model).loc[:, ['S', 'DX']]
    df_orbit = pd.merge(df_orbit, input_files.joined_frame('X', ['CO', 'CORMS']), how='inner',
                        left_index=True, right_index=True)
    mask = meas_input.accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
    df_filtered = df_orbit.loc[mask, :]
    dispersions = df_filtered.loc[:, "DX"].to_numpy()
    denom = np.sum(dispersions ** 2)
    if denom == 0.:
        raise ValueError("Cannot compute dpp probably no arc BPMs.")
    amps = input_files.get_data(df_filtered, "CO")
    if amps.ndim == 1:
        return np.sum(dispersions * amps) / denom
    else:
        numer = np.sum(dispersions[:, None] * input_files.get_data(df_filtered, "CO"), axis=0)
    return numer / denom


def calculate_amp_dpoverp(input_files: InputFiles, meas_input: DotDict):
    df_orbit = pd.DataFrame(meas_input.accelerator.model).loc[:, ['S', 'DX']]
    df_orbit = pd.merge(df_orbit, input_files.joined_frame('X', ['AMPX', 'AMPZ']), how='inner',
                        left_index=True, right_index=True)
    # TODO often missing AMPZ causes FutureWarning in future will be KeyError
    if np.all(np.isnan(input_files.get_data(df_orbit, "AMPZ"))):
        return np.zeros(len(input_files["X"]))
    mask = meas_input.accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
    df_filtered = df_orbit.loc[mask, :]
    amps = input_files.get_data(df_filtered, 'AMPX') * input_files.get_data(df_filtered, 'AMPZ')
    mask_zeros = (amps > 0) if amps.ndim == 1 else (np.sum(amps, axis=1) > 0)
    dispersions = df_filtered.loc[mask_zeros, "DX"].to_numpy()
    denom = np.sum(dispersions ** 2)
    if denom == 0.:
        raise ValueError("Cannot compute dpp probably no arc BPMs.")
    if amps.ndim == 1:
        numer = np.sum(dispersions * amps[mask_zeros])
    else:
        numer = np.sum(dispersions[:, None] * amps[mask_zeros, :], axis=0)
    return numer / denom


