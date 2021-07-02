"""
Dpp
---

This module contains deltap over p calculations related functionality of ``optics_measurements``.
It provides functions to computes and arrange dp over p.
"""
import logging

import numpy as np
import pandas as pd

DPP_TOLERANCE = 1e-4
AMP_DPP_TOLERANCE = 1e-5
LOGGER = logging.getLogger(__name__)


def arrange_dpps(dpps):
    """
    Grouping of dpp-values and averaging them in the bins, also zeroes the bin closest to zero.
    """
    closest_to_zero = np.argmin(np.abs(dpps))
    ranges = _compute_ranges(dpps)
    zero_offset = np.mean(_values_in_range(_find_range_with_element(ranges, closest_to_zero), dpps))
    LOGGER.debug(f"dp/p closest to zero is {dpps[closest_to_zero]}")
    if np.abs(zero_offset) > DPP_TOLERANCE:
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


def _compute_ranges(dpps):
    ordered_ids = np.argsort(dpps)
    ranges = [[ordered_ids[0]]]
    if len(ordered_ids) == 1:
        return ranges
    for idx in ordered_ids[1:]:
        if abs(dpps[ranges[-1][0]] - dpps[idx]) < DPP_TOLERANCE:
            ranges[-1].append(idx)
        else:
            ranges.append([idx])
    return ranges


def append_amp_dpp(list_of_tfs, dpp_values):
    for i, dpp in enumerate(dpp_values):
        list_of_tfs[i].headers["DPPAMP"] = dpp if (not np.isnan(dpp) and np.abs(dpp) > AMP_DPP_TOLERANCE) else 0.0
    return list_of_tfs


def append_dpp(list_of_tfs, dpp_values):
    for i, dpp in enumerate(dpp_values):
        list_of_tfs[i].headers["DPP"] = dpp
    return list_of_tfs


def calculate_dpoverp(input_files, meas_input): 
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


def calculate_amp_dpoverp(input_files, meas_input):
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
    dispersions = df_filtered.loc[mask_zeros, "DX"]
    denom = np.sum(dispersions ** 2)
    if denom == 0.:
        raise ValueError("Cannot compute dpp probably no arc BPMs.")
    if amps.ndim == 1:
        numer = np.sum(dispersions * amps[mask_zeros])
    else:
        numer = np.sum(dispersions[:, None] * amps[mask_zeros, :], axis=0)
    return numer / denom
