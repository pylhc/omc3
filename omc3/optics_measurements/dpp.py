"""
.. module: dpp

Created on 2017

:author: Elena Fol, Lukas Malina

Arranges \frac{\Delta p}{p} of given files
"""

import logging
import numpy as np
import pandas as pd

DPP_TOLERANCE = 0.0001
LOGGER = logging.getLogger(__name__)


def arrange_dpp(list_of_tfs):
    """
    Grouping of dpp-values in the given linx,liny-files and computing new values
    """
    list_of_tfs_arranged = []
    for tfs_file in list_of_tfs:
        if "DPP" not in tfs_file.headers:
            tfs_file.headers["DPP"] = 0.0  # calculate_dpp(tfs_file, model)
    if len(list_of_tfs) == 1:
        only_dpp = list_of_tfs[0].headers["DPP"]
        if np.abs(only_dpp) > DPP_TOLERANCE:
            LOGGER.warning(
                'It looks like the file you are analyzing has too '
                'high momentum deviation {}. Optics parameters might '
                'be wrong.'.format(only_dpp)
            )
        list_of_tfs[0].headers["DPP"] = 0.0
        return list_of_tfs
    dpp_values = [tfs_file.DPP for tfs_file in list_of_tfs]
    if 0.0 in dpp_values:
        LOGGER.warning('Exact 0.0 found, the dp/p values are probably already grouped.')
        return list_of_tfs
    closest_to_zero = np.argmin(np.absolute(dpp_values))
    ordered_indices = np.argsort(dpp_values)
    ranges = _compute_ranges(list_of_tfs, ordered_indices)
    offset_range = _find_range_with_element(ranges, closest_to_zero)
    offset_dpps = _values_in_range(offset_range, dpp_values)
    LOGGER.debug("dp/p closest to zero is {}".format(dpp_values[closest_to_zero]))
    zero_offset = np.mean(offset_dpps)
    LOGGER.debug("Detected dpp differences, aranging as: {0}, zero offset: {1}."
                 .format(ranges, zero_offset))
    for idx in range(len(dpp_values)):
        range_to_use = _find_range_with_element(ranges, idx)
        dpps_from_range = _values_in_range(range_to_use, dpp_values)
        range_mean = np.mean(dpps_from_range)
        list_of_tfs[idx].headers["DPP"] = range_mean - zero_offset
        list_of_tfs_arranged.append(list_of_tfs[idx])
    return list_of_tfs_arranged


def _values_in_range(range_to_use, dpp_values):
    dpps_from_range = []
    for dpp_idx in range_to_use:
        dpps_from_range.append(dpp_values[dpp_idx])
    return dpps_from_range


def _find_range_with_element(ranges, element):
    range_with_element = None
    for dpp_range in ranges:
        if element in dpp_range:
            range_with_element = dpp_range
    return range_with_element


def _compute_ranges(list_of_tfs, ordered_indices):
    list_of_ranges = []
    last_range = None
    for idx in ordered_indices:
        if (list_of_ranges and
                _is_in_same_range(list_of_tfs[last_range[0]].DPP,
                                  list_of_tfs[idx].DPP)):
            last_range.append(idx)
        else:
            new_range = [idx]
            list_of_ranges.append(new_range)
            last_range = new_range
    return list_of_ranges


def _is_in_same_range(a, b):
    return a + DPP_TOLERANCE >= b >= a - DPP_TOLERANCE


#TODO
def calculate_dpoverp(meas_input, input_files, model, header_dict):
    df_orbit = pd.DataFrame(model).loc[:, ['S', 'DX']]
    df_orbit = pd.merge(df_orbit, input_files.joined_frame('X', ['CO', 'CORMS']), how='inner',
                        left_index=True, right_index=True)
    mask = meas_input.accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
    df_filtered = df_orbit.loc[mask, :]
    dispersions = df_filtered.loc[:, "DX"] * 1e3 # conversion to milimeters
    denom = np.sum(dispersions ** 2)
    if denom == 0.:
        raise ValueError("Cannot compute dpp probably no arc BPMs.")
    numer = np.sum(dispersions[:, None] * df_filtered.loc[:, input_files.get_columns(df_orbit, 'CO')].values, axis=0)
    return numer / denom


def calculate_amp_dpoverp(meas_input, input_files, model, header_dict):
    df_orbit = pd.DataFrame(model).loc[:, ['S', 'DX']]
    df_orbit = pd.merge(df_orbit, input_files.joined_frame('X', ['AMPX', 'AMPZ']), how='inner',
                        left_index=True, right_index=True)
    mask = meas_input.accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
    df_filtered = df_orbit.loc[mask, :]
    amps = df_filtered.loc[:, input_files.get_columns(df_orbit, 'AMPX')].values * df_filtered.loc[:, input_files.get_columns(df_orbit, 'AMPZ')].values
    mask_zeros = (amps > 0) if amps.ndim == 1 else (np.sum(amps,axis=1) > 0)

    dispersions = df_filtered.loc[mask_zeros, "DX"] * 1e3 # conversion to milimeters
    denom = np.sum(dispersions ** 2)
    if denom == 0.:
        raise ValueError("Cannot compute dpp probably no arc BPMs.")
    if amps.ndim == 1:
        numer = np.sum(dispersions * amps[mask_zeros])
    else:
        numer = np.sum(dispersions[:, None] * amps[mask_zeros, :], axis=0)
    print(numer / denom)
    return numer / denom



