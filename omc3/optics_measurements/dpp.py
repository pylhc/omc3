"""
.. module: dpp

Created on 2017

:author: Elena Fol

Arranges \frac{\Delta p}{p} of given files
"""

import logging
import numpy as np

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

# TODO actual dpp calculation from orbit moved from hole_in_one
"""
def _calc_dp_over_p(main_input, bpm_data):
    model_twiss = tfs.read(main_input.model)
    model_twiss.set_index("NAME", inplace=True)
    sequence = model_twiss.headers["SEQUENCE"].lower().replace("b1", "").replace("b2", "")
    if sequence != "lhc":
        return 0.0  # TODO: What do we do with other accels.
    accel_cls = manager.get_accel_class(accel=sequence)
    arc_bpms_mask = accel_cls.get_element_types_mask(bpm_data.index, types=["arc_bpm"])
    mask = accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])

    arc_bpms_mask = np.array([bool(x) for x in arc_bpms_mask])
    arc_bpm_data = bpm_data.loc[arc_bpms_mask]
    # We need it in mm:
    dispersions = model_twiss.loc[arc_bpm_data.index, "DX"] * 1e3
    closed_orbits = np.mean(arc_bpm_data, axis=1)
    numer = np.sum(dispersions * closed_orbits)
    denom = np.sum(dispersions ** 2)
    if denom == 0.:
        raise ValueError("Cannot compute dpp probably no arc BPMs.")
    dp_over_p = numer / denom
    return dp_over_p
"""
