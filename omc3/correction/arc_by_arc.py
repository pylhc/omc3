""" 
Arc-by-Arc Global Correction
----------------------------

In this module, functions are provided to modify the linear equation problem 
in global correction, correcting the phase advance at each BPM, into a 
problem of correcting the phase-advances over the whole arcs.

This is done by identifying the closest BPM to the IPs defining the arc,
available in the measurement data and summing all measured phase-advances between these.

In the current implementation, only the measured data is modified
to contain the arc phase advances, which will then be globally corrected with
the given correctors.

In a future implementation this should be extended to loop over each arc and 
correct each individually with only the correctors available in the respective arc.

For now everything is very LHC specific, a future implementation should also
extract the accelerator specific parts into the accelerator class.

See https://github.com/pylhc/omc3/issues/480 .
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs

from omc3.correction.constants import DIFF, ERROR, MODEL, VALUE, WEIGHT
from omc3.definitions.constants import PLANE_TO_NUM 
from omc3.optics_measurements.constants import NAME, NAME2, PHASE, TUNE

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal


LHC_ARCS = ('81', '12', '23', '34', '45', '56', '67', '78')


def reduce_phase_measurements_to_arcs(
    meas_dict: dict[str, pd.DataFrame], 
    model: tfs.TfsDataFrame, 
    include_ips: str | None = None
    ):
    """ Reduce the phase-advance in the given measurement to the phase-advance 
    between two BPM-pairs at the extremities of each arc.

    Args:
        meas_dict (dict[str, pd.DataFrame]): Dictionary of measurements as used in Global Correction.
        model (tfs.TfsDataFrame): Model of the machine, used only the get the tunes from the headers.
        include_ips (str | None): Include the IPs of each arc. Can be either 'left', 'right', 'both' or None

    Returns:
        dict[str, pd.DataFrame]: The modified measurement dict.
    """
    meas_dict = meas_dict.copy()

    for plane, tune_num in PLANE_TO_NUM.items():
        phase_df = meas_dict[f"{PHASE}{plane}"]

        bpm_pairs = get_arc_by_arc_bpm_pairs(phase_df.index, include_ips)
        new_phase_df = get_bpm_pair_phases(phase_df, bpm_pairs=bpm_pairs, tune=model.headers[f"{TUNE}{tune_num}"])

        meas_dict[f"{PHASE}{plane}"] = new_phase_df
    return meas_dict


# Identify BPM Pairs -----------------------------------------------------------

def get_arc_by_arc_bpm_pairs(bpms:  Sequence[str],  include_ips: str | None = None) -> dict[str, tuple[str, str]]:
    """Get a dictionary of bpm_pairs for each arc, defining the start and end of each arc.

    Args:
        bpms (Sequence[str]): List of BPMs.
        include_ips (str | None): Include the IPs of each arc. Can be either 'left', 'right', 'both' or None

    Returns:
        dict[str, tuple[str, str]]: Mapping of arc to BPM pairs to use for each arc.
    """
    bpm_pairs = {}
    for arc in LHC_ARCS:
        bpm_pairs[arc] = get_left_right_pair(bpms, arc)
    
    if include_ips is None:
        return bpm_pairs

    # Include IPs within each arc
    # i.e. choose the closest BPMs on the other side of the IP ---
    bpm_pairs_with_ips = {}
    for idx, arc in enumerate(LHC_ARCS):
        prev_arc = LHC_ARCS[idx-1]
        next_arc = LHC_ARCS[(idx+1) % len(LHC_ARCS)]

        bpm_left, bpm_right = bpm_pairs[arc]
        if include_ips in ('left', 'both'):
            bpm_left = bpm_pairs[prev_arc][1]
        
        if include_ips in ('right', 'both'): 
            bpm_right = bpm_pairs[next_arc][0]
        
        bpm_pairs_with_ips[arc] = (bpm_left, bpm_right)

    return bpm_pairs_with_ips


def get_left_right_pair(bpms: Sequence[str], arc: str) -> tuple[str, str]:
    """ Get the pair of BPMs that are furthest apart in the given arc, i.e.
    the ones closest to the IPs defining the arc, left and right. 
    
    Args:
        bpms (Sequence[str]): List of BPMs.
        arc (str): Arc to find the BPMs in (e.g. '12')
    
    Returns:
        tuple[str, str]: The found BPM pair.
    """
    left_of_arc = identify_closest_arc_bpm_to_ip(bpms, ip=int(arc[0]), side='R')
    right_of_arc = identify_closest_arc_bpm_to_ip(bpms, ip=int(arc[1]), side='L')
    return left_of_arc, right_of_arc


def identify_closest_arc_bpm_to_ip(bpms: Sequence[str], ip: int, side: Literal["L", "R"]) -> str:
    """ Pick the BPM with the lowest index from the given sequence, that is on the 
    given side of the given IP.
    
    TODO: Use a regex instead, filtering the list by [LR]IP and choose the lowest via sort.
    This would assure that also BPMW etc. could be used. (jdilly, 2025)
    """
    beam = list(bpms)[0][-1]

    indices = range(1, 15)
    for ii in indices:
        bpm = f'BPM.{ii}{side}{ip}.B{beam}'
        if bpm in bpms:
            return bpm
    
    msg = (
        f"No BPM up to index {ii} could be found in the measurement of arc on {side} of IP{ip} "
        f" in beam {beam} for the arc-by-arc phase correction."
    )
    raise ValueError(msg)


# Phase Summation --------------------------------------------------------------

def get_bpm_pair_phases(phase_df: pd.DataFrame, bpm_pairs: dict[tuple[str, str]], tune: float) -> pd.DataFrame:
    """Create a new DataFrame containing as entries the phase advances between the given bpm pairs.
    The format/columns are the same as used by global correction.

    Args:
        phase_df (pd.DataFrame): Old DataFrame containing all the phase advances between the measured BPMs.
        bpm_pairs (dict[tuple[str, str]]): Identified BPM pairs to be used.
        tune (float): Model tune of the machine.

    Returns:
        pd.DataFrame: New DataFrame containing the phase advances between the given bpm pairs. 
    """
    arc_meas: list[dict] = []
    for bpm_pair in bpm_pairs.values():
        results = {
            NAME: bpm_pair[0],
            NAME2: bpm_pair[1],
            WEIGHT: phase_df.loc[bpm_pair[0], WEIGHT],
            VALUE: circular_sum_phase(phase_df[VALUE], tune, bpm_pair),
            MODEL: circular_sum_phase(phase_df[MODEL], tune, bpm_pair),
            ERROR: circular_sum_phase_error(phase_df[ERROR], bpm_pair)
        }
        results[DIFF] = results[VALUE] - results[MODEL]
        arc_meas.append(results)
    
    return pd.DataFrame(arc_meas).set_index(NAME)


def circular_sum_phase(phases: pd.Series, tune: float, bpm_pair: tuple[str, str]):
    """ Calculate the sum of the phases from bpm to bpm 
    of the given bpm pair, taking into account the circularity of the accelerator. """
    idx_0, idx_1 = phases.index.get_loc(bpm_pair[0]), phases.index.get_loc(bpm_pair[1])
    if idx_0 < idx_1:
        return sum(phases[bpm_pair[0]:bpm_pair[1]])
    
    # cycle phases
    inverted_result = sum(phases[bpm_pair[1]:bpm_pair[0]])
    return tune - inverted_result


def circular_sum_phase_error(phase_errors: pd.Series, bpm_pair: tuple[str, str]):
    """ Calculate the sum of the phases errors from bpm to bpm 
    of the given bpm pair, taking into account the circularity of the accelerator. """
    idx_0, idx_1 = phase_errors.index.get_loc(bpm_pair[0]), phase_errors.index.get_loc(bpm_pair[1])
    if idx_0 < idx_1:
        return np.sqrt(np.sum(phase_errors[bpm_pair[0]:bpm_pair[1]]**2))

    # cycle errors
    selection = pd.concat([phase_errors.loc[:bpm_pair[1]], phase_errors.loc[bpm_pair[0]:]])
    return np.sqrt(np.sum(selection**2))
