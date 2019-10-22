"""
clean
--------------------

Cleaning functionality of harpy.

"""


import numpy as np
import pandas as pd

from utils import logging_tools
from utils.contexts import timeit

LOGGER = logging_tools.getLogger(__name__)
NTS_LIMIT = 8.  # Noise to signal limit


def clean(harpy_input, bpm_data, model):
    """
    Cleans BPM TbT matrix: removes BPMs not present in the model and based on specified cuts.
    Also cleans the noise using singular value decomposition.

    Args:
        harpy_input: The input object containing the analysis settings
        bpm_data: DataFrame of BPM TbT matrix indexed by BPM names
        model: model containing BPMs longitudinal locations indexed by BPM names

    Returns:
        Clean BPM matrix, its decomposition, bad BPMs summary and estimated BPM resolutions
    """
    if not harpy_input.clean:
        return bpm_data, None, [], None
    bpm_data, bpms_not_in_model = _get_only_model_bpms(bpm_data, model)
    if bpm_data.empty:
        raise AssertionError("Check BPMs names! None of the BPMs was found in the model!")
    with timeit(lambda spanned: LOGGER.debug(f"Time for filtering: {spanned}")):
        bpm_data, bad_bpms_clean = _cut_cleaning(harpy_input, bpm_data, model)
    with timeit(lambda spanned: LOGGER.debug(f"Time for SVD clean: {spanned}")):
        bpm_data, bpm_res, bad_bpms_svd, usv = _svd_clean(bpm_data, harpy_input)
        all_bad_bpms = bpms_not_in_model + bad_bpms_clean + bad_bpms_svd
    return bpm_data, usv, all_bad_bpms, bpm_res


def _get_only_model_bpms(bpm_data, model):
    if model is None:
        return bpm_data, []
    bpm_data_in_model = bpm_data.loc[model.index.intersection(bpm_data.index)]
    not_in_model = bpm_data.index.difference(model.index)
    return bpm_data_in_model, [f"{bpm} not found in model" for bpm in not_in_model]


def _cut_cleaning(harpy_input, bpm_data, model):
    LOGGER.debug(f"Number of BPMs in the input {bpm_data.index.size}")
    known_bad_bpms = _detect_known_bad_bpms(bpm_data, harpy_input.bad_bpms)
    bpm_flatness = _detect_flat_bpms(bpm_data, harpy_input.peak_to_peak)
    bpm_spikes = _detect_bpms_with_spikes(bpm_data, harpy_input.max_peak)
    exact_zeros = _detect_bpms_with_exact_zeros(bpm_data, harpy_input.keep_exact_zeros)
    all_bad_bpms = _index_union(known_bad_bpms, bpm_flatness, bpm_spikes, exact_zeros)
    original_bpms = bpm_data.index

    bpm_data = bpm_data.loc[bpm_data.index.difference(all_bad_bpms, sort=False)]
    bad_bpms_with_reasons = _get_bad_bpms_summary(
        harpy_input, known_bad_bpms, bpm_flatness, bpm_spikes, exact_zeros
    )
    _report_clean_stats(original_bpms.size, bpm_data.index.size)
    bpm_data = _fix_polarity(harpy_input.wrong_polarity_bpms, bpm_data)
    if model is not None and harpy_input.first_bpm is not None:
        bpm_data = _resync_bpms(harpy_input, bpm_data, model)
    return bpm_data, bad_bpms_with_reasons


def _svd_clean(bpm_data, harpy_input):
    u_mat, sv_mat, bpm_data_mean = svd_decomposition(bpm_data, harpy_input.sing_val)
    clean_u, dominant_bpms = _clean_dominant_bpms(u_mat, harpy_input.svd_dominance_limit)
    clean_data = clean_u.dot(sv_mat) + bpm_data_mean
    bpm_res = (clean_data - bpm_data.loc[clean_u.index]).std(axis=1)
    LOGGER.debug(f"Average BPM resolution: {np.mean(bpm_res)}")
    average_signal = np.mean(np.std(clean_data, axis=1))
    LOGGER.debug(f"np.mean(np.std(A, axis=1): {average_signal}")
    if np.mean(bpm_res) > NTS_LIMIT * average_signal:
        raise ValueError("The data is too noisy. The most probable explanation "
                         "is that there was no excitation or it was very low.")
    return clean_data, bpm_res, dominant_bpms, (clean_u, sv_mat - np.mean(sv_mat, axis=1)[:, None])


def _detect_known_bad_bpms(bpm_data, list_of_bad_bpms):
    """  Searches for known bad BPMs  """
    return bpm_data.index.intersection(list_of_bad_bpms)


def _detect_flat_bpms(bpm_data, min_peak_to_peak):
    """  Detects BPMs with the same values for all turns  """
    cond = ((bpm_data.max(axis=1) - bpm_data.min(axis=1)).abs() < min_peak_to_peak)
    bpm_flatness = bpm_data[cond].index
    if bpm_flatness.size:
        LOGGER.debug(f"Flat BPMS detected (diff min/max <= {min_peak_to_peak}. "
                     f"BPMs removed: {bpm_flatness.size}")
    return bpm_flatness


def _detect_bpms_with_spikes(bpm_data, max_peak_cut):
    """  Detects BPMs with spikes > max_peak_cut  """
    too_high = bpm_data[bpm_data.max(axis=1) > max_peak_cut].index
    too_low = bpm_data[bpm_data.min(axis=1) < -max_peak_cut].index
    bpm_spikes = too_high.union(too_low)
    if bpm_spikes.size:
        LOGGER.debug(f"Spikes > {max_peak_cut} detected. BPMs removed: {bpm_spikes.size}")
    return bpm_spikes


def _detect_bpms_with_exact_zeros(bpm_data, keep_exact_zeros):
    """  Detects BPMs with exact zeros due to OP workaround  """
    if keep_exact_zeros:
        LOGGER.debug("Skipped exact zero check")
        return pd.DataFrame()
    exact_zeros = bpm_data[~np.all(bpm_data, axis=1)].index
    if exact_zeros.size:
        LOGGER.debug(f"Exact zeros detected. BPMs removed: {exact_zeros.size}")
    return exact_zeros


def _get_bad_bpms_summary(harpy_input, known_bad_bpms, bpm_flatness, bpm_spikes, exact_zeros):
    return ([f"{bpm_name} Known bad BPM" for bpm_name in known_bad_bpms] +
            [f"{bpm_name} Flat BPM, the difference between min/max is smaller than "
            f"{harpy_input.peak_to_peak}" for bpm_name in bpm_flatness] +
            [f"{bpm_name} Spiky BPM, found spike higher than "
            f"{harpy_input.max_peak}" for bpm_name in bpm_spikes] +
            [f"{bpm_name} Found an exact zero" for bpm_name in exact_zeros])


def _report_clean_stats(n_total_bpms, n_good_bpms):
    LOGGER.debug("Filtering done:")
    if n_total_bpms == 0:
        raise ValueError("Total Number of BPMs after filtering is zero")
    n_bad_bpms = n_total_bpms - n_good_bpms
    LOGGER.debug(f"(Statistics for file reading) Total BPMs: {n_total_bpms}, "
                 f"Good BPMs: {n_good_bpms} ({(100 * n_good_bpms / n_total_bpms):2.2f}%), "
                 f"Bad BPMs: {n_bad_bpms} ({(100 * n_bad_bpms / n_total_bpms):2.2f}%)")
    if (n_good_bpms / n_total_bpms) < 0.5:
        raise ValueError("More than half of BPMs are bad. "
                         "This could be cause a bunch not present in the machine has been "
                         "selected or because a problem with the phasing of the BPMs.")


def _index_union(*indices):
    new_index = pd.Index([])
    for index in indices:
        new_index = new_index.union(index)
    return new_index


def _fix_polarity(wrong_polarity_names, bpm_data):
    """  Fixes wrong polarity  """
    bpm_data.loc[wrong_polarity_names] *= -1
    return bpm_data


def _resync_bpms(harpy_input, bpm_data, model):
    """  Resynchronizes BPMs between the injection point and start of the lattice.  """
    LOGGER.debug("Will resynchronize BPMs")
    bpm_pos = model.index.get_loc(harpy_input.first_bpm)
    if harpy_input.opposite_direction:
        mask = np.array([x in model.index[bpm_pos::-1] for x in bpm_data.index])
    else:
        mask = np.array([x in model.index[bpm_pos:] for x in bpm_data.index])
    bpm_data.loc[mask] = np.roll(bpm_data.loc[mask], -1, axis=1)
    return bpm_data.iloc[:, :-1]


def svd_decomposition(bpm_data, num_singular_values):
    """
    Computes reduced (n largest values) singular value docomposition of a matrix (bpm_data)

    Args:
        bpm_data: matrix to be decomposed
        num_singular_values: input options object that contains

    Returns:
        An indexed DataFrame of U matrix, product of S and V^T martices, and mean of original matrix
    """
    bpm_data_mean = bpm_data.to_numpy().mean()
    u_mat, svt_mat = _get_decomposition(bpm_data - bpm_data_mean, num_singular_values)
    return pd.DataFrame(index=bpm_data.index, data=u_mat), svt_mat, bpm_data_mean


def _get_decomposition(matrix, num):
    """
    Removes noise floor
    Requiring K singular values from MxN matrix results in matrices sized: ((MxK) x diag(K) x (K,N))

    Returns:
        U (MxK),  SVt (diag(K).(K,N))
    """
    u_mat, s_mat, vt_mat = np.linalg.svd(matrix / np.sqrt(matrix.shape[1]), full_matrices=False)
    available = np.sum(s_mat > 0.)
    if num > available:
        LOGGER.warning(f"Requested more singular values than available(={available})")
    keep = min(num, available)
    LOGGER.debug(f"Number of singular values to keep: {keep}")
    return (u_mat[:, :keep],
            np.dot(np.sqrt(matrix.shape[1]) * np.diag(s_mat[:keep]), vt_mat[:keep, :]))


def _clean_dominant_bpms(u_mat, svd_dominance_limit):
    if svd_dominance_limit < 1 / np.sqrt(2):
        LOGGER.warning(f"The svd_dominance_limit looks too low: {svd_dominance_limit}")
    dominant_bpms = u_mat[np.max(u_mat.abs(), axis=1) > svd_dominance_limit].index
    if dominant_bpms.size > 0:
        LOGGER.debug(f"Bad BPMs from SVD detected. Number of BPMs removed: {dominant_bpms.size}")
    clean_u = u_mat.loc[u_mat.index.difference(dominant_bpms, sort=False)]
    return clean_u, [f"{bpm_name} Dominant BPM in SVD, peak value > {svd_dominance_limit}"
                     for bpm_name in dominant_bpms]
