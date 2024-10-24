"""
Clean
-----

This module contains the cleaning functionality of ``harpy``.
"""
import numpy as np
import pandas as pd

from omc3.utils import logging_tools
from omc3.utils.contexts import timeit

LOGGER = logging_tools.getLogger(__name__)
NTS_LIMIT = 8.  # Noise to signal limit


def clean(harpy_input, bpm_data, model):
    """
    Cleans BPM TbT matrix: removes BPMs not present in the model and based on specified cuts.
    Also cleans the noise using singular value decomposition.

    Args:
        harpy_input: The input object containing the analysis settings.
        bpm_data: DataFrame of BPM TbT matrix indexed by BPM names.
        model: model containing BPMs longitudinal locations indexed by BPM names.

    Returns:
        Clean BPM matrix, its decomposition, bad BPMs summary and estimated BPM resolutions
    """
    bpm_data, bpms_not_in_model = _get_only_model_bpms(bpm_data, model)
    if bpm_data.empty:
        raise AssertionError("Check BPMs names! None of the BPMs were found in the model!")
    if not harpy_input.clean:
        return bpm_data, None, bpms_not_in_model, None
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
    bpm_data_mean = np.mean(bpm_data.to_numpy(), axis=1)
    u_mat, sv_mat, u_mask = svd_decomposition(bpm_data - bpm_data_mean[:, None],
                                              harpy_input.sing_val,
                                              dominance_limit=harpy_input.svd_dominance_limit,
                                              num_iter=harpy_input.num_svd_iterations)

    clean_u, dominant_bpms = _clean_dominant_bpms(u_mat, u_mask, harpy_input.svd_dominance_limit)
    clean_data = clean_u.dot(sv_mat) + bpm_data_mean[np.all(u_mask, axis=1), None]
    bpm_res = (clean_data - bpm_data.loc[clean_u.index]).std(axis=1)
    orbit_offset = (clean_data - bpm_data.loc[clean_u.index]).mean(axis=1)
    LOGGER.debug(f"Average closed orbit offset: {np.mean(orbit_offset)}")
    LOGGER.debug(f"Average BPM resolution: {np.mean(bpm_res)}")
    average_signal = np.mean(np.std(clean_data, axis=1))
    LOGGER.debug(f"np.mean(np.std(A, axis=1): {average_signal}")
    if np.mean(bpm_res) > NTS_LIMIT * average_signal:
        raise ValueError("The data is too noisy. The most probable explanation "
                         "is that there was no excitation or it was very low.")
    return clean_data, bpm_res, dominant_bpms, (clean_u, sv_mat - np.mean(sv_mat, axis=1)[:, None])


def _detect_known_bad_bpms(bpm_data, list_of_bad_bpms):
    """Searches for known bad BPMs."""
    return bpm_data.index.intersection(list_of_bad_bpms)


def _detect_flat_bpms(bpm_data, min_peak_to_peak):
    """Detects BPMs with the same values for all turns."""
    cond = ((bpm_data.max(axis=1) - bpm_data.min(axis=1)).abs() < min_peak_to_peak)
    bpm_flatness = bpm_data[cond].index
    if bpm_flatness.size:
        LOGGER.debug(f"Flat BPMS detected (diff min/max <= {min_peak_to_peak}. "
                     f"BPMs removed: {bpm_flatness.size}")
    return bpm_flatness


def _detect_bpms_with_spikes(bpm_data, max_peak_cut):
    """Detects BPMs with spikes > `max_peak_cut`."""
    too_high = bpm_data[bpm_data.max(axis=1) > max_peak_cut].index
    too_low = bpm_data[bpm_data.min(axis=1) < -max_peak_cut].index
    bpm_spikes = too_high.union(too_low)
    if bpm_spikes.size:
        LOGGER.debug(f"Spikes > {max_peak_cut} detected. BPMs removed: {bpm_spikes.size}")
    return bpm_spikes


def _detect_bpms_with_exact_zeros(bpm_data, keep_exact_zeros):
    """Detects BPMs with exact zeros due to OP workaround."""
    if keep_exact_zeros:
        LOGGER.debug("Skipped exact zero check")
        return pd.Index([])
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
    if n_good_bpms == 0:
        raise ValueError("Total Number of BPMs after filtering is zero.")
    n_bad_bpms = n_total_bpms - n_good_bpms
    LOGGER.debug(f"(Statistics for file reading) Total BPMs: {n_total_bpms}, "
                 f"Good BPMs: {n_good_bpms} ({(100 * n_good_bpms / n_total_bpms):2.2f}%), "
                 f"Bad BPMs: {n_bad_bpms} ({(100 * n_bad_bpms / n_total_bpms):2.2f}%)")
    if (n_good_bpms / n_total_bpms) < 0.5:
        raise ValueError("More than half of BPMs are bad. "
                         "This could be because a bunch not present in the machine has been "
                         "selected or because of a problem with the phasing of the BPMs.")


def _index_union(*indices):
    new_index = pd.Index([])
    for index in indices:
        new_index = new_index.union(index)
    return new_index


def _fix_polarity(wrong_polarity_names, bpm_data):
    """Fixes wrong polarity."""
    bpm_data.loc[wrong_polarity_names, :] = -1 * bpm_data.loc[wrong_polarity_names, :].to_numpy()
    return bpm_data


def _resync_bpms(harpy_input, bpm_data, model):
    """Resynchronizes BPMs between the injection point and start of the lattice."""
    LOGGER.debug("Will resynchronize BPMs")
    bpm_pos = model.index.get_loc(harpy_input.first_bpm)
    if harpy_input.opposite_direction:
        mask = np.array([x in model.index[bpm_pos::-1] for x in bpm_data.index])
    else:
        mask = np.array([x in model.index[bpm_pos:] for x in bpm_data.index])
    bpm_data.loc[mask] = np.roll(bpm_data.loc[mask], -1, axis=1)
    return bpm_data.iloc[:, :-1]


def svd_decomposition(matrix, num_singular_values, dominance_limit=None, num_iter=None):
    """
    Computes reduced (K largest values) singular value decomposition of a matrix.
    Requiring K singular values from MxN matrix results in matrices sized:
    `((M,K) x diag(K) x (K,N))`

    Args:
        matrix: matrix to be decomposed.
        num_singular_values: Required number of singular values for reconstruction.
        dominance_limit: limit on SVD dominance.
        num_iter: maximal number of iteration to remove elements and renormalise matrices.

    Returns:
        An indexed DataFrame of U matrix `(M,K)`,
        the product of S and V^T matrices `(diag(K).x(K,N))`,
        and the U matrix mask for cleaned elements.
    """
    u_mat, s_mat, vt_mat = np.linalg.svd(matrix, full_matrices=False)
    u_mat, s_mat, u_mat_mask = _remove_dominant_elements(u_mat, s_mat, dominance_limit, num_iter=num_iter)

    available = np.sum(s_mat > 0.)
    if num_singular_values > available:
        LOGGER.warning(f"Requested more singular values than available(={available})")
    keep = min(num_singular_values, available)
    LOGGER.debug(f"Number of singular values to keep: {keep}")

    indices = np.argsort(s_mat)[::-1][:keep]
    return (pd.DataFrame(index=matrix.index, data=u_mat[:, indices]),
            np.dot(np.diag(s_mat[indices]), vt_mat[indices, :]),
            u_mat_mask[:, :int(np.max(indices))+1])


def _remove_dominant_elements(u_mat, s_mat, dominance_limit, num_iter=3):
    u_mat_mask = np.ones(u_mat.shape, dtype=bool)
    if dominance_limit is None:
        return u_mat, s_mat, u_mat_mask
    if dominance_limit < 1 / np.sqrt(2):
        LOGGER.warning(f"The svd_dominance_limit looks too low: {dominance_limit}")

    for i in range(num_iter):
        if np.all(np.abs(u_mat) <= dominance_limit):
            break
        condition = np.logical_and(np.abs(u_mat) > dominance_limit,
                                   np.abs(u_mat) == np.max(np.abs(u_mat), axis=0))
        u_mat_mask[condition] = False
        u_mat[condition] = 0.0
        norms = np.sqrt(np.sum(np.square(u_mat), axis=0))
        u_mat = u_mat / norms
        s_mat = s_mat * norms
    return u_mat, s_mat, u_mat_mask


def _clean_dominant_bpms(u_mat, u_mat_mask, svd_dominance_limit):
    dominant_bpms = u_mat[np.any(~u_mat_mask, axis=1)].index
    if dominant_bpms.size > 0:
        LOGGER.debug(f"Bad BPMs from SVD detected. Number of BPMs removed: {dominant_bpms.size}")
    clean_u = u_mat.loc[u_mat.index.difference(dominant_bpms, sort=False)]
    return clean_u, [f"{bpm_name} Dominant BPM in SVD, peak value > {svd_dominance_limit}"
                     for bpm_name in dominant_bpms]
