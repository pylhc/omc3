import numpy as np
import pandas as pd

from utils import logging_tools
from utils.contexts import timeit

LOGGER = logging_tools.getLogger(__name__)
NTS_LIMIT = 8.  # Noise to signal limit
PLANES = ("X", "Y")


def cut_clean(clean_input, bpm_data, model):
    LOGGER.debug(f"clean: number of BPMs in the input {bpm_data.index.size}")
    known_bad_bpms = detect_known_bad_bpms(bpm_data, clean_input.bad_bpms)
    bpm_flatness = detect_flat_bpms(bpm_data, clean_input.peak_to_peak)
    bpm_spikes = detect_bpms_with_spikes(bpm_data, clean_input.max_peak)
    if not clean_input.no_exact_zeros:
        exact_zeros = detect_bpms_with_exact_zeros(bpm_data)
    else:
        exact_zeros = pd.DataFrame()
        LOGGER.debug(f"clean: Skipped exact zero check {exact_zeros.size}")

    original_bpms = bpm_data.index
    all_bad_bpms = _index_union(known_bad_bpms, bpm_flatness, bpm_spikes, exact_zeros)
    bpm_data = bpm_data.loc[bpm_data.index.difference(all_bad_bpms)]

    bad_bpms_with_reasons = _get_bad_bpms_summary(
        clean_input, known_bad_bpms, bpm_flatness, bpm_spikes, exact_zeros
    )
    _report_clean_stats(original_bpms.size, bpm_data.index.size)
    bpm_data = fix_polarity(clean_input.wrong_polarity_bpms, bpm_data)
    if clean_input.first_bpm is not None:
        bpm_data = resync_bpms(clean_input, bpm_data, model)
    return bpm_data, bad_bpms_with_reasons


def fix_polarity(wrong_polarity_names, bpm_data):
    """  Fixes wrong polarity  """
    bpm_data.loc[wrong_polarity_names] *= -1
    return bpm_data


def resync_bpms(clean_input, bpm_data, model):
    """  Resynchronizes BPMs between the injection point and start of the lattice.  """
    LOGGER.debug("Will resynchronize BPMs")
    bpm_pos = model.index.get_loc(clean_input.first_bpm)
    if clean_input.opposite_direction:
        mask = np.array([x in model.index[bpm_pos::-1] for x in bpm_data.index])
    else:
        mask = np.array([x in model.index[bpm_pos:] for x in bpm_data.index])
    bpm_data.loc[mask] = np.roll(bpm_data.loc[mask], -1, axis=1)
    return bpm_data.iloc[:, :-1]


def detect_known_bad_bpms(bpm_data, list_of_bad_bpms):
    """  Searches for known bad BPMs  """
    return bpm_data.index.intersection(list_of_bad_bpms)


def detect_flat_bpms(bpm_data, min_peak_to_peak):
    """  Detects BPMs with the same values for all turns  """
    cond = ((bpm_data.max(axis=1) - bpm_data.min(axis=1)).abs() < min_peak_to_peak)
    bpm_flatness = bpm_data[cond].index
    if bpm_flatness.size:
        LOGGER.debug(f"Flat BPMS detected (diff min/max <= {min_peak_to_peak}. "
                     f"BPMs removed: {bpm_flatness.size}")
    return bpm_flatness


def detect_bpms_with_spikes(bpm_data, max_peak_cut):
    """  Detects BPMs with spikes > max_peak_cut  """
    too_high = bpm_data[bpm_data.max(axis=1) > max_peak_cut].index
    too_low = bpm_data[bpm_data.min(axis=1) < -max_peak_cut].index
    bpm_spikes = too_high.union(too_low)
    if bpm_spikes.size:
        LOGGER.debug(f"Spikes > {max_peak_cut} detected. BPMs removed: {bpm_spikes.size}")
    return bpm_spikes


def detect_bpms_with_exact_zeros(bpm_data):
    """  Detects BPMs with exact zeros due to OP workaround  """
    exact_zeros = bpm_data[~np.all(bpm_data, axis=1)].index
    if exact_zeros.size:
        LOGGER.debug(f"Exact zeros detected. BPMs removed: {exact_zeros.size}")
    return exact_zeros


def svd_decomposition(clean_input, bpm_data):
    bpm_data_mean = bpm_data.values.mean()
    normalized_data = bpm_data - bpm_data_mean
    u_mat, svt_mat = _get_decomposition(normalized_data, clean_input.sing_val)
    return pd.DataFrame(index=bpm_data.index, data=u_mat), svt_mat, bpm_data_mean


def svd_clean(bpm_data, clean_input):
    u_mat, sv_mat, bpm_data_mean = svd_decomposition(clean_input, bpm_data)
    clean_u, dominance_summary = _clean_dominant_bpms(u_mat, clean_input.svd_dominance_limit)
    good_bpm_data = clean_u.dot(sv_mat) + bpm_data_mean
    bpm_res = (good_bpm_data - bpm_data.loc[clean_u.index]).std(axis=1)
    LOGGER.debug(f"Average BPM resolution: {np.mean(bpm_res)}")
    average_signal = np.mean(np.std(good_bpm_data, axis=1))
    LOGGER.debug(f"np.mean(np.std(A, axis=1): {average_signal}")
    if np.mean(bpm_res) > NTS_LIMIT * average_signal:
        raise ValueError("The data is too noisy. The most probable explanation "
                         "is that there was no excitation or it was very low.")
    return good_bpm_data, bpm_res, dominance_summary, (clean_u, sv_mat - np.mean(sv_mat, axis=1)[:, None])


def _get_bad_bpms_summary(clean_input, known_bad_bpms, bpm_flatness, bpm_spikes, exact_zeros):
    return ([f"{bpm_name} Known bad BPM" for bpm_name in known_bad_bpms] +
            [f"{bpm_name} Flat BPM, the difference between min/max is smaller than "
            f"{clean_input.peak_to_peak}" for bpm_name in bpm_flatness] +
            [f"{bpm_name} Spiky BPM, found spike higher than "
            f"{clean_input.max_peak}" for bpm_name in bpm_spikes] +
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


def _get_decomposition(matrix, num):
    """
    Removes noise floor
    Requiring K singular values from MxN matrix results in matrices sized: ((MxK) x diag(K) x (K,N))
    Returns: U (MxK),  SVt (diag(K).(K,N))
    """
    u_mat, s_mat, vt_mat = np.linalg.svd(matrix / np.sqrt(matrix.shape[1]), full_matrices=False)
    available = np.sum(s_mat > 0.)
    if num > available:
        LOGGER.warning(f"Requested more singular values than available(={available})")
    keep = min(num, available)
    LOGGER.debug(f"Number of singular values to keep: {keep}")
    return u_mat[:, :keep], np.dot(np.sqrt(matrix.shape[1]) * np.diag(s_mat[:keep]), vt_mat[:keep, :])


def _clean_dominant_bpms(u_mat, svd_dominance_limit):
    if svd_dominance_limit < 1 / np.sqrt(2):
        LOGGER.warning(f"The svd_dominance_limit looks too low: {svd_dominance_limit}")
    dominant_bpms = u_mat[np.max(u_mat.abs(), axis=1) > svd_dominance_limit].index
    if dominant_bpms.size > 0:
        LOGGER.debug(f"Bad BPMs from SVD detected. Number of BPMs removed: {dominant_bpms.size}")
    clean_u = u_mat.loc[u_mat.index.difference(dominant_bpms)]
    return clean_u, [f"{bpm_name} Dominant BPM in SVD, peak value > {svd_dominance_limit}"
                     for bpm_name in dominant_bpms]


def get_only_model_bpms(bpm_data, model):
    bpm_data_in_model = bpm_data.loc[model.index.intersection(bpm_data.index)]
    not_in_model = bpm_data.index.difference(model.index)
    return bpm_data_in_model, [f"{bpm} not found in model" for bpm in not_in_model]


def clean(clean_input, bpm_datas, model_tfs):
    usvs = {"X": None, "Y": None}
    all_bad_bpms = {"X": [], "Y": []}
    bpm_ress = {"X": None, "Y": None}
    if not clean_input.clean:
        return bpm_datas, usvs, all_bad_bpms, bpm_ress
    new_bpm_datas = bpm_datas
    for plane in PLANES:
        bpm_data, bpms_not_in_model = get_only_model_bpms(bpm_datas[plane], model_tfs)
        if bpm_data.empty:
            raise AssertionError("Check BPMs names! None of the BPMs was found in the model!")
        with timeit(lambda spanned: LOGGER.debug(f"Time for filtering: {spanned}")):
            bpm_data, bad_bpms_clean = cut_clean(clean_input, bpm_data, model_tfs)
        with timeit(lambda spanned: LOGGER.debug(f"Time for SVD clean: {spanned}")):
            bpm_data, bpm_res, bad_bpms_svd, usv = svd_clean(bpm_data, clean_input,)
        bpm_ress[plane] = bpm_res
        all_bad_bpms[plane] = bpms_not_in_model + bad_bpms_clean + bad_bpms_svd
        new_bpm_datas[plane] = bpm_data
        usvs[plane] = usv

    return new_bpm_datas, usvs, all_bad_bpms, bpm_ress