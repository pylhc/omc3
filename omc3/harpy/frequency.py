"""
Frequency
---------

This module contains the frequency calculations related functionality of ``harpy``.

It provides functions to calculate the frequency spectra of turn-by-turn data.
This calculation is done using a combination of SVD decomposition zero_padded `fft` to speed up the
analysis.
Also searches for resonances in the calculated spectra.
"""
from numbers import Number

import numpy as np
import pandas as pd

from omc3.utils import logging_tools, outliers
from omc3.definitions.constants import PLANES, PI2
from omc3.harpy.constants import (COL_TUNE, COL_AMP, COL_MU,
                                  COL_NATTUNE, COL_NATAMP, COL_NATMU,
                                  COL_FREQ, COL_PHASE)
from optics_functions.rdt import get_all_to_order

LOGGER = logging_tools.getLogger(__name__)

def _get_resonance_lines(order):
    resonances = {'X': [], 
                  'Y': [], 
                  'Z': [(1, 0, 1), (0, 1, 1), (1, 0, -1), (0, 1, -1)]}
    # Get all the rdts up to a given order
    fterms = get_all_to_order(order)
    # Some rdts can't be seen depending on the plane, filter them
    for (j,k,l,m) in fterms:
        if j != 0:
            resonances['X'].append((1-j+k, m-l, 0))
        if l != 0:
            resonances['Y'].append((k-j, 1-l+m, 0))
    return resonances


MAIN_LINES = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}
Z_TOLERANCE = 0.0003


def estimate_tunes(harpy_input, usvs):
    """
    Estimates the tunes from the `FFT` of decomposed data.

    Args:
        harpy_input: Analysis settings.
        usvs: dictionary per plane of (U and SV^T matrices).

    Returns:
        list of estimated tunes [x, y, z].
    """
    tunex = _estimate_tune(usvs["X"][1], harpy_input.window, q_s=False)
    tuney = _estimate_tune(usvs["Y"][1], harpy_input.window, q_s=False)
    if harpy_input.autotunes == "transverse":
        return [tunex, tuney, 0]
    tunez = _estimate_tune(usvs["X"][1], harpy_input.window, q_s=True)
    return [tunex, tuney, tunez]


def _estimate_tune(sv_mat, window, q_s=False):
    upper_power = int(np.power(2, np.ceil(np.log2(sv_mat.shape[1]))))
    synchro_limit = int(upper_power / 16)  # 0.03125: synchrotron tunes are lower, betatron higher
    coefs = np.abs(np.fft.rfft(np.mean(sv_mat, axis=0) * windowing(sv_mat.shape[1], window),
                               n=2 * upper_power))
    if q_s:
        return np.fft.rfftfreq(2 * upper_power)[np.argmax(coefs[:synchro_limit])]
    return np.fft.rfftfreq(2 * upper_power)[synchro_limit + np.argmax(coefs[synchro_limit:])]


def harpy_per_plane(harpy_input, bpm_matrix, usv, tunes, plane):
    """
    Calculates the spectra of TbT data, finds the main lines and cleans the BPMs for which the
    main line hasn't been found or is too far from its average over BPMs.

    Args:
        harpy_input: Analysis settings.
        bpm_matrix: TbT BPM matrix.
        usv: U and SV^T matrices decomposed matrices, can be ``None``.
        tunes: list of tunes [x, y, z].
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        A tuple of DataFrame (containing the main lines), Spectra, and Bad BPMs summary.
    """
    df = pd.DataFrame(index=bpm_matrix.index)
    frequencies, coefficients = windowed_padded_rfft(harpy_input, bpm_matrix, tunes, usv)
    df, not_tune_bpms = _get_main_resonances(tunes, dict(FREQS=frequencies, COEFFS=coefficients),
                                                plane, harpy_input.tolerance, df)
    cleaned_by_tune_bpms = clean_by_tune(df.loc[:, f"{COL_TUNE}{plane}"], harpy_input.tune_clean_limit)
    df = df.loc[df.index.difference(cleaned_by_tune_bpms)]

    df[f"{COL_MU}{plane}"] = _realign_phases(df.loc[:, f"{COL_MU}{plane}"].to_numpy(),
                                          df.loc[:, f"{COL_TUNE}{plane}"].to_numpy(), bpm_matrix.shape[1])

    bad_bpms_summaries = _get_bad_bpms_summary(not_tune_bpms, cleaned_by_tune_bpms)
    bpm_matrix = bpm_matrix.loc[df.index]
    spectra = dict(FREQS=frequencies.loc[df.index], COEFFS=coefficients.loc[df.index])

    if _get_natural_tunes(harpy_input, tunes) is not None:
        df_nattunes = _calculate_natural_tunes(
            spectra, _get_natural_tunes(harpy_input, tunes), harpy_input.tolerance, plane
        )
        df = pd.concat([df, df_nattunes], axis=1, sort=False)

        df[f"{COL_NATMU}{plane}"] = _realign_phases(df.loc[:, f"{COL_NATMU}{plane}"].to_numpy(),
                                                 df.loc[:, f"{COL_NATTUNE}{plane}"].to_numpy(),

                                                 bpm_matrix.shape[1])
    if tunes[2] > 0:
        df, _ = _get_main_resonances(tunes, spectra, "Z", Z_TOLERANCE, df)
        df[f"{COL_MU}Z"] = _realign_phases(
            df.loc[:, f"{COL_MU}Z"].to_numpy(),
            df.loc[:, f"{COL_TUNE}Z"].to_numpy(),
            bpm_matrix.shape[1],
        )
    return df, spectra, bad_bpms_summaries


def find_resonances(tunes, nturns, plane, spectra, order_resonances):
    """
    Finds higher order lines in the spectra.

    Args:
        tunes: list of tunes [x, y, z].
        nturns: length of analysed data.
        plane: marking the horizontal or vertical plane, **X** or **Y**.
        spectra: frequencies and complex coefficients.

    Returns:
        A DataFrame.
    """
    resonance_lines = _get_resonance_lines(order_resonances)

    df = pd.DataFrame(index=spectra["FREQS"].index, dtype=pd.Float64Dtype())
    resonances_freqs = _compute_resonances_with_freqs(plane, tunes, resonance_lines)
    if tunes[2] > 0.0:
        resonances_freqs.update(_compute_resonances_with_freqs("Z", tunes, resonance_lines))
    for resonance in resonances_freqs.keys():
        tolerance = _get_resonance_tolerance(resonance, nturns)
        max_coefs, max_freqs = _search_highest_coefs(resonances_freqs[resonance], tolerance,
                                                     spectra["FREQS"], spectra["COEFFS"])
        resstr = _get_resonance_suffix(resonance)
        columns = [f"{COL_FREQ}{resstr}", f"{COL_AMP}{resstr}", f"{COL_PHASE}{resstr}"]
        df_resonance = _get_freqs_amps_phases(max_freqs, max_coefs, resonances_freqs[resonance])
        df_resonance.columns = columns
        df.loc[:, columns] = df_resonance

        df.loc[:, f"{COL_PHASE}{resstr}"] = _realign_phases(df.loc[:, f"{COL_PHASE}{resstr}"].to_numpy(),
                                               df.loc[:, f"{COL_FREQ}{resstr}"].to_numpy(), nturns)

    return df


def _get_main_resonances(tunes, spectra, plane, tolerance, df):
    freq = sum(r * t for r, t in zip(tunes, MAIN_LINES[plane])) % 1
    max_coefs, max_freqs = _search_highest_coefs(freq, tolerance, spectra["FREQS"], spectra["COEFFS"])
    if not np.any(max_coefs) and plane != "Z":
        raise ValueError(f"No main {plane} resonances found, "
                         f"try to increase the tolerance or adjust the tunes")
    bad_bpms_by_tune = spectra["COEFFS"].loc[max_coefs == 0.].index
    columns = [f"{COL_TUNE}{plane}", f"{COL_AMP}{plane}", f"{COL_MU}{plane}"]
    df_main = _get_freqs_amps_phases(max_freqs, max_coefs, freq)
    df_main.columns = columns
    df.loc[:, columns] = df_main
    if plane != "Z":
        df = df.loc[df.index.difference(bad_bpms_by_tune)]
    return df, bad_bpms_by_tune


def _calculate_natural_tunes(spectra, nattunes, tolerance, plane):
    columns = [f"{COL_NATTUNE}{plane}", f"{COL_NATAMP}{plane}", f"{COL_NATMU}{plane}"]
    x, y, _ = nattunes
    freq = x % 1 if plane == "X" else y % 1
    max_coefs, max_freqs = _search_highest_coefs(freq, tolerance, spectra["FREQS"], spectra["COEFFS"])
    df = _get_freqs_amps_phases(max_freqs, max_coefs, freq)
    df.columns = columns
    return df


def _get_freqs_amps_phases(max_freqs: pd.Series, max_coefs: pd.Series, freq: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{COL_FREQ}": max_freqs,
            f"{COL_AMP}": np.abs(max_coefs),
            f"{COL_PHASE}": np.sign(0.5 - freq) * np.angle(max_coefs) / PI2,
        },
        dtype=pd.Float64Dtype(),
    )


def _realign_phases(phase_data, freq_data, nturns):
    mid_phase = (phase_data + freq_data * nturns / 2) % 1
    return np.where(np.abs(mid_phase) > 0.5, mid_phase - np.sign(mid_phase), mid_phase)


def clean_by_tune(tunes: pd.Series, tune_clean_limit: Number) -> pd.Series:
    """
    Looks for outliers in the tunes pandas Series and returns their indices.

    Args:
        tunes (pd.Series): Pandas series with the tunes per BPM and the BPM names as index.
        tune_clean_limit: No BPM will find as outlier if its distance to the
            average is lower than this limit.
    """
    bad_bpms_mask = outliers.get_filter_mask(tunes, limit=tune_clean_limit)  # returns ``True`` for good BPMs
    bad_bpms_names = tunes[~bad_bpms_mask].index
    return bad_bpms_names


def _get_bad_bpms_summary(not_tune_bpms, cleaned_by_tune_bpms):
    return ([f"{bpm_name} The main resonance has not been found" for bpm_name in not_tune_bpms] +
            [f"{bpm_name} tune is too far from average" for bpm_name in cleaned_by_tune_bpms])


def _search_highest_coefs(freq, tolerance, frequencies, coefficients):
    """
    Finds the highest coefficients in frequencies/coefficients in freq +- tolerance.

    Args:
        freq: frequency from interval (0, 1).
        tolerance:
        frequencies:
        coefficients:

    Returns:
        Tuple of maximum coefficients and the corresponding frequencies.
    """
    p_freq = freq if freq < 0.5 else 1 - freq
    min_val, max_val = p_freq - tolerance, p_freq + tolerance
    freq_vals = frequencies.to_numpy()
    coefs_vals = coefficients.to_numpy()
    on_window_mask = (freq_vals >= min_val) & (freq_vals <= max_val)
    filtered_coefs = np.where(on_window_mask, coefs_vals, 0)
    filtered_amps = np.abs(filtered_coefs)
    max_indices = np.argmax(filtered_amps, axis=1)
    max_coefs = filtered_coefs[np.arange(coefs_vals.shape[0]), max_indices]
    max_coefs = pd.Series(index=coefficients.index, data=max_coefs)
    max_pfreqs = freq_vals[np.arange(freq_vals.shape[0]), max_indices]
    max_freqs = max_pfreqs if freq < 0.5 else 1 - max_pfreqs
    max_freqs = pd.Series(index=coefficients.index, data=np.where(max_coefs != 0, max_freqs, 0))
    return max_coefs, max_freqs


def _get_resonance_suffix(resonance):
    x, y, z = resonance
    return f"{x}{y}{z if z else ''}".replace("-", "_")


def _compute_resonances_with_freqs(plane, tunes, resonance_lines):
    """
    Computes the frequencies in [0, 1) for all the resonances listed in the ``RESONANCE_LISTS``,
    together with the natural tunes frequencies if given.
    """
    freqs = [sum(r * t for r, t in zip(tunes, resonance)) % 1 for resonance in resonance_lines[plane]]
    return dict(zip(resonance_lines[plane], freqs))


def _compute_resonance_freqs(plane, tunes, resonance_lines):
    return [sum(r * t for r, t in zip(tunes, resonance)) % 1 for resonance in resonance_lines[plane]]


def _get_resonance_tolerance(resonance, n_turns):
    x, y, z = resonance
    return (abs(x) + abs(y) + bool(z) * (abs(z) - 1)) * max(1e-4, 1 / n_turns)


def windowed_padded_rfft(harpy_input, matrix, tunes, svd=None):
    """
    Calculates the spectra using specified windowing function and zero-padding.

    Args:
        harpy_input: A `HarpyInput` object.
        matrix: `pd.DataFrame` of TbT matrix (BPMs x turns).
        tunes: list of tunes [x, y, z].
        svd: reduced (U_matrix, np.dot(S_matrix, V_matrix)) of original TbT matrix, defaults to
            ``None``.

    Returns:
        Tuple of `pd.DataFrames`, for frequencies and coefficients.
    """
    padded_len, output_len = np.power(2, harpy_input.turn_bits), np.power(2, harpy_input.output_bits)
    sub_bins = int(padded_len / output_len)
    mask = get_freq_mask(harpy_input, tunes, 2 / matrix.shape[1])
    n_bins = int(np.sum(mask) / sub_bins)
    n_bpms = len(matrix.index)
    if svd is None:
        tbt_matrix = matrix.loc[:, :].to_numpy()
        coefs = np.fft.rfft(tbt_matrix * windowing(tbt_matrix.shape[1], window=harpy_input.window),
                            n=padded_len * 2)[:, mask]
    else:
        u, sv = svd
        s_vt_freq = np.fft.rfft(sv * windowing(sv.shape[1], window=harpy_input.window),
                                n=padded_len * 2)
        coefs = np.dot(u, s_vt_freq[:, mask])
    argsmax = (np.indices((n_bpms, n_bins))[1] * sub_bins +
               np.argmax(np.abs(np.reshape(coefs, (n_bpms, n_bins, sub_bins))), axis=2))
    # two 2 in following line is because we have just half of spectra
    coefficients = pd.DataFrame(index=matrix.index, data=2 * coefs[np.arange(n_bpms)[:, None], argsmax])
    del coefs
    frequencies = pd.DataFrame(index=matrix.index, data=np.outer(np.ones(n_bpms), np.fft.rfftfreq(
            padded_len * 2)[mask])[np.arange(n_bpms)[:, None], argsmax])
    return frequencies, coefficients


def windowing(length, window='hamming'):
    """
    Provides specified windowing function of given length.

    Currently, the following windowing functions are implemented (sorted by increasing width of
    main lobe, also decreasing spectral leakage in closest lobes):
    ``rectangle``, ``welch``, ``triangle``, ``hann``, ``hamming``, ``nuttal3``, and ``nuttal4``.

    Args:
        length: length of the window.
        window: type of the windowing function.

    Returns:
        Normalised windowing function of specified type and length.
    """
    ints2pi = PI2 * np.arange(length) / (length - 1)
    windows = {
        "nuttal4": 0.3125 - 0.46875 * np.cos(ints2pi) + 0.1875 * np.cos(2 * ints2pi) - 0.03125 * np.cos(3 * ints2pi),
        "nuttal3": 0.375 - 0.5 * np.cos(ints2pi) + 0.125 * np.cos(2 * ints2pi),
        "hamming": (25 / 46) - (21 / 46) * np.cos(ints2pi),
        "hann": 0.5 - 0.5 * np.cos(ints2pi),
        "welch": 1 - np.square((ints2pi / np.pi) - 1),
        "triangle": 1 - np.abs((ints2pi / np.pi) - 1),
        "rectangle": np.ones(length)
    }
    if window not in windows.keys():
        raise NotImplementedError(f"Unknown windowing function {window}")
    return windows[window] / np.sum(windows[window])


def get_freq_mask(harpy_input, tunes, auto_tol):
    """
    Computes mask to get intervals around resonances in frequency domain.

    Args:
        harpy_input: A `HarpyInput` object.
        tunes: list of tunes [x, y, z].
        auto_tol: automatically calculated tolerance.

    Returns:
        Boolean array.
    """
    if "full_spectra" in harpy_input.to_write:
        mask = np.ones(int(np.power(2, harpy_input.turn_bits)) + 1, dtype=bool)
        mask[-1] = False
        return mask
    mask = np.zeros(int(np.power(2, harpy_input.turn_bits)) + 1, dtype=bool)
    nattunes = _get_natural_tunes(harpy_input, tunes)
    if nattunes is not None:
        mask = _get_partial_freq_mask(harpy_input, mask, list(nattunes), harpy_input.tolerance)
    tol = harpy_input.tolerance if harpy_input.autotunes is None else auto_tol
    freqs = (list(tunes))
    
    resonance_lines = _get_resonance_lines(harpy_input.resonances)
    for plane in PLANES:
        freqs.extend(_compute_resonance_freqs(plane, tunes, resonance_lines))
    if tunes[2]:
        freqs.extend(_compute_resonance_freqs("Z", tunes, resonance_lines))
    return _get_partial_freq_mask(harpy_input, mask, freqs, tol)


def _get_natural_tunes(harpy_input, tunes):
    if harpy_input.natdeltas is not None:
        return [r + t if t != 0 else 0 for r, t in zip(tunes, harpy_input.natdeltas)]
    if harpy_input.nattunes is not None:
        return harpy_input.nattunes
    return None


def _get_partial_freq_mask(harpy_input, start_mask, frequencies, tolerance):
    bins = np.power(2, harpy_input.output_bits)
    sub_bins = np.power(2, harpy_input.turn_bits - harpy_input.output_bits)
    mask = start_mask[:]
    freqs = np.array(frequencies) % 1
    freqs = np.where(freqs > 0.5, 1 - freqs, freqs)
    for freq in freqs:
        if freq == 0.0:
            continue
        mask[int(np.floor(max(0, freq - tolerance) * 2 * bins) * sub_bins):
             int(np.ceil(min(0.5, freq + tolerance) * 2 * bins) * sub_bins)] = True
    mask[-1] = False  # 2^n + 1 is undesired
    return mask
