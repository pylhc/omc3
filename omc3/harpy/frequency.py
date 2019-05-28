"""
frequency
---------------------------

Calculates the frequency spectra of turn-by-turn data.
Uses a combination of SVD decomposition zero_padded fft to speed up the analysis
Also searches of resonances in the calculated spectra.
"""
from collections import OrderedDict
import numpy as np
import pandas as pd
from utils import outliers, logging_tools
from harpy import kicker
LOGGER = logging_tools.getLogger(__name__)
PI2I = 2 * np.pi * complex(0, 1)

RESONANCES = {
    "X": ((2, 0, 0), (3, 0, 0), (0, 1, 0), (0, 2, 0), (1, 1, 0), (1, -1, 0),
          (1, 2, 0), (1, -2, 0), (1, -3, 0), (2, -2, 0), (2, -1, 0)),
    "Y": ((0, 2, 0), (0, 3, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0), (1, -1, 0),
          (1, -2, 0), (1, -3, 0), (2, -1, 0), (2, 1, 0)),
    "Z": ((1, 0, 1), (0, 1, 1), (1, 0, -1), (0, 1, -1))
}

MAIN_LINES = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}
Z_TOLERANCE = 0.0003
PLANES = ("X", "Y")


def estimate_tunes(harpy_input, usvs):
    """
    Estimates the tunes from FFT of decomposed data

    Args:
        harpy_input: Analysis settings
        usvs: dictionary per plane of (U and SV^T matrices)

    Returns:
        list of estimated tunes [x, y, z]
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
    Calculates spectra of TbT data, finds the main lines and cleans the BPMs,
    for which the main line hasn't been found or is too far from its average over BPMs

    Args:
        harpy_input: Analysis settings
        bpm_matrix: TbT BPM matrix
        usv: U and SV^T matrices decomposed matrices, can be None
        tunes: list of tunes [x, y, z]
        plane: "X" or "Y" marking the horizontal or vertical plane

    Returns:
        DataFrame, Spectra, Bad BPMs summary
    """
    panda = pd.DataFrame(index=bpm_matrix.index, columns=OrderedDict())
    frequencies, coefficients = windowed_padded_rfft(harpy_input, bpm_matrix, tunes, usv)
    panda, not_tune_bpms = _get_main_resonances(tunes, dict(FREQS=frequencies, COEFFS=coefficients),
                                                plane, harpy_input.tolerance, panda)
    cleaned_by_tune_bpms = clean_by_tune(panda.loc[:, f"TUNE{plane}"], harpy_input.tune_clean_limit)
    panda = panda.loc[panda.index.difference(cleaned_by_tune_bpms)]
    panda[f"MU{plane}"] = _realign_phases(panda.loc[:, f"MU{plane}"].values,
                                          panda.loc[:, f"TUNE{plane}"].values, bpm_matrix.shape[1])
    bad_bpms_summaries = _get_bad_bpms_summary(not_tune_bpms, cleaned_by_tune_bpms)
    bpm_matrix = bpm_matrix.loc[panda.index]
    spectra = dict(FREQS=frequencies.loc[panda.index], COEFFS=coefficients.loc[panda.index])

    if harpy_input.is_free_kick:
        panda = kicker.phase_correction(bpm_matrix, panda, plane)
    if _get_natural_tunes(harpy_input, tunes) is not None:
        panda = panda.join(_calculate_natural_tunes(spectra, _get_natural_tunes(harpy_input, tunes),
                                                    harpy_input.tolerance, plane))
        panda[f"NATMU{plane}"] = _realign_phases(panda.loc[:, f"NATMU{plane}"].values,
                                                 panda.loc[:, f"NATTUNE{plane}"].values,
                                                 bpm_matrix.shape[1])
    if tunes[2] > 0:
        panda, _ = _get_main_resonances(tunes, spectra, "Z", Z_TOLERANCE, panda)
    return panda, spectra, bad_bpms_summaries


def find_resonances(tunes, nturns, plane, spectra):
    """
    Finds higher order lines in the spectra

    Args:
        tunes: list of tunes [x, y, z]
        nturns: length of analysed data
        plane: "X" or "Y" marking the horizontal or vertical plane
        spectra: frequencies and complex coefficients

    Returns:
        DataFrame
    """
    df = pd.DataFrame(index=spectra["FREQS"].index, columns=OrderedDict())
    resonances_freqs = _compute_resonances_with_freqs(plane, tunes)
    if tunes[2] > 0.0:
        resonances_freqs.update(_compute_resonances_with_freqs("Z", tunes))
    for resonance in resonances_freqs.keys():
        tolerance = _get_resonance_tolerance(resonance, nturns)
        max_coefs, max_freqs = _search_highest_coefs(resonances_freqs[resonance], tolerance,
                                                     spectra["FREQS"], spectra["COEFFS"])
        resstr = _get_resonance_suffix(resonance)
        df[f"FREQ{resstr}"], df[f"AMP{resstr}"], df[f"PHASE{resstr}"] = _get_freqs_amps_phases(
            max_freqs, max_coefs, resonances_freqs[resonance])
        df[f"PHASE{resstr}"] = _realign_phases(df.loc[:, f"PHASE{resstr}"].values,
                                               df.loc[:, f"FREQ{resstr}"].values, nturns)
    return df


def _get_main_resonances(tunes, spectra, plane, tolerance, df):
    freq = sum(r * t for r, t in zip(tunes, MAIN_LINES[plane])) % 1
    max_coefs, max_freqs = _search_highest_coefs(freq, tolerance, spectra["FREQS"], spectra["COEFFS"])
    if not np.any(max_coefs) and plane != "Z":
        raise ValueError(f"No main {plane} resonances found, "
                         f"try to increase the tolerance or adjust the tunes")
    bad_bpms_by_tune = spectra["COEFFS"].loc[max_coefs == 0.].index
    df[f"TUNE{plane}"], df[f"AMP{plane}"], df[f"MU{plane}"] = _get_freqs_amps_phases(
        max_freqs, max_coefs, freq)
    if plane != "Z":
        df = df.loc[df.index.difference(bad_bpms_by_tune)]
    return df, bad_bpms_by_tune


def _calculate_natural_tunes(spectra, nattunes, tolerance, plane):
    df = pd.DataFrame(index=spectra["FREQS"].index, columns=OrderedDict())
    x, y, _ = nattunes
    freq = x % 1 if plane == "X" else y % 1
    max_coefs, max_freqs = _search_highest_coefs(freq, tolerance, spectra["FREQS"], spectra["COEFFS"])
    df[f"NATTUNE{plane}"], df[f"NATAMP{plane}"], df[f"NATMU{plane}"] = _get_freqs_amps_phases(
        max_freqs, max_coefs, freq)
    return df


def _get_freqs_amps_phases(max_freqs, max_coefs, freq):
    return max_freqs, np.abs(max_coefs), np.sign(0.5 - freq) * np.angle(max_coefs) / (2 * np.pi)


def _realign_phases(phase_data, freq_data, nturns):
    mid_phase = (phase_data + freq_data * nturns / 2) % 1
    return np.where(np.abs(mid_phase) > 0.5, mid_phase - np.sign(mid_phase), mid_phase)


def clean_by_tune(tunes, tune_clean_limit):
    """
    This function looks for outliers in the tunes pandas Series and returns
    their indices.

    Args:
        tunes: Pandas series with the tunes per BPM and the BPM names as
            index.
        tune_clean_limit: No BPM will find as outlier if its distance to the
            average is lower than this limit.
    """
    bad_bpms_mask = outliers.get_filter_mask(tunes, limit=tune_clean_limit)
    bad_bpms_names = tunes[~bad_bpms_mask].index
    return bad_bpms_names


def _get_bad_bpms_summary(not_tune_bpms, cleaned_by_tune_bpms):
    return ([f"{bpm_name} The main resonance has not been found" for bpm_name in not_tune_bpms] +
            [f"{bpm_name} tune is too far from average" for bpm_name in cleaned_by_tune_bpms])


def _search_highest_coefs(freq, tolerance, frequencies, coefficients):
    """
    Finds highest coefficients in frequencies/coefficients in freq +- tolerance.

    Args:
        freq: frequency from interval (0, 1)
        tolerance:
        frequencies:
        coefficients:

    Returns:
        Tupel of maximum coefficients and corresponding frequencies

    """
    p_freq = freq if freq < 0.5 else 1 - freq
    min_val, max_val = p_freq - tolerance, p_freq + tolerance
    freq_vals = frequencies.values
    coefs_vals = coefficients.values
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


def _compute_resonances_with_freqs(plane, tunes):
    """
    Computes the frequencies in [0, 1) for all the resonances listed in the RESONANCE_LISTS,
    together with the natural tunes frequencies if given.
    """
    freqs = [sum(r * t for r, t in zip(tunes, resonance)) % 1 for resonance in RESONANCES[plane]]
    return dict(zip(RESONANCES[plane], freqs))


def _compute_resonance_freqs(plane, tunes):
    return [sum(r * t for r, t in zip(tunes, resonance)) % 1 for resonance in RESONANCES[plane]]


def _get_resonance_tolerance(resonance, n_turns):
    x, y, z = resonance
    return (abs(x) + abs(y) + bool(z) * (abs(z) - 1)) * max(1e-4, 1 / n_turns)


def windowed_padded_rfft(harpy_input, matrix, tunes, svd=None):
    """
    Calculates the spectra using specified windowing function and zero-padding

    Args:
        harpy_input: HarpyInput object
        matrix: pd.DataFrame of TbT matrix (BPMs x turns)
        tunes: list of tunes [x, y, z]
        svd: reduced (U_matrix, np.dot(S_matrix, V_matrix)) of original TbT matrix, default None

    Returns:
        DataFrames (tupel): frequencies, coefficients

    """
    padded_len, output_len = np.power(2, harpy_input.turn_bits), np.power(2, harpy_input.output_bits)
    sub_bins = int(padded_len / output_len)
    mask = get_freq_mask(harpy_input, tunes, 2 / matrix.shape[1])
    n_bins = int(np.sum(mask) / sub_bins)
    n_bpms = len(matrix.index)
    if svd is None:
        tbt_matrix = matrix.loc[:, :].values
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

    Currently, the following windowing functions are implemented
    (sorted by increasing width of main lobe, also decreasing spectral leakage in closest lobes):
    ``rectangle``, ``welch``, ``trinagle``, ``hann``, ``hamming``, ``nuttal3``, ``nuttal4``

    Args:
        length: length of the window
        window: type of the windowing function

    Returns:
        Normalised windowing function of specified type and length
    """
    ints2pi = 2 * np.pi * np.arange(length) / (length - 1)
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
    Computes mask to get intervals around resonances in frequency domain

    Args:
        harpy_input: HarpyInput object
        tunes: list of tunes [x, y, z]
        auto_tol: automatically calculated tolerance

    Returns:
        Boolean array

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
    for plane in PLANES:
        freqs.extend(_compute_resonance_freqs(plane, tunes))
    if tunes[2]:
        freqs.extend(_compute_resonance_freqs("Z", tunes))
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
