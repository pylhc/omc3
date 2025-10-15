"""
Frequency
---------

This module contains the frequency calculations related functionality of ``harpy``.

It provides functions to calculate the frequency spectra of turn-by-turn data.
This calculation is done using a combination of SVD decomposition zero_padded `fft` to speed up the
analysis.
Also searches for resonances in the calculated spectra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from optics_functions.rdt import get_all_to_order

from omc3.definitions.constants import PI2, PLANES
from omc3.harpy.constants import (
    COL_AMP,
    COL_FREQ,
    COL_MU,
    COL_NATAMP,
    COL_NATMU,
    COL_NATTUNE,
    COL_PHASE,
    COL_TUNE,
)
from omc3.utils import logging_tools, outliers

if TYPE_CHECKING:
    from numbers import Number

    from generic_parser import DotDict

LOGGER = logging_tools.getLogger(__name__)


def _get_resonance_lines(order: int) -> dict[str, list[tuple[int, int, int]]]:
    resonances = {
        "X": [],
        "Y": [],
        "Z": [(1, 0, 1), (0, 1, 1), (1, 0, -1), (0, 1, -1)],
    }
    # Get all the rdts up to a given order
    fterms = get_all_to_order(order)
    # Some rdts can't be seen depending on the plane, filter them
    for j, k, l, m in fterms:  # noqa: E741 (these variable names are ok)
        if j != 0:
            resonances["X"].append((1 - j + k, m - l, 0))
        if l != 0:
            resonances["Y"].append((k - j, 1 - l + m, 0))
    return resonances


MAIN_LINES = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}
Z_TOLERANCE = 0.0003


def estimate_tunes(
    harpy_input: DotDict, usvs: dict[str, tuple[pd.DataFrame, np.ndarray]]
) -> list[float]:
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


def _estimate_tune(sv_mat: np.ndarray, window: str, q_s: bool = False) -> float:
    upper_power = int(np.power(2, np.ceil(np.log2(sv_mat.shape[1]))))
    synchro_limit = int(upper_power / 16)  # 0.03125: synchrotron tunes are lower, betatron higher

    windowed_data = np.mean(sv_mat, axis=0) * windowing(sv_mat.shape[1], window)
    coefs = np.abs(np.fft.rfft(windowed_data, n=2 * upper_power))
    freqs = np.fft.rfftfreq(2 * upper_power)
    if q_s:
        max_index = np.argmax(coefs[:synchro_limit])
    else:
        max_index = synchro_limit + np.argmax(coefs[synchro_limit:])

    return freqs[max_index]


def harpy_per_plane(
    harpy_input: DotDict,
    bpm_matrix: pd.DataFrame,
    usv: tuple[pd.DataFrame, np.ndarray],
    tunes: list[float],
    plane: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
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
    harpy_results = pd.DataFrame(index=bpm_matrix.index)
    frequencies, coefficients = windowed_padded_rfft(harpy_input, bpm_matrix, tunes, usv)
    harpy_results, not_tune_bpms = _get_main_resonances(
        tunes,
        {"FREQS": frequencies, "COEFFS": coefficients},
        plane,
        harpy_input.tolerance,
        harpy_results,
    )
    cleaned_by_tune_bpms = clean_by_tune(
        harpy_results.loc[:, f"{COL_TUNE}{plane}"], harpy_input.tune_clean_limit
    )
    harpy_results = harpy_results.loc[harpy_results.index.difference(cleaned_by_tune_bpms)]

    harpy_results = _realign_phases(
        harpy_results, f"{COL_MU}{plane}", f"{COL_TUNE}{plane}", bpm_matrix.shape[1]
    )

    bad_bpms_summaries = _get_bad_bpms_summary(not_tune_bpms, cleaned_by_tune_bpms)
    bpm_matrix = bpm_matrix.loc[harpy_results.index]
    spectra = {
        "FREQS": frequencies.loc[harpy_results.index],
        "COEFFS": coefficients.loc[harpy_results.index],
    }

    tune_tol = harpy_input.tolerance
    if (nattunes := _get_natural_tunes(harpy_input, tunes)) is not None:
        # Each plane that has 0 for the nattunes is ignored.
        if any(
            abs(tune - nattune) < tune_tol for nattune, tune in zip(nattunes, tunes) if nattune != 0
        ):
            raise ValueError(
                "At least one of the driven tunes is within the tolerance window of finding the natural tunes. "
                "Please check the input parameters."
            )
        df_nattunes = _calculate_natural_tunes(spectra, nattunes, tune_tol, plane)
        harpy_results = pd.concat([harpy_results, df_nattunes], axis=1, sort=False)

        harpy_results = _realign_phases(
            harpy_results,
            f"{COL_NATMU}{plane}",
            f"{COL_NATTUNE}{plane}",
            bpm_matrix.shape[1],
        )
    if tunes[2] > 0:
        harpy_results, _ = _get_main_resonances(tunes, spectra, "Z", Z_TOLERANCE, harpy_results)
        harpy_results = _realign_phases(
            harpy_results, f"{COL_MU}Z", f"{COL_TUNE}Z", bpm_matrix.shape[1]
        )
    return harpy_results, spectra, bad_bpms_summaries


def find_resonances(
    tunes: list[float],
    nturns: int,
    plane: str,
    spectra: dict[str, pd.DataFrame],
    order_resonances: int,
) -> pd.DataFrame:
    """
    Finds higher order lines in the spectra.

    Args:
        tunes: list of tunes [x, y, z].
        nturns: length of analysed data.
        plane: marking the horizontal or vertical plane, **X** or **Y**.
        spectra: frequencies and complex coefficients.
        order_resonances: highest order of resonances to be found.

    Returns:
        A DataFrame with the found resonances. The DataFrame has BPM names as index and
        columns for each resonance in the format '{resonance}_FREQ', '{resonance}_AMP',
        and '{resonance}_PHASE', where {resonance} is the resonance identifier (e.g., '10'
        for the (1,0,0) resonance).
        Each row contains the frequency, amplitude, and phase of the corresponding resonance
        for that BPM.
    """
    resonance_lines = _get_resonance_lines(order_resonances)

    resonance_df = pd.DataFrame(index=spectra["FREQS"].index, dtype=pd.Float64Dtype())
    resonances_freqs = _compute_resonances_with_freqs(plane, tunes, resonance_lines)
    if tunes[2] > 0.0:
        resonances_freqs.update(_compute_resonances_with_freqs("Z", tunes, resonance_lines))
    for resonance in resonances_freqs:
        tolerance = _get_resonance_tolerance(resonance, nturns)
        max_coefs, max_freqs = _search_highest_coefs(
            resonances_freqs[resonance], tolerance, spectra["FREQS"], spectra["COEFFS"]
        )

        # Add the frequencies, amplitudes and phases to the DataFrame
        resstr = _get_resonance_suffix(resonance)
        columns = [f"{COL_FREQ}{resstr}", f"{COL_AMP}{resstr}", f"{COL_PHASE}{resstr}"]
        resonance_df.loc[:, columns] = _get_freqs_amps_phases(
            max_freqs, max_coefs, resonances_freqs[resonance], columns
        )

        resonance_df = _realign_phases(
            resonance_df, f"{COL_PHASE}{resstr}", f"{COL_FREQ}{resstr}", nturns
        )

    return resonance_df


def _get_main_resonances(
    tunes: list[float],
    spectra: dict[str, pd.DataFrame],
    plane: str,
    tolerance: float,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Index]:
    # Find all the main resonances
    freq = tunes["XYZ".index(plane)] % 1

    # Find the main resonance, amplitude and phase within tolerance for this tune
    max_coefs, max_freqs = _search_highest_coefs(
        freq, tolerance, spectra["FREQS"], spectra["COEFFS"]
    )
    if not np.any(max_coefs) and plane != "Z":
        raise ValueError(
            f"No main {plane} resonances found, try to increase the tolerance or adjust the tunes"
        )
    bad_bpms_by_tune = spectra["COEFFS"].loc[max_coefs == 0.0].index

    # Add the frequencies, amplitudes and phases to the DataFrame
    columns = [f"{COL_TUNE}{plane}", f"{COL_AMP}{plane}", f"{COL_MU}{plane}"]
    df.loc[:, columns] = _get_freqs_amps_phases(max_freqs, max_coefs, freq, columns)

    if plane != "Z":
        df = df.loc[df.index.difference(bad_bpms_by_tune)]
    return df, bad_bpms_by_tune


def _calculate_natural_tunes(
    spectra: dict[str, pd.DataFrame], nattunes: list[float], tolerance: float, plane: str
) -> pd.DataFrame:
    """
    Calculates the natural tunes, amplitudes, and phases for the given plane.

    Args:
        spectra: Dictionary containing 'FREQS' and 'COEFFS' DataFrames with frequency and coefficient data.
        nattunes: List of natural tunes [x, y, z], where the relevant tune is selected based on the plane.
        tolerance: Tolerance value for searching the highest coefficients around the natural tune frequency.
        plane: The plane for which to calculate natural tunes ('X' or 'Y').

    Returns:
        A DataFrame containing the natural tune frequencies, amplitudes, and phases for each BPM.
        The columns are named according to the plane (e.g., 'NATTUNEX', 'NATAMPX', 'NATMUX').
    """
    # Get the relavant fractional tune
    freq = nattunes["XY".index(plane)] % 1

    # Find the natural tune, amplitude and phase
    max_coefs, max_freqs = _search_highest_coefs(
        freq, tolerance, spectra["FREQS"], spectra["COEFFS"]
    )

    # Add the frequencies, amplitudes and phases to the DataFrame
    columns = [f"{COL_NATTUNE}{plane}", f"{COL_NATAMP}{plane}", f"{COL_NATMU}{plane}"]
    return _get_freqs_amps_phases(max_freqs, max_coefs, freq, columns)


def _get_freqs_amps_phases(
    max_freqs: pd.Series, max_coefs: pd.Series, freq: float, columns: list[str]
) -> pd.DataFrame:
    """
    Creates a DataFrame with frequencies, amplitudes, and phases from the given series.

    Args:
        max_freqs: Series of maximum frequencies found for each BPM.
        max_coefs: Series of maximum coefficients (complex) found for each BPM.
        freq: The target frequency around which the search was performed.
        columns: A list of exactly 3 strings specifying the column names for the DataFrame.
            - columns[0]: The name for the frequency column (e.g., 'TUNEX', 'NATTUNEX').
            - columns[1]: The name for the amplitude column (e.g., 'AMPX', 'NATAMPX').
            - columns[2]: The name for the phase column (e.g., 'MUX', 'NATMUX').

    Returns:
        A DataFrame with the specified columns containing the frequencies, amplitudes, and phases.
        Amplitudes are the absolute values of max_coefs, and phases are adjusted based on the freq.
    """
    amps = np.abs(max_coefs)
    phases = np.sign(0.5 - freq) * np.angle(max_coefs) / PI2
    return pd.DataFrame(
        {
            columns[0]: max_freqs,
            columns[1]: amps,
            columns[2]: phases,
        },
        dtype=pd.Float64Dtype(),
    )


def _realign_phases(df: pd.DataFrame, phase_col: str, tune_col: str, nturns: int) -> pd.DataFrame:
    """
    Realigns the phases in the DataFrame to be within [-0.5, 0.5] by shifting phases outside this range.

    Args:
        df (pd.DataFrame): The DataFrame containing the phase and tune data.
        phase_col (str): The name of the column containing phase data.
        tune_col (str): The name of the column containing tune/frequency data.
        nturns (int): The number of turns, used in the phase calculation.

    Returns:
        pd.DataFrame: A new DataFrame with the realigned phases.
    """
    phase_data = df.loc[:, phase_col]
    freq_data = df.loc[:, tune_col]
    mid_phase = (phase_data + freq_data * nturns / 2) % 1
    realigned_phases = np.where(np.abs(mid_phase) > 0.5, mid_phase - np.sign(mid_phase), mid_phase)
    # Return a new DataFrame to avoid modifying the original one, with the updated phase column
    return df.assign(**{phase_col: realigned_phases})


def clean_by_tune(tunes: pd.Series, tune_clean_limit: Number) -> pd.Series:
    """
    Looks for outliers in the tunes pandas Series and returns their indices.

    Args:
        tunes (pd.Series): Pandas series with the tunes per BPM and the BPM names as index.
        tune_clean_limit: No BPM will find as outlier if its distance to the
            average is lower than this limit.
    """
    bad_bpms_mask = outliers.get_filter_mask(
        tunes, limit=tune_clean_limit
    )  # returns ``True`` for good BPMs
    return tunes[~bad_bpms_mask].index


def _get_bad_bpms_summary(not_tune_bpms: pd.Index, cleaned_by_tune_bpms: pd.Index) -> list[str]:
    return [f"{bpm_name} The main resonance has not been found" for bpm_name in not_tune_bpms] + [
        f"{bpm_name} tune is too far from average" for bpm_name in cleaned_by_tune_bpms
    ]


def _search_highest_coefs(
    freq: float, tolerance: float, frequencies: pd.DataFrame, coefficients: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
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


def _get_resonance_suffix(resonance: tuple[int, int, int]) -> str:
    x, y, z = resonance
    return f"{x}{y}{z if z else ''}".replace("-", "_")


def _compute_resonances_with_freqs(
    plane: str, tunes: list[float], resonance_lines: dict[str, list[tuple[int, int, int]]]
) -> dict[tuple[int, int, int], float]:
    """
    Computes the frequencies in [0, 1) for all the resonances listed in the ``RESONANCE_LISTS``,
    together with the natural tunes frequencies if given.
    """
    return {
        resonance: sum(r * t for r, t in zip(tunes, resonance)) % 1
        for resonance in resonance_lines[plane]
    }


def _compute_resonance_freqs(
    plane: str, tunes: list[float], resonance_lines: dict[str, list[tuple[int, int, int]]]
) -> list[float]:
    return [
        sum(r * t for r, t in zip(tunes, resonance)) % 1 for resonance in resonance_lines[plane]
    ]


def _get_resonance_tolerance(resonance: tuple[int, int, int], n_turns: int) -> float:
    x, y, z = resonance
    return (abs(x) + abs(y) + bool(z) * (abs(z) - 1)) * max(1e-4, 1 / n_turns)


def windowed_padded_rfft(
    harpy_input: DotDict,
    bpm_matrix: pd.DataFrame,
    tunes: list,
    svd: tuple[pd.DataFrame, np.ndarray] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates the spectra using specified windowing function and zero-padding.

    Args:
        harpy_input: A `HarpyInput` object.
        bpm_matrix: `pd.DataFrame` of TbT matrix (BPMs x turns).
        tunes: list of tunes [x, y, z].
        svd: reduced (U_matrix, np.dot(S_matrix, V_matrix)) of original TbT matrix, defaults to
            ``None``.

    Returns:
        Tuple of `pd.DataFrames`, for frequencies and coefficients.
    """
    # Define lengths for padding and output
    padded_len = np.power(2, harpy_input.turn_bits)
    output_len = np.power(2, harpy_input.output_bits)
    sub_bins = int(padded_len / output_len)

    # Get frequency mask
    mask = get_freq_mask(harpy_input, tunes, 2 / bpm_matrix.shape[1])
    n_bins = int(np.sum(mask) / sub_bins)
    n_bpms = len(bpm_matrix.index)

    if svd is None:
        # Perform FFT on the original TbT matrix
        tbt_matrix = bpm_matrix.loc[:, :].to_numpy()
        windowed_matrix = tbt_matrix * windowing(tbt_matrix.shape[1], window=harpy_input.window)
        coefs = np.fft.rfft(windowed_matrix, n=padded_len * 2)[:, mask]
    else:
        # Perform FFT on the SVD-reduced matrix if given
        u, sv = svd
        windowed_sv = sv * windowing(sv.shape[1], window=harpy_input.window)
        s_vt_freq = np.fft.rfft(windowed_sv, n=padded_len * 2)
        coefs = np.dot(u, s_vt_freq[:, mask])

    # Reshape coefficients into bins and sub-bins, then find the index of the maximum amplitude in each sub-bin
    # This identifies the best frequency within each bin for noise reduction
    bin_start_indices = np.indices((n_bpms, n_bins))[1] * sub_bins
    coefs_reshaped = np.reshape(coefs, (n_bpms, n_bins, sub_bins))
    max_sub_bin_indices = np.argmax(np.abs(coefs_reshaped), axis=2)
    selected_indices = bin_start_indices + max_sub_bin_indices
    # Free memory by deleting the reshaped array, as it's no longer needed
    del coefs_reshaped

    # Extract the selected coefficients and scale by 2 (since rfft gives half-spectrum)
    # This gives the complex amplitudes at the selected frequencies
    selected_coefficients = 2 * coefs[np.arange(n_bpms)[:, None], selected_indices]
    coefficients = pd.DataFrame(index=bpm_matrix.index, data=selected_coefficients)
    # Free memory by deleting the original coefs array
    del coefs

    # Compute frequencies corresponding to the selected indices
    # Since the frequency array is the same for all BPMs, compute it once and index into it
    freqs = np.fft.rfftfreq(padded_len * 2)[mask]
    selected_frequencies = freqs[selected_indices]
    frequencies = pd.DataFrame(index=bpm_matrix.index, data=selected_frequencies)

    return frequencies, coefficients


def windowing(length: int, window: str = "hamming") -> np.ndarray:
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
        "nuttal4": 0.3125
        - 0.46875 * np.cos(ints2pi)
        + 0.1875 * np.cos(2 * ints2pi)
        - 0.03125 * np.cos(3 * ints2pi),
        "nuttal3": 0.375 - 0.5 * np.cos(ints2pi) + 0.125 * np.cos(2 * ints2pi),
        "hamming": (25 / 46) - (21 / 46) * np.cos(ints2pi),
        "hann": 0.5 - 0.5 * np.cos(ints2pi),
        "welch": 1 - np.square((ints2pi / np.pi) - 1),
        "triangle": 1 - np.abs((ints2pi / np.pi) - 1),
        "rectangle": np.ones(length),
    }
    if window not in windows:
        raise NotImplementedError(f"Unknown windowing function {window}")
    return windows[window] / np.sum(windows[window])


def get_freq_mask(harpy_input: DotDict, tunes: list, auto_tol: float) -> np.ndarray:
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
    freqs = list(tunes)

    resonance_lines = _get_resonance_lines(harpy_input.resonances)
    for plane in PLANES:
        freqs.extend(_compute_resonance_freqs(plane, tunes, resonance_lines))
    if tunes[2]:
        freqs.extend(_compute_resonance_freqs("Z", tunes, resonance_lines))
    return _get_partial_freq_mask(harpy_input, mask, freqs, tol)


def _get_natural_tunes(harpy_input: DotDict, tunes: list) -> list | None:
    if harpy_input.natdeltas is not None:
        return [r + t if t != 0 else 0 for r, t in zip(tunes, harpy_input.natdeltas)]
    if harpy_input.nattunes is not None:
        return harpy_input.nattunes
    return None


def _get_partial_freq_mask(
    harpy_input: DotDict, start_mask: np.ndarray, frequencies: list, tolerance: float
) -> np.ndarray:
    bins = np.power(2, harpy_input.output_bits)
    sub_bins = np.power(2, harpy_input.turn_bits - harpy_input.output_bits)
    mask = start_mask[:]
    freqs = np.array(frequencies) % 1
    freqs = np.where(freqs > 0.5, 1 - freqs, freqs)
    for freq in freqs:
        if freq == 0.0:
            continue
        start_idx = int(np.floor(max(0, freq - tolerance) * 2 * bins) * sub_bins)
        end_idx = int(np.ceil(min(0.5, freq + tolerance) * 2 * bins) * sub_bins)
        mask[start_idx:end_idx] = True
    mask[-1] = False  # 2^n + 1 is undesired
    return mask
