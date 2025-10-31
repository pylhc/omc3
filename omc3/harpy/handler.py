"""
Handler
-------

This module contains high-level functions to manage most functionality of ``harpy``.
Tools are provided to handle the cleaning, frequency analysis and resonance search for a
single-bunch `TbtData`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs

from omc3.definitions import formats
from omc3.definitions.constants import PLANE_TO_NUM as P2N
from omc3.definitions.constants import PLANES
from omc3.harpy import clean, frequency, kicker
from omc3.harpy.constants import (
    COL_AMP,
    COL_BPM_RES,
    COL_CO,
    COL_COEFFS,
    COL_CORMS,
    COL_ERR,
    COL_FREQS,
    COL_MU,
    COL_NAME,
    COL_NATAMP,
    COL_NATTUNE,
    COL_NOISE,
    COL_NOISE_SCALED,
    COL_PHASE,
    COL_PK2PK,
    COL_S,
    COL_TIME,
    COL_TUNE,
    FILE_AMPS_EXT,
    FILE_FREQS_EXT,
    FILE_LIN_EXT,
    MAINLINE_UNIT,
)
from omc3.utils import logging_tools
from omc3.utils.contexts import timeit

if TYPE_CHECKING:
    from generic_parser import DotDict
    from turn_by_turn import TbtData

LOGGER = logging_tools.get_logger(__name__)
ALL_PLANES = (*PLANES, "Z")
PLANE_TO_NUM = {**P2N, "Z": 3}


def run_per_bunch(
    tbt_data: TbtData, harpy_input: DotDict, file: Path
) -> dict[str, tfs.TfsDataFrame]:
    """
    Cleans data, analyses frequencies and searches for resonances.

    Args:
        tbt_data: single bunch `TbtData`.
        harpy_input: Analysis settings taken from the commandline.

    Returns:
        Dictionary with a `TfsDataFrame` per plane.
    """
    model = None
    if harpy_input.model is not None:
        model = tfs.read(harpy_input.model, index=COL_NAME).loc[:, COL_S]

    bpm_datas, usvs, lins, bad_bpms = {}, {}, {}, {}
    output_file_path = harpy_input.outputdir / file.name

    for plane in PLANES:
        bpm_data = _get_cut_tbt_matrix(tbt_data, harpy_input.turns, plane)
        bpm_data = _scale_to_meters(bpm_data, harpy_input.unit)
        bpm_data, usvs[plane], bad_bpms[plane], bpm_res = clean.clean(harpy_input, bpm_data, model)
        lins[plane], bpm_datas[plane] = _closed_orbit_analysis(bpm_data, model, bpm_res)

    tune_estimates = (
        harpy_input.tunes
        if harpy_input.autotunes is None
        else frequency.estimate_tunes(
            harpy_input,
            usvs
            if harpy_input.clean
            else {
                "X": clean.svd_decomposition(bpm_datas["X"], harpy_input.sing_val),
                "Y": clean.svd_decomposition(bpm_datas["Y"], harpy_input.sing_val),
            },
        )
    )

    spectra = {}
    for plane in PLANES:
        with timeit(lambda spanned: LOGGER.debug(f"Time for harmonic_analysis: {spanned}")):
            harpy_results, spectra[plane], bad_bpms_summaries = frequency.harpy_per_plane(
                harpy_input, bpm_datas[plane], usvs[plane], tune_estimates, plane
            )
        if "bpm_summary" in harpy_input.to_write:
            bad_bpms[plane].extend(bad_bpms_summaries)
            _write_bad_bpms(output_file_path, plane, bad_bpms[plane])

        if "spectra" in harpy_input.to_write or "full_spectra" in harpy_input.to_write:
            _write_spectrum(output_file_path, plane, spectra[plane])

        lins[plane] = lins[plane].loc[harpy_results.index].join(harpy_results)

        if harpy_input.is_free_kick:
            # Free kick assumes exponentially decaying oscillations.
            lins[plane] = kicker.phase_correction(bpm_datas[plane], lins[plane], plane)

    measured_tunes = [
        lins["X"][f"{COL_TUNE}X"].mean(),
        lins["Y"][f"{COL_TUNE}Y"].mean(),
        lins["X"][f"{COL_TUNE}Z"].mean() if tune_estimates[2] > 0 else 0,
    ]

    for plane in PLANES:
        lins[plane] = lins[plane].join(
            frequency.find_resonances(
                tunes=measured_tunes,
                nturns=bpm_datas[plane].shape[1],
                plane=plane,
                spectra=spectra[plane],
                order_resonances=harpy_input.resonances,
            )
        )
        lins[plane] = _add_calculated_phase_errors(lins[plane])
        lins[plane] = _sync_phase(lins[plane], plane)
        lins[plane] = _rescale_amps_to_main_line_and_compute_noise(lins[plane], plane)
        lins[plane] = lins[plane].sort_values(COL_S, axis=0, ascending=True)
        lins[plane] = tfs.TfsDataFrame(
            lins[plane], headers=_compute_headers(lins[plane], tbt_data.meta.get("date"))
        )
        if "lin" in harpy_input.to_write:
            _write_lin_tfs(output_file_path, plane, lins[plane])
    return lins


def _get_cut_tbt_matrix(tbt_data: TbtData, turn_indices: list[int], plane: str) -> pd.DataFrame:
    start = max(0, min(turn_indices))
    end = min(max(turn_indices), tbt_data.matrices[0][plane].shape[1])
    return tbt_data.matrices[0][plane].iloc[:, start:end].T.reset_index(drop=True).T


def _scale_to_meters(bpm_data: pd.DataFrame, unit: str) -> pd.DataFrame:
    scales_to_meters = {"um": 1e-6, "mm": 0.001, "cm": 0.01, "m": 1}
    bpm_data.iloc[:, :] = bpm_data.iloc[:, :].to_numpy() * scales_to_meters[unit]
    return bpm_data


def _closed_orbit_analysis(
    bpm_data: pd.DataFrame, model: None | pd.DataFrame, bpm_res
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Initialize the linear frame DataFrame with BPM names and positions
    lin_frame = pd.DataFrame(
        index=bpm_data.index.to_numpy(),
        data={
            COL_NAME: bpm_data.index.to_numpy(),
            COL_S: np.arange(bpm_data.index.size) if model is None else model.loc[bpm_data.index],
        },
    )
    lin_frame[COL_BPM_RES] = 0.0 if bpm_res is None else bpm_res.loc[lin_frame.index]
    with timeit(lambda spanned: LOGGER.debug(f"Time for orbit_analysis: {spanned}")):
        lin_frame = _get_orbit_data(lin_frame, bpm_data)

    # Subtract the mean from BPM data
    mean_subtracted_bpm_data = bpm_data.subtract(bpm_data.mean(axis=1), axis=0)

    return lin_frame, mean_subtracted_bpm_data


def _get_orbit_data(lin_frame: pd.DataFrame, bpm_data: pd.DataFrame) -> pd.DataFrame:
    lin_frame[COL_PK2PK] = np.max(bpm_data, axis=1) - np.min(bpm_data, axis=1)
    lin_frame[COL_CO] = np.mean(bpm_data, axis=1)
    lin_frame[COL_CORMS] = np.std(bpm_data, axis=1) / np.sqrt(bpm_data.shape[1])
    # TODO: Magic number 10?: Maybe accelerator dependent ... LHC 6-7?
    lin_frame[COL_NOISE] = lin_frame.loc[:, COL_BPM_RES] / np.sqrt(bpm_data.shape[1]) / 10.0
    return lin_frame


def _add_calculated_phase_errors(lin_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add calculated phase errors to the linear frame for all MU and PHASE columns.

    For each column starting with COL_MU or COL_PHASE, computes the corresponding
    error column using the spectral phase error formula.
    """
    if lin_frame[COL_NOISE].max() == 0.0:
        return lin_frame  # Skip if no noise data available

    noise_values = lin_frame[COL_NOISE].to_numpy()

    # Get all columns that represent phases (MU or PHASE)
    phase_columns = [col for col in lin_frame.columns if col.startswith((COL_MU, COL_PHASE))]

    for phase_col in phase_columns:
        # Corresponding amplitude column (e.g., MUX -> AMPX)
        amplitude_col = phase_col.replace(COL_MU, COL_AMP).replace(COL_PHASE, COL_AMP)

        # Calculate and assign the error
        lin_frame[f"{COL_ERR}{phase_col}"] = _get_spectral_phase_error(
            amplitude=lin_frame[amplitude_col],
            noise=noise_values,
        )

    return lin_frame


def _get_spectral_phase_error(amplitude: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """
    When the error is too big (> 2*pi*0.25 more or less) the noise is not Gaussian anymore.
    In such a case the distribution is almost uniform, so we set the error to be 0.3, which is
    the standard deviation of uniformly distributed phases.
    This approximation does not bias the error by more than 20%, and that is only for large errors.
    """
    error = noise / (np.where(amplitude > 0.0, amplitude, 1e-15) * 2 * np.pi)
    return np.where(error > 0.25, 0.3, error)


def _sync_phase(lin_frame: pd.DataFrame, plane: str) -> pd.DataFrame:
    """
    Produces ``MUXSYNC`` and ``MUYSYNC`` columns that are ``MUX/MUY`` but shifted such that for
    BPM at index 0 is always 0. It allows to compare phases of consecutive measurements and if
    some measurements stick out remove them from the data set. Original author is **skowron**.
    """
    phase = lin_frame[f"{COL_MU}{plane}"].to_numpy()
    phase = phase - phase[0]
    lin_frame[f"{COL_MU}{plane}SYNC"] = np.where(np.abs(phase) > 0.5, phase - np.sign(phase), phase)
    return lin_frame


def _compute_headers(panda: pd.DataFrame, date: None | pd.Timestamp = None) -> dict[str, float]:
    headers = {}
    for plane in ALL_PLANES:
        for prefix in ("", "NAT"):
            try:
                bpm_tunes = panda[f"{prefix}{COL_TUNE}{plane}"]
            except KeyError:
                pass
            else:
                headers[f"{prefix}Q{PLANE_TO_NUM[plane]}"] = np.mean(bpm_tunes)
                headers[f"{prefix}Q{PLANE_TO_NUM[plane]}RMS"] = np.std(
                    bpm_tunes
                )  # TODO: not really the RMS?
    if date:
        headers[COL_TIME] = date.strftime(formats.TIME)
    headers[MAINLINE_UNIT] = "m"
    return headers


def _write_bad_bpms(
    output_path_without_suffix: Path | str, plane: str, bad_bpms_with_reasons: str
) -> None:
    bad_bpms_file = Path(f"{output_path_without_suffix}.bad_bpms_{plane.lower()}")
    bad_bpms_file.write_text("\n".join(bad_bpms_with_reasons) + "\n")


def _write_spectrum(
    output_path_without_suffix: Path | str, plane: str, spectra: tfs.TfsDataFrame
) -> None:
    tfs.write(
        f"{output_path_without_suffix}{FILE_AMPS_EXT.format(plane=plane.lower())}",
        spectra[COL_COEFFS].abs().T,
    )
    tfs.write(
        f"{output_path_without_suffix}{FILE_FREQS_EXT.format(plane=plane.lower())}",
        spectra[COL_FREQS].T,
    )


def _write_lin_tfs(output_path_without_suffix: Path, plane: str, lin_frame: pd.DataFrame):
    tfs.write(
        f"{output_path_without_suffix}{FILE_LIN_EXT.format(plane=plane.lower())}",
        lin_frame,
    )


def _rescale_amps_to_main_line_and_compute_noise(df: pd.DataFrame, plane: str) -> pd.DataFrame:
    """
    Rescale secondary amplitudes to main line amplitude and compute noise-related errors.
    """
    cols = [col for col in df.columns if col.startswith(COL_AMP)]
    cols.remove(f"{COL_AMP}{plane}")
    df[cols] = df[cols].div(df[f"{COL_AMP}{plane}"], axis="index")
    amps = df[f"{COL_AMP}{plane}"].to_numpy()

    if df[COL_NOISE].max() == 0.0:
        return df  # Do not calculate errors when no noise was calculated
    noise_scaled = df[COL_NOISE] / amps
    df[COL_NOISE_SCALED] = noise_scaled
    df[f"{COL_ERR}{COL_AMP}{plane}"] = df[COL_NOISE]
    if f"{COL_NATTUNE}{plane}" in df.columns:
        df[f"{COL_ERR}{COL_NATAMP}{plane}"] = df[COL_NOISE]

    # Create dedicated dataframe with error columns to assign later (cleaner
    # and faster than assigning individual columns)
    df_amp = pd.DataFrame(
        data={
            f"{COL_ERR}{col}": noise_scaled * np.sqrt(1 + np.square(df.loc[:, col])) for col in cols
        },
        index=df.index,
        dtype=pd.Float64Dtype(),
    )
    df[df_amp.columns] = df_amp
    return df
