"""
Module harpy.handler
---------------------

Handles the cleaning, frequency analysis and resonance search for a single-bunch TbtData.

"""
from os.path import join, basename
from collections import OrderedDict

import numpy as np
import pandas as pd

import tfs
from utils.contexts import timeit
from utils import logging_tools
from harpy import frequency, clean

LOGGER = logging_tools.get_logger(__name__)
PLANES = ("X", "Y")
ALL_PLANES = ("X", "Y", "Z")
PLANE_TO_NUM = {"X": 1, "Y": 2, "Z": 3}


def run_per_bunch(tbt_data, harpy_input):
    """
    Cleans data, analyses frequencies and searches resonances

    Args:
        tbt_data: single bunch TbtData
        harpy_input: Analysis settings

    Returns:
        Dictionary of TfsDataFrames per plane
    """
    model = tfs.read(harpy_input.model, index="NAME").loc[:, 'S']
    bpm_datas, usvs, lins, bad_bpms = {}, {}, {}, {}
    output_file_path = _get_output_path_without_suffix(harpy_input.outputdir, harpy_input.files)
    for plane in PLANES:
        bpm_data = _get_cut_tbt_matrix(tbt_data, harpy_input.turns, plane)
        bpm_data = _scale_to_mm(bpm_data, harpy_input.unit)
        bpm_data, usvs[plane], bad_bpms[plane], bpm_res = clean.clean(harpy_input, bpm_data, model)
        lins[plane], bpm_datas[plane] = _closed_orbit_analysis(bpm_data, model, bpm_res)

    tune_estimates = harpy_input.tunes if harpy_input.autotunes is None else frequency.estimate_tunes(
        harpy_input, usvs if harpy_input.clean else
        dict(X=clean.svd_decomposition(bpm_datas["X"], harpy_input.sing_val),
             Y=clean.svd_decomposition(bpm_datas["Y"], harpy_input.sing_val)))

    spectra = {}
    for plane in PLANES:
        with timeit(lambda spanned: LOGGER.debug(f"Time for harmonic_analysis: {spanned}")):
            harpy_results, spectra[plane], bad_bpms_summaries = frequency.harpy_per_plane(
                harpy_input, bpm_datas[plane], usvs[plane], tune_estimates, plane)
        if "bpm_summary" in harpy_input.to_write:
            bad_bpms[plane].extend(bad_bpms_summaries)
            _write_bad_bpms(output_file_path, plane, bad_bpms[plane])
        if "spectra" in harpy_input.to_write or "full_spectra" in harpy_input.to_write:
            _write_spectrum(output_file_path, plane, spectra[plane])
        lins[plane] = lins[plane].loc[harpy_results.index].join(harpy_results)

    measured_tunes = [lins["X"]["TUNEX"].mean(), lins["Y"]["TUNEY"].mean(),
                      lins["X"]["TUNEZ"].mean() if tune_estimates[2] > 0 else 0]
    nturns = bpm_datas["X"].shape[1]

    for plane in PLANES:
        lins[plane] = lins[plane].join(frequency.find_resonances(
            measured_tunes, nturns, plane, spectra[plane]))
        lins[plane] = _add_calculated_phase_errors(lins[plane])
        lins[plane] = _sync_phase(lins[plane], plane)
        lins[plane] = _rescale_amps_to_main_line_and_compute_noise(lins[plane], plane)
        lins[plane] = lins[plane].sort_values('S', axis=0, ascending=True)
        lins[plane] = tfs.TfsDataFrame(lins[plane], headers=_compute_headers(lins[plane]))
        if "lin" in harpy_input.to_write:
            _write_lin_tfs(output_file_path, plane, lins[plane])
    return lins


def _get_cut_tbt_matrix(tbt_data, turn_indices, plane):
    start = max(0, min(turn_indices))
    end = min(max(turn_indices), tbt_data.matrices[0][plane].shape[1])
    return tbt_data.matrices[0][plane].iloc[:, start:end]


def _scale_to_mm(bpm_data, unit):
    scales_to_mm = {'um': 1000, 'mm': 1, 'cm': 0.1, 'm': 0.001}
    return bpm_data * scales_to_mm[unit]


def _closed_orbit_analysis(bpm_data, model, bpm_res):
    lin_frame = pd.DataFrame(index=bpm_data.index,
                             data=OrderedDict([("NAME", bpm_data.index),
                                               ("S", model.loc[bpm_data.index])]))
    lin_frame['BPM_RES'] = 0.0 if bpm_res is None else bpm_res.loc[lin_frame.index]
    with timeit(lambda spanned: LOGGER.debug(f"Time for orbit_analysis: {spanned}")):
        lin_frame = _get_orbit_data(lin_frame, bpm_data)
    return lin_frame, bpm_data.subtract(bpm_data.mean(axis=1), axis=0)


def _get_orbit_data(lin_frame, bpm_data):
    lin_frame['PK2PK'] = np.max(bpm_data, axis=1) - np.min(bpm_data, axis=1)
    lin_frame['CO'] = np.mean(bpm_data, axis=1)
    lin_frame['CORMS'] = np.std(bpm_data, axis=1) / np.sqrt(bpm_data.shape[1])
    # TODO: Magic number 10?: Maybe accelerator dependent ... LHC 6-7?
    lin_frame['NOISE'] = lin_frame.loc[:, 'BPM_RES'] / np.sqrt(bpm_data.shape[1]) / 10.0
    return lin_frame


def _add_calculated_phase_errors(lin_frame):
    noise = lin_frame.loc[:, 'NOISE'].values
    if np.max(noise) == 0.0:
        return lin_frame   # Do not calculated errors when no noise was calculated
    for name_root in ('MU', 'PHASE'):
        cols = [col for col in lin_frame.columns.values if name_root in col]
        for col in cols:
            lin_frame[f"ERR_{col}"] = _get_spectral_phase_error(
                lin_frame.loc[:, f"{col.replace(name_root, 'AMP')}"], noise)
    return lin_frame


def _get_spectral_phase_error(amplitude, noise):
    """
    When the error is too big (> 2*pi*0.25 more or less) the noise is not Gaussian anymore.
    In such a case the distribution is almost uniform, so we set the error to be 0.3,
    which is the standard deviation of uniformly distributed phases.
    This approximation does not bias the error by more than 20%, and that is only for large errors.
    """
    error = noise / (np.where(amplitude > 0.0, amplitude, 1e-15) * 2 * np.pi)
    return np.where(error > 0.25, 0.3, error)


def _sync_phase(lin_frame, plane):
    """ Produces MUXSYNC and MUYSYNC column that is MUX/Y but shifted such that for bpm at index 0
     is always 0. It allows to compare phases of consecutive measurements and if some measurements
     stick out remove them from the data set. author: skowron
    """
    phase = lin_frame.loc[:, f"MU{plane}"].values
    phase = phase - phase[0]
    lin_frame[f"MU{plane}SYNC"] = np.where(np.abs(phase) > 0.5, phase - np.sign(phase), phase)
    return lin_frame


def _compute_headers(panda):
    headers = OrderedDict()
    for plane in ALL_PLANES:
        for prefix in ("", "NAT"):
            try:
                bpm_tunes = panda.loc[:, f"{prefix}TUNE{plane}"]
            except KeyError:
                pass
            else:
                headers[f"{prefix}Q{PLANE_TO_NUM[plane]}"] = np.mean(bpm_tunes)
                headers[f"{prefix}Q{PLANE_TO_NUM[plane]}RMS"] = np.std(bpm_tunes)
    headers["DPP"] = 0.0  # TODO later remove - should be calculated in measure_optics
    return headers


def _write_bad_bpms(output_path_without_suffix, plane, bad_bpms_with_reasons):
    with open(f"{output_path_without_suffix}.bad_bpms_{plane.lower()}", 'w') as bad_bpms_file:
        for line in bad_bpms_with_reasons:
            bad_bpms_file.write(f"{line}\n")


def _write_spectrum(output_path_without_suffix, plane, spectra):
    tfs.write(f"{output_path_without_suffix}.amps{plane.lower()}", spectra["COEFFS"].abs().T)
    tfs.write(f"{output_path_without_suffix}.freqs{plane.lower()}", spectra["FREQS"].T)


def _write_lin_tfs(output_path_without_suffix, plane, lin_frame):
    tfs.write(f"{output_path_without_suffix}.lin{plane.lower()}", lin_frame)


def _get_output_path_without_suffix(output_dir, file_path):
    return join(output_dir, basename(file_path))


def _rescale_amps_to_main_line_and_compute_noise(panda, plane):
    """
    TODO    follows non-transpararent convention
    TODO    the consequent analysis has to be changed if removed
    """
    cols = [col for col in panda.columns.values if col.startswith('AMP')]
    cols.remove(f"AMP{plane}")
    panda.loc[:, cols] = panda.loc[:, cols].div(panda.loc[:, f"AMP{plane}"], axis="index")
    # Division by two for backwards compatibility with Drive, i.e. the unit is [2mm]
    panda[f"AMP{plane}"] = panda.loc[:, f"AMP{plane}"].values / 2
    if f"NATAMP{plane}" in panda.columns:
        panda[f"NATAMP{plane}"] = panda.loc[:, f"NATAMP{plane}"].values / 2

    if np.max(panda.loc[:, 'NOISE'].values) == 0.0:
        return panda  # Do not calculated errors when no noise was calculated
    noise_scaled = panda.loc[:, 'NOISE'] / panda.loc[:, f"AMP{plane}"]
    panda.loc[:, "NOISE_SCALED"] = noise_scaled
    panda.loc[:, f"ERR_AMP{plane}"] = panda.loc[:, 'NOISE']
    if f"NATTUNE{plane}" in panda.columns:
        panda.loc[:, f"ERR_NATAMP{plane}"] = panda.loc[:, 'NOISE']
    for col in cols:
        this_amp = panda.loc[:, col]
        panda.loc[:, f"ERR_{col}"] = noise_scaled * np.sqrt(1 + np.square(this_amp))
    return panda
