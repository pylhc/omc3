from os.path import join, basename
from collections import OrderedDict

import numpy as np
import pandas as pd

import tfs
from utils.contexts import timeit
from utils import logging_tools
from harpy import core, clean, kicker

LOGGER = logging_tools.get_logger(__name__)
PLANES = ("X", "Y")


def run_per_bunch(tbt_data, harpy_input):
    model = tfs.read(harpy_input.model, index="NAME").loc[:, 'S']
    bpm_datas, usvs, lins, bad_bpms = {}, {}, {}, {}
    for plane in PLANES:
        bpm_data = _get_cut_tbt_matrix(tbt_data, harpy_input.turns, plane)
        bpm_data = _scale_to_mm(bpm_data, harpy_input.unit)
        bpm_data, usvs[plane], bad_bpms[plane], bpm_res = clean.clean(harpy_input, bpm_data, model)
        lins[plane], bpm_datas[plane] = _closed_orbit_analysis(bpm_data, model, bpm_res)

    all_bad_bpms, lins = _do_harpy(harpy_input, bpm_datas, usvs, lins, bad_bpms)
    for plane in PLANES:
        write_bad_bpms(harpy_input.file, all_bad_bpms[plane], harpy_input.outputdir, plane)
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


def _do_harpy(main_input, bpm_datas, usvs, lin_frames, all_bad_bpms):
    harpy_iterator = core.harpy(
        main_input,
        bpm_datas["X"], usvs["X"],
        bpm_datas["Y"], usvs["Y"],
    )
    lin = {}
    for plane in PLANES:
        with timeit(lambda spanned: LOGGER.debug(f"Time for harmonic_analysis: {spanned}")):
            harpy_results, spectr, bad_bpms_summaries = harpy_iterator.__next__()
            lin_frame = lin_frames[plane]
            lin_frame = lin_frame.loc[harpy_results.index].join(harpy_results)
        if main_input.is_free_kick:
            bpm_data = bpm_datas[plane]
            lin_frame = kicker.phase_correction(bpm_data, lin_frame, plane)
        lin_frame = _sync_phase(lin_frame, plane)
        lin_frame = _rescale_amps_to_main_line(lin_frame, plane)
        lin_frame = _add_resonances_noise(lin_frame, plane)
        lin_frame = lin_frame.sort_values('S', axis=0, ascending=True)
        headers = _compute_headers(lin_frame, plane)
        write_harpy_output(main_input, lin_frame, headers, spectr, plane)
        all_bad_bpms[plane].extend(bad_bpms_summaries)
        lin[plane] = tfs.TfsDataFrame(lin_frame, headers=headers)
    return all_bad_bpms, lin


def _sync_phase(lin_frame, plane):
    """ Produces MUXSYNC and MUYSYNC column that is MUX/Y but
        shifted such that for bpm at index 0 is always 0.
        It allows to compare phases of consecutive measurements
        and if some measurements stick out remove them from the data set.
        author: skowron
        """
    phase = lin_frame.loc[:, f"MU{plane}"].values
    phase = phase - phase[0]
    lin_frame[f"MU{plane}SYNC"] = np.where(np.abs(phase) > 0.5, phase - np.sign(phase), phase)
    return lin_frame


def _rescale_amps_to_main_line(panda, plane):
    cols = [col for col in panda.columns.values if col.startswith('AMP')]
    cols.remove(f"AMP{plane}")
    panda.loc[:, cols] = panda.loc[:, cols].div(panda.loc[:, f"AMP{plane}"], axis="index")
    return panda


def _add_resonances_noise(lin_frame, plane):
    if np.max(lin_frame.loc[:, 'NOISE'].values) == 0.0:
        return lin_frame   # Do not calculated errors when no noise was calculated
    cols = [col for col in lin_frame.columns.values if col.startswith('AMP')]
    cols.remove(f"AMP{plane}")
    noise_scaled = lin_frame.loc[:, 'NOISE'] / lin_frame.loc[:, f"AMP{plane}"]
    lin_frame.loc[:, "NOISE_SCALED"] = noise_scaled
    lin_frame.loc[:, f"ERR_AMP{plane}"] = lin_frame.loc[:, 'NOISE']
    lin_frame.loc[:, f"ERR_MU{plane}"] = _get_spectral_phase_error(
            lin_frame.loc[:, f"AMP{plane}"],
            lin_frame.loc[:, "NOISE"],
        )
    if f"NATTUNE{plane}" in lin_frame.columns:
        lin_frame.loc[:, f"ERR_NATAMP{plane}"] = lin_frame.loc[:, 'NOISE']
        lin_frame.loc[:, f"ERR_NATMU{plane}"] = _get_spectral_phase_error(
                lin_frame.loc[:, f"NATAMP{plane}"],
                lin_frame.loc[:, "NOISE"],
            )
    for col in cols:
        this_amp = lin_frame.loc[:, col]
        lin_frame.loc[:, f"ERR_{col}"] = noise_scaled * np.sqrt(1 + np.square(this_amp))
        err_phase_col = f"ERR_{col.replace('AMP', 'PHASE')}"
        if col == "AMPZ":
            err_phase_col = "ERR_MUZ"
        lin_frame.loc[:, err_phase_col] = _get_spectral_phase_error(this_amp, noise_scaled)
    return lin_frame


def _get_spectral_phase_error(amplitude, noise):
    """
    When the error is too big (> 2*pi*0.25 more or less) the noise is not
    Gaussian anymore and it close to an uniform distribution, so we set the
    error to be 0.3 as it is the standard deviation of uniformly
    distributed phases. This is an approximation that should keep the error
    of the error below 20%.
    """
    error = noise / (np.where(amplitude > 0.0, amplitude, 1e-15) * 2 * np.pi)
    return np.where(error > 0.25, 0.3, error)


def _compute_headers(panda, plane):
    plane_number = {"X": "1", "Y": "2"}[plane]
    headers = OrderedDict()
    tunes = panda.loc[:, f"TUNE{plane}"]
    headers[f"Q{plane_number}"] = np.mean(tunes)
    headers[f"Q{plane_number}RMS"] = np.std(tunes)
    try:
        nattunes = panda.loc[:, f"NATTUNE{plane}"]
        headers[f"NATQ{plane_number}"] = np.mean(nattunes)
        headers[f"NATQ{plane_number}RMS"] = np.std(nattunes)
    except KeyError:
        pass  # No natural tunes
    try:
        ztunes = panda.loc[:, "TUNEZ"]
        headers["Q3"] = np.mean(ztunes)
        headers["Q3RMS"] = np.std(ztunes)
    except KeyError:
        pass  # No tunez
    headers["DPP"] = 0.0  # TODO remove
    return headers


def write_bad_bpms(bin_path, bad_bpms_with_reasons, output_dir, plane):
    bad_bpms_file = get_output_path(output_dir, bin_path, ".bad_bpms_" + plane.lower())
    with open(bad_bpms_file, 'w') as bad_bpms_writer:
        for line in bad_bpms_with_reasons:
            bad_bpms_writer.write(line + '\n')


def write_harpy_output(main_input, harpy_data_frame, headers, spectrum, plane):
    output_file = get_output_path(main_input.outputdir, main_input.file,
                                          ".lin" + plane.lower())
    tfs.write(output_file, harpy_data_frame, headers)
    if "spectra" in main_input.to_write or "full_spectra" in main_input.to_write:
        _write_full_spectrum(main_input, spectrum, plane)


def _write_full_spectrum(main_input, spectrum, plane):
    spectr_amps_files = get_output_path(main_input.outputdir, main_input.file,
                                                ".amps" + plane.lower())
    amps_df = spectrum["COEFS"].abs().T
    tfs.write(spectr_amps_files, amps_df)
    spectr_freqs_files = get_output_path(main_input.outputdir, main_input.file,
                                                 ".freqs" + plane.lower())
    freqs_df = spectrum["FREQS"].T
    tfs.write(spectr_freqs_files, freqs_df)


def get_output_path(output_dir, path, suffix):
    return join(output_dir, basename(path) + suffix)
