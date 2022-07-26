"""
Resonance Driving Terms
-----------------------

This module contains RDT calculations related functionality of ``optics_measurements``.
It provides functions to compute global resonance driving terms **f_jklm**.
"""
from copy import deepcopy
from os.path import join

import numpy as np
import pandas as pd
import tfs
from scipy.optimize import curve_fit
from scipy.sparse import diags

from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import ERR, EXT, AMPLITUDE
from omc3.optics_measurements.toolbox import df_diff
from omc3.utils import iotools, logging_tools, stats
from optics_functions.rdt import get_all_to_order, jklm2str

NBPMS_FOR_90 = 3
LOGGER = logging_tools.get_logger(__name__)

def _generate_plane_rdts(order):
    """
    This helper function generates two dictionnaries representing on what plane(s)
    a RDT can be seen and which tune it is a multiple of.

    For a given j+k+l+m = n multiple order, a line can be seen:
      - on the horizontal axis if j != 0, at:
          (1 - j + k) Qx + (m - l) Qy
      - on the vertical axis if l != 0, at:
          (k - j) Qx + (1 - l + m) Qy

    Reference equations are (3.27) and (3.28) in
    [Andrea Franchi's Thesis](https://repository.gsi.de/record/55413/files/GSI-Diss-2006-07.pdf)
    """
    # Get all the valid RDTs up to a certain order
    all_rdts = get_all_to_order(order)

    single_plane = {'X': [], 'Y': []}
    double_plane = {'X': [], 'Y': []}
    # Iterate through our RDTs and classify them depending on what plane they act
    for (j,k,l,m) in all_rdts:
        if j == 0 and l == 0:  # the RDT can't be seen on any plane
            continue
        if l+m == 0 and j != 0:  # The line where the RDT is seen is a multiple of the Qx line
            single_plane['X'].append((j,k,l,m))  # e.g. f1400, f3000, f4000
        elif j+k == 0 and l != 0:  # same, but for the Qy line
            single_plane['Y'].append((j,k,l,m))  # e.g. f0030,f0040
        elif j == 0:  # The RDT can only be seen on the vertical plane and uses both Qx and Qy
            double_plane['Y'].append((j,k,l,m))  # e.g. f0111, f0120
        elif l == 0: # same, but for the horizontal plane
            double_plane['X'].append((j,k,l,m))  # e.g. f1001, f1002
        else:  # The RDT can be seen on both planes and is a multiple of both Qx and Qy
            double_plane['X'].append((j,k,l,m))  # e.g. f1020, f1120
            double_plane['Y'].append((j,k,l,m))

    return single_plane, double_plane


def calculate(measure_input, input_files, tunes, phases, invariants, header):
    """

    Args:
        measure_input:
        input_files:
        tunes:
        invariants:
        header:

    Returns:
    """
    LOGGER.info(f"Start of RDT analysis")
    meas_input = deepcopy(measure_input)
    meas_input["compensation"] = "none"
    LOGGER.info(f"Calculating RDTs up to magnet order {meas_input['rdt_magnet_order']}")

    single_plane_rdts, double_plane_rdts = _generate_plane_rdts(meas_input["rdt_magnet_order"])
    for plane in PLANES:
        bpm_names = input_files.bpms(plane=plane, dpp_value=0)
        for_rdts = _best_90_degree_phases(meas_input, bpm_names, phases, tunes, plane)
        LOGGER.info(f"Average phase advance between BPM pairs: {for_rdts.loc[:,'MEAS'].mean()}")
        for rdt in single_plane_rdts[plane]:
            try:
                df = _process_rdt(meas_input, input_files, for_rdts, invariants, plane, rdt)
            except ValueError as e:  # catch the AMP line not being found in the lin file
                LOGGER.warning(f"RDT calculation failed for {jklm2str(*rdt)}: {str(e)}")
                continue
            write(df, add_freq_to_header(header, plane, rdt), meas_input, plane, rdt)
    for plane in PLANES:
        bpm_names = input_files.bpms(dpp_value=0)
        for_rdts = _best_90_degree_phases(meas_input, bpm_names, phases, tunes, plane)
        LOGGER.info(f"Average phase advance between BPM pairs: {for_rdts.loc[:, 'MEAS'].mean()}")
        for rdt in double_plane_rdts[plane]:
            try:
                df = _process_rdt(meas_input, input_files, for_rdts, invariants, plane, rdt)
            except ValueError as e:
                LOGGER.warning(f"RDT calculation failed for {jklm2str(*rdt)}: {str(e)}")
                continue
            write(df, add_freq_to_header(header, plane, rdt), meas_input, plane, rdt)


def write(df, header, meas_input, plane, rdt):
    outputdir = join(meas_input.outputdir, "rdt", _rdt_to_order_and_type(rdt))
    iotools.create_dirs(outputdir)
    tfs.write(join(outputdir, f"f{_rdt_to_str(rdt)}_{plane.lower()}{EXT}"), df, header,
              save_index="NAME")


def _rdt_to_str(rdt):
    j, k, l, m = rdt
    return f"{j}{k}{l}{m}"


def _rdt_to_order_and_type(rdt):
    j, k, l, m = rdt
    rdt_type = "normal" if (l + m) % 2 == 0 else "skew"
    orders = dict(((1, "dipole"), 
                   (2, "quadrupole"), 
                   (3, "sextupole"), 
                   (4, "octupole"),
                   (5, "decapole"),
                   (6, "dodecapole"),
                   (7, "tetradecapole"),
                   (8, "hexadecapole"),
                 ))
    return f"{rdt_type}_{orders[j + k + l + m]}"


def _best_90_degree_phases(meas_input, bpm_names, phases, tunes, plane):
    filtered = phases[plane]["uncompensated"]["MEAS"].loc[bpm_names, bpm_names]
    phase_meas = pd.concat(
        (filtered % 1, (filtered.iloc[:, :NBPMS_FOR_90] + tunes[plane]["Q"]) % 1), axis=1)
    second_bmps = np.abs(phase_meas * _get_n_upper_diagonals(NBPMS_FOR_90, phase_meas.shape)
                         - 0.25).idxmin(axis=1)
    filtered.iloc[-NBPMS_FOR_90:, :NBPMS_FOR_90] = (filtered.iloc[-NBPMS_FOR_90:,
                                                    :NBPMS_FOR_90] + tunes[plane]["Q"]) % 1
    filtered["NAME2"], filtered["MEAS"], filtered["ERRMEAS"] = second_bmps, filtered.lookup(
        bpm_names, second_bmps), phases[plane]["uncompensated"]["ERRMEAS"].lookup(bpm_names, second_bmps)
    for_rdts = pd.merge(filtered.loc[:, ["NAME2", "MEAS", "ERRMEAS"]],
                        meas_input.accelerator.model.loc[:, ["S"]], how="inner",
                        left_index=True, right_index=True)
    return for_rdts


def _get_n_upper_diagonals(n, shape):
    return diags(np.ones((n, shape[0])), np.arange(n)+1, shape=shape).toarray()


def _determine_line(rdt, plane):
    j, k, l, m = rdt
    lines = dict(X=(1 - j + k, m - l, 0),
                 Y=(k - j, 1 - l + m, 0))
    return lines[plane]


def add_freq_to_header(header, plane, rdt):
    mod_header = header.copy()
    line = _determine_line(rdt, plane)
    freq = np.mod(line@np.array([header['Q1'], header['Q2'], 0]), 1)
    mod_header["FREQ"] = freq if freq <= 0.5 else 1 - freq
    return mod_header


def _process_rdt(meas_input, input_files, phase_data, invariants, plane, rdt):
    df = pd.DataFrame(phase_data)
    second_bpms = df.loc[:, "NAME2"].to_numpy()
    df["S2"] = df.loc[second_bpms, "S"].to_numpy()
    df["COUNT"] = len(input_files.dpp_frames(plane, 0))
    line = _determine_line(rdt, plane)
    phase_sign, suffix = get_line_sign_and_suffix(line, input_files, plane)
    comp_coeffs1 = to_complex(
        input_files.joined_frame(plane, [f"AMP{suffix}"], dpp_value=0).loc[df.index, :].to_numpy(),
        phase_sign * input_files.joined_frame(plane, [f"PHASE{suffix}"], dpp_value=0).loc[df.index, :].to_numpy())
    # Multiples of tunes needs to be added to phase at second BPM if that is in second turn
    phase2 = phase_sign * input_files.joined_frame(plane, [f"PHASE{suffix}"], dpp_value=0).loc[second_bpms, :].to_numpy()
    comp_coeffs2 = to_complex(
        input_files.joined_frame(plane, [f"AMP{suffix}"], dpp_value=0).loc[second_bpms, :].to_numpy(),
        _add_tunes_if_in_second_turn(df, input_files, line, phase2))
    # Get amplitude and phase of the line from linx/liny file
    line_amp, line_phase, line_amp_e, line_phase_e = complex_secondary_lines(  # TODO use the errors
        df.loc[:, "MEAS"].to_numpy()[:, np.newaxis] * meas_input.accelerator.beam_direction,
        df.loc[:, "ERRMEAS"].to_numpy()[:, np.newaxis], comp_coeffs1, comp_coeffs2)
    rdt_phases_per_file = _calculate_rdt_phases_from_line_phases(df, input_files, line, line_phase)
    rdt_angles = stats.circular_mean(rdt_phases_per_file, period=1, axis=1) % 1
    df[f"PHASE"] = rdt_angles
    df[f"{ERR}PHASE"] = stats.circular_error(rdt_phases_per_file, period=1, axis=1)
    df[AMPLITUDE], df[f"{ERR}{AMPLITUDE}"] = _fit_rdt_amplitudes(invariants, line_amp, plane, rdt)
    df[f"REAL"] = np.cos(2 * np.pi * rdt_angles) * df.loc[:, AMPLITUDE].to_numpy()
    df[f"IMAG"] = np.sin(2 * np.pi * rdt_angles) * df.loc[:, AMPLITUDE].to_numpy()
    # in old files there was "EAMP" and "PHASE_STD"
    return df.loc[:, ["S", "COUNT", AMPLITUDE, f"{ERR}{AMPLITUDE}", "PHASE", f"{ERR}PHASE", "REAL", "IMAG"]]


def _add_tunes_if_in_second_turn(df, input_files, line, phase2):
    mask = df_diff(df, "S", "S2") > 0
    tunes = np.empty((2, len(input_files.dpp_frames("X", 0))))
    for i, plane in enumerate(PLANES):
        tunes[i] = np.array([lin.headers[f"Q{i+1}"] for lin in input_files.dpp_frames(plane, 0)])
    phase2[mask, :] = phase2[mask, :] + line[0] * tunes[0] + line[1] * tunes[1]
    return phase2


def _calculate_rdt_phases_from_line_phases(df, input_files, line, line_phase):
    phases = np.zeros((2, df.index.size, len(input_files.dpp_frames("X", 0))))
    for i, plane in enumerate(PLANES):
        if line[i] != 0:
            phases[i] = input_files.joined_frame(plane, [f"MU{plane}"], dpp_value=0).loc[df.index, :].to_numpy() % 1
    return line_phase - line[0] * phases[0] - line[1] * phases[1] + 0.25


def _fit_rdt_amplitudes(invariants, line_amp, plane, rdt):
    """
    Returns RDT amplitudes in units of meters ^ {1 - n/2}, where n is the order of RDT.
    """
    amps, err_amps = np.empty(line_amp.shape[0]), np.empty(line_amp.shape[0])
    kick_data = get_linearized_problem(invariants, plane, rdt)  # corresponding to actions in meters
    guess = np.mean(line_amp / kick_data, axis=1)

    def fitting(x, f):
        return f * x

    for i, bpm_rdt_data in enumerate(line_amp):
        popt, pcov = curve_fit(fitting, kick_data, bpm_rdt_data, p0=guess[i])
        amps[i] = popt[0]
        err_amps[i] = np.sqrt(pcov)[0] if np.isfinite(np.sqrt(pcov)[0]) else 0. # if single file is used, the error is reported as Inf, which is then overwritten with 0
    return amps, err_amps


def get_linearized_problem(invs, plane, rdt):
    """
    2 * j * f_jklm * (powers of 2Jx and 2Jy) : f_jklm is later a parameter of a fit
    we use sqrt(2J): unit is sqrt(m).
    """
    j, k, l, m = rdt
    act_x = invs["X"].T[0]
    act_y = invs["Y"].T[0]
    if plane == "X":
        return 2 * j * act_x ** (j + k - 2) * act_y ** (l + m)
    return 2 * l * act_x ** (j + k) * act_y ** (l + m - 2)


def get_line_sign_and_suffix(line, input_files, plane):
    suffix = f"{line[0]}{line[1]}".replace("-", "_")
    conj_suffix = f"{-line[0]}{-line[1]}".replace("-", "_")
    if f"AMP{suffix}" in input_files[plane][0].columns:
        return 1, suffix
    if f"AMP{conj_suffix}" in input_files[plane][0].columns:
        return -1, conj_suffix

    # The specified AMP column hasn't been found in the lin file
    msg = (f"The column AMP{suffix} has not been found in the lin{plane.lower()} file. "
            "Consider re-running the frequency analysis with a higher order resonance term")
    raise ValueError(msg)


def complex_secondary_lines(phase_adv, err_padv, sig1, sig2):
    """

    Args:
        phase_adv: phase advances between two BPMs.
        err_padv: error on the phase advance between two BPMs.
        sig1: Complex coefficients of a secondary lines at the first BPM of the pairs.
        sig2: Complex coefficients of a secondary lines at the second BPM of the pairs.

    Returns:
         `Tuple` with amplitudes, phases err_amplitudes and err_phases of the complex signal.
    """
    tp = 2.0 * np.pi
    # computing complex secondary line (h-)
    sig = (sig1 * (1 + 1j / np.tan(phase_adv * tp)) - sig2 * 1j / np.sin(phase_adv * tp)) / 2
    # computing error secondary line (h-) # TODO is this really the error?
    esig = (sig1 * 1j / np.square(np.sin(phase_adv * tp)) +
            sig2 * -1j * np.cos(phase_adv * tp) / np.square(np.sin(phase_adv * tp))) * err_padv / 2
    return (np.abs(sig), (np.angle(sig) / tp) % 1,
            np.abs(esig), (np.angle(esig) / tp) % 1)


def to_complex(amplitudes, phases, period=1):
    return amplitudes * np.exp(2j * np.pi * phases / period)
