"""
RDTs
-------------------

Computes global resonance driving terms f_jklm.
"""
from os.path import join
import numpy as np
from copy import deepcopy
from utils import logging_tools
from scipy.optimize import curve_fit
from optics_measurements import phase
from optics_measurements.toolbox import df_diff
from optics_measurements.constants import PLANES, ERR, EXT
from scipy.sparse import diags
from utils import iotools
import pandas as pd
from utils import stats
import tfs

NBPMS_FOR_90 = 3
LOGGER = logging_tools.get_logger(__name__)
SINGLE_PLANE_RDTS = {"X": ((3, 0, 0, 0), (1, 2, 0, 0),      # Normal Sextupolar
                           (4, 0, 0, 0), (1, 3, 0, 0),      # Normal Octupolar
                           ),
                     "Y": ((0, 0, 3, 0), (0, 0, 1, 2),   # Skew Sextupolar
                           (0, 0, 4, 0), (0, 0, 1, 3)    # Normal Octupolar
                           )}
DOUBLE_PLANE_RDTS = {"X": ((1, 0, 0, 1), (1, 0, 1, 0),  # Quadrupole
                           (1, 0, 2, 0), (1, 0, 0, 2),  # Normal Sextupole
                           (1, 1, 0, 1), (2, 0, 1, 0), (1, 1, 1, 0), (2, 0, 0, 1),  # Skew Sextupole
                           (2, 0, 0, 2), (1, 1, 2, 0), (1, 1, 0, 2), (2, 0, 2, 0)  # Normal Octupole
                           ),
                     "Y": ((0, 1, 1, 0), (1, 0, 1, 0),  # Quadrupole
                           (0, 1, 1, 1), (1, 0, 2, 0), (0, 1, 2, 0), (1, 0, 1, 1),  # Normal Sextupole
                           (0, 2, 1, 0), (2, 0, 1, 0),  # Skew Sextupole
                           (2, 0, 2, 0), (2, 0, 1, 1), (0, 2, 2, 0), (0, 2, 1, 1)  # Normal Octupole
                           )}


def calculate(measure_input, input_files, tunes, invariants, header):
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
    phases = {}
    for plane in PLANES:
        phases[plane], _ = phase.calculate(meas_input, input_files, tunes, plane)
        bpm_names = input_files.bpms(plane=plane, dpp_value=0)
        for_rdts = _best_90_degree_phases(meas_input, bpm_names, phases, tunes, plane)
        LOGGER.info(f"Average phase advance between BPM pairs: {for_rdts.loc[:,'MEAS'].mean()}")
        for rdt in SINGLE_PLANE_RDTS[plane]:
            df = _process_rdt(meas_input, input_files, for_rdts, invariants, plane, rdt)
            write(df, header, meas_input, plane, rdt)
    for plane in PLANES:
        bpm_names = input_files.bpms(dpp_value=0)
        for_rdts = _best_90_degree_phases(meas_input, bpm_names, phases, tunes, plane)
        LOGGER.info(f"Average phase advance between BPM pairs: {for_rdts.loc[:, 'MEAS'].mean()}")
        for rdt in DOUBLE_PLANE_RDTS[plane]:
            df = _process_rdt(meas_input, input_files, for_rdts, invariants, plane, rdt)
            write(df, header, meas_input, plane, rdt)


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
    orders = dict(((1, "dipole"), (2, "quadrupole"), (3, "sextupole"), (4, "octupole")))
    return f"{rdt_type}_{orders[j + k + l + m]}"


def _best_90_degree_phases(meas_input, bpm_names, phases, tunes, plane):
    filtered = phases[plane]["MEAS"].loc[bpm_names, bpm_names]
    phase_meas = pd.concat(
        (filtered % 1, (filtered.iloc[:, :NBPMS_FOR_90] + tunes[plane]["Q"]) % 1), axis=1)
    second_bmps = np.abs(phase_meas * _get_n_upper_diagonals(NBPMS_FOR_90, phase_meas.shape)
                         - 0.25).idxmin(axis=1)
    filtered.iloc[-NBPMS_FOR_90:, :NBPMS_FOR_90] = (filtered.iloc[-NBPMS_FOR_90:,
                                                    :NBPMS_FOR_90] + tunes[plane]["Q"]) % 1
    filtered["NAME2"], filtered["MEAS"], filtered["ERRMEAS"] = second_bmps, filtered.lookup(
        bpm_names, second_bmps), phases[plane]["ERRMEAS"].lookup(bpm_names, second_bmps)
    for_rdts = pd.merge(filtered.loc[:, ["NAME2", "MEAS", "ERRMEAS"]],
                        meas_input.accelerator.get_model_tfs().loc[:, ["S"]], how="inner",
                        left_index=True, right_index=True)
    return for_rdts


def _get_n_upper_diagonals(n, shape):
    return diags(np.ones((n, shape[0])), np.arange(n)+1, shape=shape).toarray()


def _determine_line(rdt, plane):
    j, k, l, m = rdt
    lines = dict(X=(1 - j + k, m - l, 0), Y=(k - j, 1 - l + m, 0))
    return lines[plane]


def _process_rdt(meas_input, input_files, phase_data, invariants, plane, rdt):
    df = pd.DataFrame(phase_data)
    second_bpms = df.loc[:, "NAME2"].values
    df["S2"] = df.loc[second_bpms, "S"].values
    df["COUNT"] = len(input_files.dpp_frames(plane, 0))
    line = _determine_line(rdt, plane)
    phase_sign, suffix = get_line_sign_and_suffix(line, input_files, plane)
    comp_coeffs1 = to_complex(
        input_files.joined_frame(plane, [f"AMP{suffix}"], dpp_value=0).loc[df.index, :].values,
        phase_sign * input_files.joined_frame(plane, [f"PHASE{suffix}"], dpp_value=0).loc[df.index, :].values)
    # Multiples of tunes needs to be added to phase at second BPM if that is in second turn
    phase2 = phase_sign * input_files.joined_frame(plane, [f"PHASE{suffix}"], dpp_value=0).loc[second_bpms, :].values
    comp_coeffs2 = to_complex(
        input_files.joined_frame(plane, [f"AMP{suffix}"], dpp_value=0).loc[second_bpms, :].values,
        _add_tunes_if_in_second_turn(df, input_files, line, phase2))
    # Get amplitude and phase of the line from linx/liny file
    line_amp, line_phase, line_amp_e, line_phase_e = complex_secondary_lines(  # TODO use the errors
        df.loc[:, "MEAS"].values[:, np.newaxis] * meas_input.accelerator.get_beam_direction(),
        df.loc[:, "ERRMEAS"].values[:, np.newaxis], comp_coeffs1, comp_coeffs2)
    rdt_phases_per_file = _calculate_rdt_phases_from_line_phases(df, input_files, line, line_phase)
    rdt_angles = stats.circular_mean(rdt_phases_per_file, period=1, axis=1) % 1
    df[f"PHASE"] = rdt_angles
    df[f"{ERR}PHASE"] = stats.circular_error(rdt_phases_per_file, period=1, axis=1)
    df["AMP"], df[f"{ERR}AMP"] = _fit_rdt_amplitudes(invariants, line_amp, plane, rdt)
    df[f"REAL"] = np.cos(2 * np.pi * rdt_angles) * df.loc[:, "AMP"].values
    df[f"IMAG"] = np.sin(2 * np.pi * rdt_angles) * df.loc[:, "AMP"].values
    # in old files there was "EAMP" and "PHASE_STD"
    return df.loc[:, ["S", "COUNT", "AMP", f"{ERR}AMP", "PHASE", f"{ERR}PHASE", "REAL", "IMAG"]]


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
            phases[i] = input_files.joined_frame(plane, [f"MU{plane}"], dpp_value=0).loc[df.index, :].values % 1
    return line_phase - line[0] * phases[0] - line[1] * phases[1] + 0.25


def _fit_rdt_amplitudes(invariants, line_amp, plane, rdt):
    """
    Returns RDT amplitudes in units of meters ^ {1 - n/2}, where n is the order of RDT
    """
    amps, err_amps = np.empty(line_amp.shape[0]), np.empty(line_amp.shape[0])
    kick_data = get_linearized_problem(invariants, plane, rdt)  # corresponding to actions in meters
    guess = np.mean(line_amp / kick_data, axis=1)

    def fitting(x, f):
        return f * x

    for i, bpm_rdt_data in enumerate(line_amp):
        popt, pcov = curve_fit(fitting, kick_data, bpm_rdt_data, p0=guess[i])
        amps[i], err_amps[i] = popt[0], np.sqrt(pcov)[0]
    return amps, err_amps


def get_linearized_problem(invs, plane, rdt):
    """
    2 * j * f_jklm * (powers of 2Jx and 2Jy) : f_jklm is later a parameter of a fit
    we use sqrt(2J): unit is sqrt(um) ... divide by 1000
    """
    j, k, l, m = rdt
    act_x = invs["X"].T[0] / 1000
    act_y = invs["Y"].T[0] / 1000
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
    raise ValueError(f"No data for line {line} in plane {plane}")


def complex_secondary_lines(phase_adv, err_padv, sig1, sig2):
    """

    Args:
        phase_adv: phase advances between two BPMs
        err_padv: error on the phase advance between two BPMs
        sig1: Complex coefficients of a secondary lines at the first BPM of the pairs
        sig2: Complex coefficients of a secondary lines at the second BPM of the pairs

    Returns:
         amplitudes, phases err_amplitudes and err_phases of the complex signal
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
