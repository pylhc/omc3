"""
.. module: beta_from_amplitude

Created on 05/07/18

:author: Lukas Malina

It computes beta from amplitude.
"""
from os.path import join
import numpy as np
import pandas as pd
import tfs
from optics_measurements.toolbox import df_rel_diff, df_ratio
from optics_measurements.constants import AMP_BETA_NAME, EXT, ERR, DELTA, MDL


def calculate(meas_input, input_files, tune_dict, beta_phase, header_dict, plane):
    """
    Calculates beta and fills the following TfsFiles:
        f"{AMP_BETA_NAME}{plane.lower()}{EXT}"
    Parameters:
        'measure_input': OpticsInput object
        'input_files': InputFiles object contains measurement files
        'tune_d': TuneDict contains measured tunes
        'beta_phase': contains beta functions from measured from phase
        'header_dict': dictionary of header items common for all output files
        'plane': plane
    Returns:
    """
    if meas_input.compensation == "none" and meas_input.accelerator.excitation:
        model = meas_input.accelerator.get_driven_tfs()
    else:
        model = meas_input.accelerator.get_model_tfs()
    comp_dict = {"equation": dict(eq_comp=tune_dict),
                 "model": dict(model_comp=meas_input.accelerator.get_driven_tfs()),
                 "none": dict()}
    beta_amp = beta_from_amplitude(meas_input, input_files, model, plane, **comp_dict[meas_input.compensation])
    x_ratio = phase_to_amp_ratio(meas_input, beta_phase, beta_amp, plane)
    beta_amp = add_rescaled_beta_columns(beta_amp, x_ratio, plane)
    header_d = _get_header(header_dict, tune_dict, np.std(beta_amp.loc[:, f"{DELTA}BET{plane}"].values), x_ratio, free=(meas_input.compensation != "none"))
    tfs.write(join(meas_input.outputdir, f"{AMP_BETA_NAME}{plane.lower()}{EXT}"), beta_amp, header_d, save_index='NAME')
    return x_ratio


def phase_to_amp_ratio(measure_input, beta_phase, beta_amp, plane):
    ratio = pd.merge(beta_phase.loc[:, [f"BET{plane}"]], beta_amp.loc[:, [f"BET{plane}"]],
                     how='inner', left_index=True, right_index=True, suffixes=("ph", "amp"))
    ph_over_amp = df_ratio(ratio, f"BET{plane}ph", f"BET{plane}amp")
    mask = (np.array(0.1 < np.abs(ph_over_amp)) & np.array(np.abs(ph_over_amp) < 10.0) &
            np.array(measure_input.accelerator.get_element_types_mask(ratio.index, ["arc_bpm"])))
    x_ratio = np.mean(ph_over_amp[mask])
    return x_ratio


def add_rescaled_beta_columns(df, ratio, plane):
    df[f"BET{plane}RES"] = df.loc[:, f"BET{plane}"].values * ratio
    df[f"{ERR}BET{plane}RES"] = df.loc[:, f"{ERR}BET{plane}"].values * ratio
    return df


def beta_from_amplitude(meas_input, input_files, model, plane, eq_comp=None, model_comp=None):
    df = pd.DataFrame(model).loc[:, ["S", f"MU{plane}", f"BET{plane}"]]
    df.rename(columns={f"MU{plane}": f"MU{plane}{MDL}",
                       f"BET{plane}": f"BET{plane}{MDL}"}, inplace=True)
    df = pd.merge(df, input_files.joined_frame(plane, [f"AMP{plane}", f"MU{plane}"]),
                           how='inner', left_index=True, right_index=True)
    if model_comp is not None:
        df = pd.merge(df, pd.DataFrame(model_comp.loc[:, ["S", f"BET{plane}"]].rename(columns={f"BET{plane}": f"BET{plane}comp"})),
                      how='inner', left_index=True, right_index=True)
        amp_compensation = np.sqrt(df_ratio(df, f"BET{plane}{MDL}", f"BET{plane}comp"))
        df[input_files.get_columns(df, f"AMP{plane}")] = input_files.get_data(df, f"AMP{plane}") * amp_compensation[:, np.newaxis]

    df['COUNT'] = len(input_files.get_columns(df, f"AMP{plane}"))
    df[f"AMP{plane}"] = np.mean(input_files.get_data(df, f"AMP{plane}"), axis=1)

    if eq_comp is not None:
        phases_meas = input_files.get_data(df, f"MU{plane}") * meas_input.accelerator.get_beam_direction()
        driven_tune, free_tune, ac2bpmac = eq_comp[plane]["Q"], eq_comp[plane]["QF"], eq_comp[plane]["ac2bpm"]
        k_bpmac = ac2bpmac[2]
        phase_corr = ac2bpmac[1] - phases_meas[k_bpmac] + (0.5 * driven_tune)
        phases_meas = phases_meas + phase_corr[np.newaxis, :]
        r = eq_comp.get_lambda(plane)
        phases_meas[k_bpmac:, :] = phases_meas[k_bpmac:, :] - driven_tune
        for_sqrt2j = input_files.get_data(df, f"AMP{plane}") / np.sqrt(
            df.loc[:, f"BET{plane}{MDL}"].values[:, np.newaxis])
        sqrt2j = np.mean(for_sqrt2j[meas_input.accelerator.get_element_types_mask(df.index, ["arc_bpm"])], axis=0)
        betall = (np.square(
            (input_files.get_data(df, f"AMP{plane}").T / sqrt2j[:, np.newaxis]).T) *
                  (1 + r ** 2 + 2 * r * np.cos(4 * np.pi * phases_meas)) / (1 - r ** 2))
        df[f"BET{plane}"] = np.mean(betall, axis=1)
        df[f"{ERR}BET{plane}"] = np.std(betall, axis=1)
    else:
        # amplitudes are first averaged over files then squared and averaged over BPMs
        kick = np.mean(np.square(df.loc[:, f"AMP{plane}"].values) / df.loc[:, f"BET{plane}{MDL}"].values)
        # amplitudes are first squared then averaged
        kick2 = np.mean(np.square(input_files.get_data(df, f"AMP{plane}")) / df.loc[:, f"BET{plane}{MDL}"].values[:, np.newaxis], axis=0)
        df[f"BET{plane}"] = np.square(df.loc[:, f"AMP{plane}"].values) / kick
        df[f"{ERR}BET{plane}"] = np.std((np.square(input_files.get_data(df, f"AMP{plane}")).T / kick2[:, np.newaxis]).T, axis=1)

    df[f"{DELTA}BET{plane}"] = df_rel_diff(df, f"BET{plane}", f"BET{plane}{MDL}")
    df[f"{ERR}{DELTA}BET{plane}"] = df_ratio(df, f"{ERR}BET{plane}", f"BET{plane}{MDL}")
    return df.loc[:, ['S', 'COUNT', f"BET{plane}", f"{ERR}BET{plane}", f"BET{plane}{MDL}",
                      f"MU{plane}{MDL}", f"{DELTA}BET{plane}", f"{ERR}{DELTA}BET{plane}"]]


def _get_header(header_dict, tune_d, rmsbbeat, scaling_factor, free=True):
    header = header_dict.copy()
    if free:
        header['Q1'] = tune_d["X"]["QF"]
        header['Q2'] = tune_d["Y"]["QF"]
    else:
        header['Q1'] = tune_d["X"]["Q"]
        header['Q2'] = tune_d["Y"]["Q"]
    header['RMSbetabeat'] = rmsbbeat
    header['RescalingFactor'] = scaling_factor
    return header
