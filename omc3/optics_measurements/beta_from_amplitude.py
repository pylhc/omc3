"""
Beta from Amplitude
--------------------

This module contains some of the beta calculation related functionality of ``optics_measurements``.
It provides functions to calculate beta functions from amplitude data.
"""
from __future__ import annotations
from os.path import join

import numpy as np
import pandas as pd
import tfs

from omc3.optics_measurements.constants import (AMP_BETA_NAME, DELTA, ERR, EXT,
                                                MDL, RES)
from omc3.optics_measurements.toolbox import df_ratio, df_rel_diff

from typing import TYPE_CHECKING 

if TYPE_CHECKING: 
    from generic_parser import DotDict 
    from omc3.optics_measurements.data_models import InputFiles


def calculate(meas_input: DotDict, input_files: InputFiles, tune_dict, beta_phase, header_dict, plane):
    """
    Calculates beta and fills the following `TfsFiles`: ``f"{AMP_BETA_NAME}{plane.lower()}{EXT}"``

    Args:
        meas_input: `OpticsInput` object.
        input_files: `InputFiles` object contains measurement files.
        tune_dict: `TuneDict` contains measured tunes.
        beta_phase: contains beta functions from measured from phase.
        header_dict: dictionary of header items common for all output files.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
    """
    beta_amp = beta_from_amplitude(meas_input, input_files, plane, tune_dict)
    x_ratio = phase_to_amp_ratio(meas_input, beta_phase, beta_amp, plane)
    beta_amp = add_rescaled_beta_columns(beta_amp, x_ratio, plane)
    header_d = _get_header(header_dict, np.std(beta_amp.loc[:, f"{DELTA}BET{plane}"].to_numpy()), x_ratio)
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
    df[f"BET{plane}{RES}"] = df.loc[:, f"BET{plane}"].to_numpy() * ratio
    df[f"{ERR}BET{plane}{RES}"] = df.loc[:, f"{ERR}BET{plane}"].to_numpy() * ratio
    return df


def beta_from_amplitude(meas_input, input_files, plane, tunes):
    df = pd.DataFrame(meas_input.accelerator.model).loc[:, ["S", f"MU{plane}", f"BET{plane}"]]
    df.rename(columns={f"MU{plane}": f"MU{plane}{MDL}",
                       f"BET{plane}": f"BET{plane}{MDL}"}, inplace=True)
    dpp_value = meas_input.dpp if "dpp" in meas_input.keys() else 0
    df = pd.merge(df, input_files.joined_frame(plane, [f"AMP{plane}", f"MU{plane}"], dpp_value=dpp_value),
                  how='inner', left_index=True, right_index=True)
    df['COUNT'] = len(input_files.get_columns(df, f"AMP{plane}"))

    if meas_input.compensation == "model":
        df = _compensate_by_model(input_files, meas_input, df, plane)
    if meas_input.compensation == "equation":
        df = _compensate_by_equation(input_files, meas_input, df, plane, tunes)

    amps_squared = np.square(input_files.get_data(df, f"AMP{plane}"))
    mask = meas_input.accelerator.get_element_types_mask(df.index, ["arc_bpm"])
    actions = amps_squared / df.loc[:, f"BET{plane}{MDL}"].to_numpy()[:, np.newaxis]
    betas = amps_squared / np.mean(actions[mask], axis=0, keepdims=True)
    df[f"BET{plane}"] = np.mean(betas, axis=1)
    df[f"{ERR}BET{plane}"] = np.std(betas, axis=1)
    df[f"{DELTA}BET{plane}"] = df_rel_diff(df, f"BET{plane}", f"BET{plane}{MDL}")
    df[f"{ERR}{DELTA}BET{plane}"] = df_ratio(df, f"{ERR}BET{plane}", f"BET{plane}{MDL}")
    return df.loc[:, ['S', 'COUNT', f"BET{plane}", f"{ERR}BET{plane}", f"BET{plane}{MDL}",
                      f"MU{plane}{MDL}", f"{DELTA}BET{plane}", f"{ERR}{DELTA}BET{plane}"]]


def _compensate_by_equation(input_files, meas_input, df, plane, tunes):
    phases_meas = input_files.get_data(df, f"MU{plane}") * meas_input.accelerator.beam_direction
    driven_tune, _free_tune, ac2bpmac = tunes[plane]["Q"], tunes[plane]["QF"], tunes[plane]["ac2bpm"]
    k_bpmac = ac2bpmac[2]
    phase_corr = ac2bpmac[1] - phases_meas[k_bpmac] + (0.5 * driven_tune)
    phases_meas = phases_meas + phase_corr[np.newaxis, :]
    r = tunes.get_lambda(plane)
    phases_meas[k_bpmac:, :] = phases_meas[k_bpmac:, :] - driven_tune
    amp_compensation = np.sqrt((1 + r ** 2 + 2 * r * np.cos(4 * np.pi * phases_meas)) / (1 - r ** 2))
    df[input_files.get_columns(df, f"AMP{plane}")] = input_files.get_data(df, f"AMP{plane}") * amp_compensation
    return df


def _compensate_by_model(input_files, meas_input, df, plane):
    df = pd.merge(df, pd.DataFrame(meas_input.accelerator.model_driven.loc[:, [f"BET{plane}"]]
                                   .rename(columns={f"BET{plane}": f"BET{plane}comp"})),
                  how='inner', left_index=True, right_index=True)
    amp_compensation = np.sqrt(df_ratio(df, f"BET{plane}{MDL}", f"BET{plane}comp"))
    df[input_files.get_columns(df, f"AMP{plane}")] = (input_files.get_data(df, f"AMP{plane}")
                                                      * amp_compensation[:, np.newaxis])
    return df


def _get_header(header_dict, rmsbbeat, scaling_factor):
    header = header_dict.copy()
    header['RMSbetabeat'] = rmsbbeat
    header['RescalingFactor'] = scaling_factor
    return header
