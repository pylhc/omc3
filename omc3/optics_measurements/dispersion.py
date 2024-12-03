"""
Dispersion
----------

This module contains dispersion calculations related functionality of ``optics_measurements``.
It provides functions to compute orbit, dispersion and normalised dispersion.
"""
from __future__ import annotations

from collections.abc import Sequence
from os.path import join
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs

from omc3.definitions.constants import PI2I
from omc3.optics_measurements.constants import (
    DELTA,
    DISPERSION_NAME,
    ERR,
    EXT,
    MDL,
    NORM_DISP_NAME,
    ORBIT_NAME,
)
from omc3.optics_measurements import dpp 
from omc3.utils import stats

if TYPE_CHECKING: 
    from generic_parser import DotDict

    from omc3.optics_measurements.data_models import InputFiles


def calculate_orbit(meas_input: DotDict, input_files: InputFiles, header, plane):
    """
    Calculates orbit.

    Args:
        meas_input: `OpticsInput` object
        input_files: Stores the input files tfs.
        header: `OrderedDict` containing information about the analysis.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        `TfsDataFrame` corresponding to output file.
    """
    df_orbit = _get_merged_df(meas_input, input_files, plane, ['CO', 'CORMS'])
    df_orbit[plane] = stats.weighted_mean(input_files.get_data(df_orbit, 'CO'), axis=1)
    df_orbit[f"{ERR}{plane}"] = stats.weighted_error(input_files.get_data(df_orbit, 'CO'), axis=1)
    df_orbit = _get_delta_columns(df_orbit, plane)
    output_df = df_orbit.loc[:, _get_output_columns(plane, df_orbit)]
    tfs.write(join(meas_input.outputdir, f"{ORBIT_NAME}{plane.lower()}{EXT}"), output_df, header, save_index='NAME')
    return output_df


def calculate_dispersion(meas_input: DotDict, input_files: InputFiles, header_dict, plane):
    """
    Calculates dispersion.

    Args:
        meas_input: `OpticsInput` object.
        input_files: Stores the input files tfs.
        header_dict: `OrderedDict` containing information about the analysis.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        `TfsDataFrame` corresponding to output file.
    """
    if meas_input.three_d_excitation:
        return _calculate_dispersion_3d(meas_input, input_files, header_dict, plane)
    return _calculate_dispersion_2d(meas_input, input_files, header_dict, plane)


def calculate_normalised_dispersion(meas_input: DotDict, input_files: InputFiles, beta, header_dict):
    """
    Calculates normalised dispersion.

    Args:
        meas_input: `OpticsInput` object.
        input_files: Stores the input files tfs.
        beta: measured betas to get dispersion from normalised dispersion.
        header_dict: `OrderedDict` containing information about the analysis.

    Returns:
        `TfsDataFrame` corresponding to output file.
    """
    if meas_input.three_d_excitation:
        return _calculate_normalised_dispersion_3d(meas_input, input_files, beta, header_dict)
    return _calculate_normalised_dispersion_2d(meas_input, input_files, beta, header_dict)


def _calculate_dispersion_2d(meas_input: DotDict, input_files: InputFiles, header, plane):
    dpps = input_files.dpps(plane)
    if _is_single_dpp_bin(dpps):
        return

    order = 2 if meas_input.second_order_dispersion else 1
    model = meas_input.accelerator.model
    df_orbit = _get_merged_df(meas_input, input_files, plane, ['CO', 'CORMS'])
    fit = np.polyfit(dpps, input_files.get_data(df_orbit, 'CO').T, order, cov=True)
    # in the fit results the coefficients are sorted by power in decreasing order
    if order > 1:
        df_orbit[f"D2{plane}"] = fit[0][-3, :].T
        df_orbit[f"{ERR}D2{plane}"] = np.sqrt(fit[1][-3, -3, :].T)
    df_orbit[f"D{plane}"] = fit[0][-2, :].T
    df_orbit[f"{ERR}D{plane}"] = np.sqrt(fit[1][-2, -2, :].T)
    df_orbit[plane] = fit[0][-1, :].T
    df_orbit[f"{ERR}{plane}"] = np.sqrt(fit[1][-1, -1, :].T)
    # since we get variances from the fit, maybe we can include the variances of fitted points
    df_orbit[f"DP{plane}"] = _calculate_dp(model,
                                           df_orbit.loc[:, [f"D{plane}", f"{ERR}D{plane}"]], plane)
    df_orbit = _get_delta_columns(df_orbit, plane)
    output_df = df_orbit.loc[:, _get_output_columns(plane, df_orbit)]
    tfs.write(join(meas_input.outputdir, f"{DISPERSION_NAME}{plane.lower()}{EXT}"), output_df, header, save_index='NAME')
    return output_df


def _calculate_dispersion_3d(meas_input: DotDict, input_files: InputFiles, header_dict, plane):
    """Computes dispersion from 3D kicks."""
    output, accelerator = meas_input.outputdir, meas_input.accelerator
    model = accelerator.model
    df_orbit = _get_merged_df(meas_input, input_files, plane, ['AMPZ', 'MUZ', f"AMP{plane}"])
    # work around due to scaling to main line in lin files
    unscaled_amps = (df_orbit.loc[:, input_files.get_columns(df_orbit, 'AMPZ')].to_numpy() *
                     df_orbit.loc[:, input_files.get_columns(df_orbit, f"AMP{plane}")].to_numpy())
    mask = accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
    global_factors = np.array([1 / df.headers["DPPAMP"] for df in input_files[plane]])
    # scaling to the model, and getting the synchrotron phase in the arcs
    df_orbit[f"D{plane}"], df_orbit[f"{ERR}D{plane}"] = _get_signed_dispersion(
            input_files, df_orbit, unscaled_amps * global_factors, mask)
    df_orbit[f"DP{plane}"] = _calculate_dp(model, df_orbit.loc[:, [f"D{plane}", f"{ERR}D{plane}"]], plane)
    df_orbit = _get_delta_columns(df_orbit, plane)
    output_df = df_orbit.loc[:, _get_output_columns(plane, df_orbit)]
    tfs.write(join(output, f"{DISPERSION_NAME}{plane.lower()}{EXT}"), output_df, header_dict, save_index='NAME')
    return output_df


def _calculate_normalised_dispersion_2d(meas_input: DotDict, input_files: InputFiles, beta, header):
    # TODO there are no errors from orbit
    plane = "X"
    
    dpps = input_files.dpps(plane)
    if _is_single_dpp_bin(dpps):
        return
    
    order = 2 if meas_input.second_order_dispersion else 1
    
    model = meas_input.accelerator.model
    df_orbit = _get_merged_df(meas_input, input_files, plane, ['CO', 'CORMS', f"AMP{plane}"])
    df_orbit[f"ND{plane}{MDL}"] = df_orbit.loc[:, f"D{plane}{MDL}"] / np.sqrt(
        df_orbit.loc[:, f"BET{plane}{MDL}"])
    if order > 1:
        df_orbit[f"ND2{plane}{MDL}"] = df_orbit.loc[:, f"D2{plane}{MDL}"] / np.sqrt(
            df_orbit.loc[:, f"BET{plane}{MDL}"])
    df_orbit = pd.merge(df_orbit, beta.loc[:, [f"BET{plane}", f"{ERR}BET{plane}"]], how='inner',
                        left_index=True, right_index=True)

    fit = np.polyfit(dpps, input_files.get_data(df_orbit, 'CO').T, order, cov=True)
    if order > 1:
        df_orbit['ND2X_unscaled'] = fit[0][-3, :].T / stats.weighted_mean(input_files.get_data(df_orbit, f"AMP{plane}"), axis=1)
        df_orbit['STDND2X_unscaled'] = np.sqrt(fit[1][-3, -3, :].T) / stats.weighted_mean(input_files.get_data(df_orbit, f"AMP{plane}"), axis=1)
    df_orbit['NDX_unscaled'] = fit[0][-2, :].T / stats.weighted_mean(input_files.get_data(df_orbit, f"AMP{plane}"), axis=1)  # TODO there is no error from AMPX
    df_orbit['STDNDX_unscaled'] = np.sqrt(fit[1][-2, -2, :].T) / stats.weighted_mean(input_files.get_data(df_orbit, f"AMP{plane}"), axis=1)
    mask = meas_input.accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
    global_factor = np.sum(df_orbit.loc[mask, f"ND{plane}{MDL}"].to_numpy()) / np.sum(df_orbit.loc[mask, 'NDX_unscaled'].to_numpy())
    if order > 1:
        df_orbit[f"ND2{plane}"] = global_factor * df_orbit.loc[:, 'ND2X_unscaled']
        df_orbit[f"{ERR}ND2{plane}"] = global_factor * df_orbit.loc[:, 'STDND2X_unscaled']
    df_orbit[f"ND{plane}"] = global_factor * df_orbit.loc[:, 'NDX_unscaled']
    df_orbit[f"{ERR}ND{plane}"] = global_factor * df_orbit.loc[:, 'STDNDX_unscaled']
    df_orbit = _calculate_from_norm_disp(df_orbit, model, plane)
    output_df = df_orbit.loc[:, _get_output_columns(plane, df_orbit)]
    tfs.write(join(meas_input.outputdir, f"{NORM_DISP_NAME}{plane.lower()}{EXT}"), output_df, header, save_index='NAME')
    return output_df


def _calculate_normalised_dispersion_3d(meas_input: DotDict, input_files: InputFiles, beta, header):
    """
    Computes horizontal normalised dispersion from 3D kicks, and performs model-based
    compensation, i.e. as in _free2 files.
    """
    output, accelerator = meas_input.outputdir, meas_input.accelerator
    model = accelerator.model
    driven_model = accelerator.model_driven if accelerator.excitation else model
    plane = "X"
    df_orbit = _get_merged_df(meas_input, input_files, plane, ['AMPZ', 'MUZ'])
    df_orbit[f"ND{plane}{MDL}"] = df_orbit.loc[:, f"D{plane}{MDL}"] / np.sqrt(df_orbit.loc[:, f"BET{plane}{MDL}"])
    df_orbit = pd.merge(df_orbit, beta.loc[:, [f"BET{plane}", f"{ERR}BET{plane}"]], how='inner', left_index=True, right_index=True)
    df_orbit = pd.merge(df_orbit, driven_model.loc[:, [f"BET{plane}"]], how='inner', left_index=True,
                        right_index=True, suffixes=('', '_driven'))
    mask = accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
    compensation = np.sqrt(df_orbit.loc[:, f"BET{plane}_driven"].to_numpy() / df_orbit.loc[:, f"BET{plane}{MDL}"].to_numpy())
    global_factors = np.sum(df_orbit.loc[mask, f"ND{plane}{MDL}"].to_numpy()) / np.sum(df_orbit.loc[mask, input_files.get_columns(df_orbit, 'AMPZ')].to_numpy() * compensation[mask, None], axis=0)
    # scaling to the model, and getting the synchrotron phase in the arcs
    scaled_amps = (df_orbit.loc[:, input_files.get_columns(df_orbit, 'AMPZ')].to_numpy() * global_factors) * compensation[:, None]
    df_orbit[f"ND{plane}"], df_orbit[f"{ERR}ND{plane}"] = _get_signed_dispersion(
        input_files, df_orbit, scaled_amps, mask)
    df_orbit = _calculate_from_norm_disp(df_orbit, model, plane)
    output_df = df_orbit.loc[:, _get_output_columns(plane, df_orbit)]
    tfs.write(join(output, f"{NORM_DISP_NAME}{plane.lower()}{EXT}"), output_df, header, save_index='NAME')
    return output_df


def _calculate_dp(model, disp, plane):
    _m = "meas"
    df = pd.DataFrame(model).loc[:, ['S', f"MU{plane}", f"DP{plane}", f"D{plane}",
                                     f"BET{plane}", f"ALF{plane}"]]
    df = pd.merge(df, disp.loc[:, [f"D{plane}", f"{ERR}D{plane}"]], how='inner', left_index=True,
                  right_index=True, suffixes=('', _m))
    shifted = np.roll(df.index.to_numpy(), -1)
    p_mdl_12 = df.loc[shifted, f"MU{plane}"].to_numpy() - df.loc[:, f"MU{plane}"].to_numpy()
    p_mdl_12[-1] = p_mdl_12[-1] + model['Q' + str(1+(plane == "Y"))]
    phi_12 = p_mdl_12 * 2 * np.pi
    m11 = np.sqrt(df.loc[shifted, f"BET{plane}"] / df.loc[:, f"BET{plane}"]
                  ) * (np.cos(phi_12) + df.loc[:, f"ALF{plane}"] * np.sin(phi_12))
    m12 = np.sqrt(df.loc[shifted, f"BET{plane}"] * df.loc[:, f"BET{plane}"]) * np.sin(phi_12)
    m13 = df.loc[shifted, f"D{plane}"] - m11 * df.loc[:, f"D{plane}"] - m12 * df.loc[:, f"DP{plane}"]
    return (-m13 + df.loc[shifted, f"D{plane}{_m}"] - m11 * df.loc[:, f"D{plane}{_m}"]) / m12


def _get_merged_df(meas_input, input_files, plane, meas_columns):
    model = meas_input.accelerator.model
    df = pd.DataFrame(model).reindex(columns=["S", plane, f"D{plane}", f"DP{plane}", f"MU{plane}",
                                              f"BET{plane}", f"DD{plane}"], fill_value=np.nan)
    df.rename(columns={plane: f"{plane}{MDL}", f"D{plane}": f"D{plane}{MDL}", f"DP{plane}": f"DP{plane}{MDL}",
                       f"MU{plane}": f"MU{plane}{MDL}", f"BET{plane}": f"BET{plane}{MDL}", f"DD{plane}": f"D2{plane}{MDL}"}, inplace=True)
    if not meas_input.second_order_dispersion:
        df.drop(columns=[f'D2{plane}{MDL}'], inplace=True)
    df = pd.merge(df, input_files.joined_frame(plane, meas_columns, dpp_amp=meas_input.three_d_excitation), how='inner', left_index=True, right_index=True)
    df['COUNT'] = len(input_files.get_columns(df, meas_columns[0]))
    return df


def _get_signed_dispersion(input_files, df_orbit, scaled_amps, mask):
    same_interval_phase = np.angle(np.exp(PI2I * df_orbit.loc[:, input_files.get_columns(df_orbit, 'MUZ')].to_numpy())) / (2 * np.pi)
    phase_wrt_arcs = same_interval_phase - stats.circular_mean(same_interval_phase[mask, :], period=1, axis=0)
    phase_wrt_arcs = np.abs(np.where(np.abs(phase_wrt_arcs) > 0.5, phase_wrt_arcs - np.sign(phase_wrt_arcs), phase_wrt_arcs))
    if len(input_files.get_columns(df_orbit, 'AMPZ')) > 1:
        # resolving the sign of dispersion
        dispersions = scaled_amps * np.sign(0.25 - np.abs(stats.circular_mean(phase_wrt_arcs, period=1, axis=1)))[:, None]
        # final calculation
        return np.mean(dispersions, axis=1), np.std(dispersions, axis=1) * stats.t_value_correction(
            dispersions.shape[1])
    return scaled_amps * np.sign(0.25 - np.abs(phase_wrt_arcs)), 0.0


def _get_output_columns(plane, df):
    cols = (["S", "COUNT", f"MU{plane}MDL"] +           # common columns
            _single_column_set_list(plane) +            # orbit columns
            _single_column_set_list(f"ND{plane}") +     # normalized dispersion columns
            _single_column_set_list(f"D{plane}") +      # dispersion columns
            [f"DP{plane}", f"DP{plane}{MDL}"] +         # more dispersion columns
            _single_column_set_list(f"D2{plane}") +     # second order dispersion columns
            _single_column_set_list(f"ND2{plane}"))     # second order normalized dispersion columns
    return [col for col in cols if col in df.columns]


def _single_column_set_list(base_name):
    return [f"{base_name}", f"{ERR}{base_name}", f"{DELTA}{base_name}", f"{ERR}{DELTA}{base_name}", f"{base_name}{MDL}"]


def _calculate_from_norm_disp(df, model, plane):
    df[f"D{plane}"] = df.loc[:, f"ND{plane}"] * np.sqrt(df.loc[:, f"BET{plane}"])
    df[f"{ERR}D{plane}"] = df.loc[:, f"{ERR}ND{plane}"] * np.sqrt(df.loc[:, f"BET{plane}"])
    df[f"DP{plane}"] = _calculate_dp(model, df.loc[:, [f"D{plane}", f"{ERR}D{plane}"]], plane)
    return _get_delta_columns(df, plane)


def _get_delta_columns(df, plane):
    for col in [f"{plane}", f"D{plane}", f"ND{plane}", f"D2{plane}", f"ND2{plane}"]:
        if col in df.columns:
            df[f"{DELTA}{col}"] = df.loc[:, col] - df.loc[:, f"{col}{MDL}"]
            df[f"{ERR}{DELTA}{col}"] = df.loc[:, f"{ERR}{col}"]
    return df


def _is_single_dpp_bin(dpps: Sequence[float], tolerance: float = dpp.DPP_BIN_TOLERANCE) -> bool:
    """ Checks if the files would be grouped into a single dpp-bin 
    by :func:`omc3.optics_measurements.dpp._compute_ranges`. """
    # alternatively: len(omc3.optics_measurements.dpp._compute_ranges(dpps, tolerance)) == 1 
    return np.abs(np.max(dpps) - np.min(dpps)) <= 2 * tolerance
