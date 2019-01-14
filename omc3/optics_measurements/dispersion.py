"""
.. module: dispersion

Created on 28/06/18

:author: Lukas Malina

It computes orbit, dispersion and normalised dispersion.
"""
from os.path import join
import pandas as pd
import numpy as np
import tfs
from utils import stats


SCALES = {'um': 1.0e-6, 'mm': 1.0e-3, 'cm': 1.0e-2, 'm': 1.0}
PLANES = ("X", "Y")
PI2I = 2 * np.pi * complex(0, 1)


def calculate_orbit_and_dispersion(meas_input, input_files, tune_dict, model, beta_from_phase, header_dict):
    """
    Calculates orbit and dispersion, fills the following TfsFiles:
       getCOx.out        getCOy.out
       getNDx.out        getDx.out        getDy.out
    Args:
        meas_input: Optics_input object
        input_files: Stores the input files tfs
        tune_dict: Holds tunes and phase advances
        model:  Model tfs panda
        header_dict: OrderedDict containing information about the analysis
        beta_from_phase:

    Returns:

    """
    for plane in PLANES:
        orbit_header = _get_header(header_dict, tune_dict, 'getCO' + plane.lower() + '.out', orbit=True)
        dispersion_header = _get_header(header_dict, tune_dict, 'getD' + plane.lower() + '.out')
        _calculate_orbit(meas_input, input_files, model, plane, orbit_header)
        _calculate_dispersion(meas_input, input_files, model, plane, dispersion_header)
    ndx_header = _get_header(header_dict, tune_dict, 'getNDx.out', orbit=False)
    _calculate_normalised_dispersion(meas_input, input_files, model, beta_from_phase["X"], ndx_header)


def _get_header(header_dict, tune_dict, filename, orbit=False):
    header = header_dict.copy()
    if orbit:
        header['TABLE'] = 'ORBIT'
        header['TYPE'] = 'ORBIT'
    header['Q1'] = tune_dict["X"]["Q"]
    header['Q2'] = tune_dict["Y"]["Q"]
    header['FILENAME'] = filename
    return header


def _calculate_orbit(meas_input, input_files, model, plane, header):
    df_orbit = _get_merged_df(model, input_files, plane, ['CO', 'CORMS'])
    df_orbit[plane] = stats.weighted_mean(input_files.get_data(df_orbit, 'CO'), axis=1)
    df_orbit[f"STD{plane}"] = stats.weighted_error(input_files.get_data(df_orbit, 'CO'), axis=1)
    df_orbit[f"DELTA{plane}"] = df_orbit.loc[:, plane] - df_orbit.loc[:, f"{plane}MDL"]
    output_df = df_orbit.loc[:, _get_output_columns(plane, df_orbit)]
    tfs.write(join(meas_input.outputdir, header['FILENAME']), output_df, header, save_index='NAME')
    return output_df


def _calculate_dispersion(meas_input, input_files, model, plane, header, order=2):
    df_orbit = _get_merged_df(model, input_files, plane, ['CO', 'CORMS'])
    dpps = input_files.dpps(plane)
    if np.max(dpps) - np.min(dpps) == 0.0:
        return  # temporary solution
        # raise ValueError('Cannot calculate dispersion, only a single momentum data')
    fit = np.polyfit(dpps, SCALES[meas_input.orbit_unit] * input_files.get_data(df_orbit, 'CO').T, order, cov=True)
    # in the fit results the coefficients are sorted by power in decreasing order
    if order > 1:
        df_orbit['D2' + plane] = fit[0][-3, :].T
        df_orbit['STDD2' + plane] = np.sqrt(fit[1][-3, -3, :].T)
    df_orbit['D' + plane] = fit[0][-2, :].T
    df_orbit['STDD' + plane] = np.sqrt(fit[1][-2, -2, :].T)
    df_orbit[plane] = fit[0][-1, :].T
    df_orbit['STD' + plane] = np.sqrt(fit[1][-1, -1, :].T)
    # since we get variances from the fit, maybe we can include the variances of fitted points
    df_orbit = df_orbit.loc[np.abs(df_orbit.loc[:, plane]) < meas_input.max_closed_orbit*SCALES[meas_input.orbit_unit], :]
    df_orbit['DP' + plane] = _calculate_dp(model,
                                           df_orbit.loc[:, [f"D{plane}", f"STDD{plane}"]], plane)
    df_orbit = _get_delta_columns(df_orbit, plane)
    output_df = df_orbit.loc[:, _get_output_columns(plane, df_orbit)]
    tfs.write(join(meas_input.outputdir, header['FILENAME']), output_df, header, save_index='NAME')
    return output_df


def _calculate_normalised_dispersion(meas_input, input_files, model, beta, header):
    # TODO there are no errors from orbit
    plane = "X"
    df_orbit = _get_merged_df(model, input_files, plane, ['CO', 'CORMS', f"AMP{plane}"])
    df_orbit[f"ND{plane}MDL"] = df_orbit.loc[:, f"D{plane}MDL"] / np.sqrt(
        df_orbit.loc[:, f"BET{plane}MDL"])

    df_orbit = pd.merge(df_orbit, beta.loc[:, ['BETX', 'ERRBETX']], how='inner', left_index=True,
                        right_index=True)
    dpps = input_files.dpps(plane)
    if np.max(dpps) - np.min(dpps) == 0.0:
        return  # temporary solution
        # raise ValueError('Cannot calculate dispersion, only a single dpoverp')
    fit = np.polyfit(dpps, SCALES[meas_input.orbit_unit] * input_files.get_data(df_orbit, 'CO').T, 1, cov=True)
    df_orbit['NDX_unscaled'] = fit[0][-2, :].T / stats.weighted_mean(input_files.get_data(df_orbit, 'AMPX'), axis=1)  # TODO there is no error from AMPX
    df_orbit['STDNDX_unscaled'] = np.sqrt(fit[1][-2, -2, :].T) / stats.weighted_mean(input_files.get_data(df_orbit, 'AMPX'), axis=1)
    df_orbit = df_orbit.loc[np.abs(fit[0][-1, :].T) < meas_input.max_closed_orbit * SCALES[meas_input.orbit_unit], :]
    mask = meas_input.accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
    global_factor = np.sum(df_orbit.loc[mask, 'NDXMDL'].values) / np.sum(df_orbit.loc[mask, 'NDX_unscaled'].values)
    df_orbit['NDX'] = global_factor * df_orbit.loc[:, 'NDX_unscaled']
    df_orbit['STDNDX'] = global_factor * df_orbit.loc[:, 'STDNDX_unscaled']
    df_orbit = _calculate_from_norm_disp(df_orbit, model, plane)
    output_df = df_orbit.loc[:, _get_output_columns(plane, df_orbit)]
    tfs.write(join(meas_input.outputdir, header['FILENAME']), output_df, header, save_index='NAME')
    return output_df


def calculate_ndx_from_3d(meas_input, input_files, model, driven_model, beta, header):
    """It computes horizontal normalised dispersion from 3 D kicks,
    it performs model based compensation, i.e. as in _free2 files"""
    output, accelerator = meas_input.outputdir, meas_input.accelerator
    plane = "X"
    df_orbit = _get_merged_df(model, input_files, plane, ['AMPZ', 'MUZ'])
    df_orbit[f"ND{plane}MDL"] = df_orbit.loc[:, f"D{plane}MDL"] / np.sqrt(df_orbit.loc[:, f"BET{plane}MDL"])
    df_orbit = pd.merge(df_orbit, beta.loc[:, ['BETX', 'ERRBETX']], how='inner', left_index=True, right_index=True)
    df_orbit = pd.merge(df_orbit, driven_model.loc[:, ['BETX']], how='inner', left_index=True,
                        right_index=True, suffixes=('', '_driven'))
    mask = accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
    compensation = np.sqrt(df_orbit.loc[:, 'BETX_driven'].values / df_orbit.loc[:, f"BET{plane}MDL"].values)
    global_factors = np.sum(df_orbit.loc[mask, 'NDXMDL'].values) / np.sum(df_orbit.loc[mask, input_files.get_columns(df_orbit, 'AMPZ')].values * compensation[mask, None], axis=0)
    # scaling to the model, and getting the synchrotron phase in the arcs
    scaled_amps = (df_orbit.loc[:, input_files.get_columns(df_orbit, 'AMPZ')].values * global_factors) * compensation[:, None]
    df_orbit[f"ND{plane}"], df_orbit[f"STDND{plane}"] = _get_signed_dispersion(
        input_files, df_orbit, scaled_amps, mask)
    df_orbit = _calculate_from_norm_disp(df_orbit, model, plane)
    output_df = df_orbit.loc[:, _get_output_columns(plane, df_orbit)]
    tfs.write(join(output, header['FILENAME']), output_df, header, save_index='NAME')
    return output_df


def calculate_dx_from_3d(meas_input, input_files, model, header_dict, tune_dict):
    """It computes  dispersion from 3 D kicks"""
    output, accelerator = meas_input.outputdir, meas_input.accelerator
    dispersion_dfs = {}
    global_factors = None
    for plane in PLANES:
        df_orbit = _get_merged_df(model, input_files, plane, ['AMPZ', 'MUZ', f"AMP{plane}"])
        # work around due to scaling to main line in lin files
        unscaled_amps = (df_orbit.loc[:, input_files.get_columns(df_orbit, 'AMPZ')].values *
                         df_orbit.loc[:, input_files.get_columns(df_orbit, f"AMP{plane}")].values)
        mask = accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
        if plane == "X":
            # global factors are 1 / dppamp ... or up to factor 2 if complex spectra
            global_factors = np.sum(df_orbit.loc[mask, f"D{plane}MDL"].values) / np.sum(unscaled_amps[mask, :], axis=0)
        # scaling to the model, and getting the synchrotron phase in the arcs
        df_orbit[f"D{plane}"], df_orbit[f"STDD{plane}"] = _get_signed_dispersion(
                input_files, df_orbit, unscaled_amps * global_factors, mask)
        df_orbit['DP' + plane] = _calculate_dp(model, df_orbit.loc[:, ['D' + plane, 'STDD' + plane]], plane)
        df_orbit = _get_delta_columns(df_orbit, plane)
        output_df = df_orbit.loc[:, _get_output_columns(plane, df_orbit)]
        dispersion_dfs[plane] = output_df
        header = _get_header(header_dict, tune_dict, 'getD' + plane.lower() + '.out')
        tfs.write(join(output, header['FILENAME']), output_df, header, save_index='NAME')
    return dispersion_dfs


def _calculate_dp(model, disp, plane):
    df = pd.DataFrame(model).loc[:, ['S', 'MU' + plane, 'DP' + plane, 'D' + plane,
                                     'BET' + plane, 'ALF' + plane]]
    df = pd.merge(df, disp.loc[:, ['D' + plane, 'STDD' + plane]], how='inner', left_index=True,
                  right_index=True, suffixes=('', 'meas'))
    shifted = np.roll(df.index.values, -1)
    p_mdl_12 = df.loc[shifted, 'MU' + plane].values - df.loc[:, 'MU' + plane].values
    p_mdl_12[-1] = p_mdl_12[-1] + model['Q' + str(1+(plane == "Y"))]
    phi_12 = p_mdl_12 * 2 * np.pi
    m11 = np.sqrt(df.loc[shifted, 'BET' + plane] / df.loc[:, 'BET' + plane]
                  ) * (np.cos(phi_12) + df.loc[:, 'ALF' + plane] * np.sin(phi_12))
    m12 = np.sqrt(df.loc[shifted, 'BET' + plane] * df.loc[:, 'BET' + plane]) * np.sin(phi_12)
    m13 = df.loc[shifted, 'D' + plane] - m11 * df.loc[:, 'D' + plane] - m12 * df.loc[:, 'DP' + plane]
    return (-m13 + df.loc[shifted, 'D' + plane + 'meas'] - m11 * df.loc[:, 'D' + plane + 'meas']) / m12


def _get_merged_df(model, input_files, plane, meas_columns):
    df = pd.DataFrame(model).loc[:, ["S", plane, f"D{plane}", f"DP{plane}", f"MU{plane}", f"BET{plane}"]]
    df.rename(columns={plane: f"{plane}MDL", f"D{plane}": f"D{plane}MDL", f"DP{plane}": f"DP{plane}MDL",
                       f"MU{plane}": f"MU{plane}MDL", f"BET{plane}": f"BET{plane}MDL"}, inplace=True)
    df = pd.merge(df, input_files.joined_frame(plane, meas_columns), how='inner', left_index=True, right_index=True)
    df['COUNT'] = len(input_files.get_columns(df, meas_columns[0]))
    return df


def _get_signed_dispersion(input_files, df_orbit, scaled_amps, mask):
    same_interval_phase = np.angle(np.exp(PI2I * df_orbit.loc[:, input_files.get_columns(df_orbit, 'MUZ')].values)) / (2 * np.pi)
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
    cols = (["S", "COUNT", f"MU{plane}MDL"] +  # common columns
            _single_column_set_list(plane) +  # orbit columns
            _single_column_set_list(f"ND{plane}") +  # normalized dispersion columns
            _single_column_set_list(f"D{plane}") +  # dispersion columns
            [f"DP{plane}", f"DP{plane}MDL"] +  # more dispersion columns
            [f"D2{plane}", f"STDD2{plane}"])      # second order dispersion columns
    return [col for col in cols if col in df.columns]


def _single_column_set_list(base_name):
    return [f"{base_name}", f"STD{base_name}", f"DELTA{base_name}", f"{base_name}MDL"]


def _calculate_from_norm_disp(df, model, plane):
    df[f"D{plane}"] = df.loc[:, f"ND{plane}"] * np.sqrt(df.loc[:, f"BET{plane}"])
    df[f"STDD{plane}"] = df.loc[:, f"STDND{plane}"] * np.sqrt(df.loc[:, f"BET{plane}"])
    df[f"DP{plane}"] = _calculate_dp(model, df.loc[:, [f"D{plane}", f"STDD{plane}"]], plane)
    return _get_delta_columns(df, plane)


def _get_delta_columns(df, plane):
    for col in [f"{plane}", f"D{plane}", f"ND{plane}"]:
        if col in df.columns:
            df[f"DELTA{col}"] = df.loc[:, col] - df.loc[:, f"{col}MDL"]
    return df
