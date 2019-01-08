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
        _calculate_orbit(model, input_files, plane, orbit_header, meas_input.outputdir)
        _calculate_dispersion(model, input_files, plane, dispersion_header, meas_input.orbit_unit, meas_input.max_closed_orbit, meas_input.outputdir)
    ndx_header = _get_header(header_dict, tune_dict, 'getNDx.out', orbit=False)
    _calculate_normalised_dispersion(model, input_files, beta_from_phase["X"], ndx_header,
                                     meas_input.orbit_unit, meas_input.max_closed_orbit, meas_input.outputdir, meas_input.accelerator)


def _get_header(header_dict, tune_dict, filename, orbit=False):
    header = header_dict.copy()
    if orbit:
        header['TABLE'] = 'ORBIT'
        header['TYPE'] = 'ORBIT'
    header['Q1'] = tune_dict["X"]["Q"]
    header['Q2'] = tune_dict["Y"]["Q"]
    header['FILENAME'] = filename
    return header


def _calculate_orbit(model, input_files, plane, header, output):
    df_orbit = pd.DataFrame(model).loc[:, ['S', 'MU' + plane, plane]]
    df_orbit.rename(columns={'MU' + plane: 'MU' + plane + 'MDL', plane: plane + 'MDL'}, inplace=True)
    df_orbit = pd.merge(df_orbit, input_files.joined_frame(plane, ['CO', 'CORMS']), how='inner',
                        left_index=True, right_index=True)
    df_orbit['COUNT'] = len(input_files.get_columns(df_orbit, 'CO'))
    df_orbit[plane] = stats.weighted_mean(input_files.get_data(df_orbit, 'CO'), axis=1)
    df_orbit['STD' + plane] = stats.weighted_error(input_files.get_data(df_orbit, 'CO'), axis=1)
    df_orbit['DELTA' + plane] = df_orbit.loc[:, plane] - df_orbit.loc[:, plane + 'MDL']
    output_df = df_orbit.loc[:, ['S', 'COUNT', plane, 'STD' + plane, plane + 'MDL', 'MU' + plane + 'MDL', 'DELTA' + plane]]
    tfs.write(join(output, header['FILENAME']), output_df, header, save_index='NAME')
    return output_df


def _calculate_dispersion(model, input_files, plane, header, unit, cut, output, order=2):
    df_orbit = pd.DataFrame(model).loc[:, ['S', 'MU' + plane, 'DP' + plane, 'D' + plane, plane]]
    df_orbit.rename(columns={'MU' + plane: 'MU' + plane + 'MDL', 'DP' + plane: 'DP' + plane + 'MDL',
                             'D' + plane: 'D' + plane + 'MDL', plane: plane + 'MDL'}, inplace=True)
    df_orbit = pd.merge(df_orbit, input_files.joined_frame(plane, ['CO', 'CORMS']), how='inner',
                        left_index=True, right_index=True)
    df_orbit['COUNT'] = len(input_files.get_columns(df_orbit, 'CO'))
    dpps = input_files.dpps(plane)
    if np.max(dpps) - np.min(dpps) == 0.0:
        return  # temporary solution
        # raise ValueError('Cannot calculate dispersion, only a single momentum data')
    fit = np.polyfit(dpps, SCALES[unit] * input_files.get_data(df_orbit, 'CO').T, order, cov=True)
    # in the fit results the coefficients are sorted by power in decreasing order
    if order > 1:
        df_orbit['D2' + plane] = fit[0][-3, :].T
        df_orbit['STDD2' + plane] = np.sqrt(fit[1][-3, -3, :].T)
    df_orbit['D' + plane] = fit[0][-2, :].T
    df_orbit['STDD' + plane] = np.sqrt(fit[1][-2, -2, :].T)
    df_orbit[plane] = fit[0][-1, :].T
    df_orbit['STD' + plane] = np.sqrt(fit[1][-1, -1, :].T)
    # since we get variances from the fit, maybe we can include the variances of fitted points
    df_orbit = df_orbit.loc[np.abs(df_orbit.loc[:, plane]) < cut*SCALES[unit], :]
    df_orbit['DP' + plane] = _calculate_dp(model,
                                           df_orbit.loc[:, ['D' + plane, 'STDD' + plane]], plane)
    df_orbit['DELTAD' + plane] = df_orbit.loc[:, 'D' + plane] - df_orbit.loc[:, 'D' + plane + 'MDL']
    if order > 1:
        output_df = df_orbit.loc[
                    :,
                    ['S', 'COUNT', 'D2' + plane, 'STDD2' + plane, 'D' + plane, 'STDD' + plane, plane, 'STD' + plane, 'DP' + plane,
                     'D' + plane + 'MDL', 'DP' + plane + 'MDL', 'MU' + plane + 'MDL',
                     'DELTAD' + plane]]
    else:
        output_df = df_orbit.loc[
                :, ['S', 'COUNT', 'D' + plane, 'STDD' + plane, plane, 'STD' + plane, 'DP' + plane,
                    'D' + plane + 'MDL', 'DP' + plane + 'MDL', 'MU' + plane + 'MDL', 'DELTAD' + plane]]
    tfs.write(join(output, header['FILENAME']), output_df, header, save_index='NAME')
    return output_df


def _calculate_normalised_dispersion(model, input_files, beta, header, unit, cut, output, accelerator):
    # TODO there are no errors from orbit
    df_orbit = pd.DataFrame(model).loc[:, ['S', 'MUX', 'DPX', 'DX', 'X', 'BETX']]
    df_orbit['NDXMDL'] = df_orbit.loc[:, 'DX'] / np.sqrt(df_orbit.loc[:, 'BETX'])
    df_orbit.rename(columns={'MUX': 'MUXMDL', 'DPX': 'DPXMDL', 'DX': 'DXMDL', 'X': 'XMDL'}, inplace=True)
    df_orbit['COUNT'] = len(input_files.get_columns(df_orbit, 'CO'))
    dpps = input_files.dpps("X")
    df_orbit = pd.merge(df_orbit, input_files.joined_frame("X", ['CO', 'CORMS', 'AMPX']),
                        how='inner', left_index=True, right_index=True)
    df_orbit = pd.merge(df_orbit, beta.loc[:, ['BETX', 'ERRBETX']], how='inner', left_index=True,
                        right_index=True, suffixes=('', '_phase'))
    if np.max(dpps) - np.min(dpps) == 0.0:
        return  # temporary solution
        # raise ValueError('Cannot calculate dispersion, only a single dpoverp')
    fit = np.polyfit(dpps, SCALES[unit] * input_files.get_data(df_orbit, 'CO').T, 1, cov=True)
    df_orbit['NDX_unscaled'] = fit[0][-2, :].T / stats.weighted_mean(input_files.get_data(df_orbit, 'AMPX'), axis=1)  # TODO there is no error from AMPX
    df_orbit['STDNDX_unscaled'] = np.sqrt(fit[1][-2, -2, :].T) / stats.weighted_mean(input_files.get_data(df_orbit, 'AMPX'), axis=1)
    df_orbit = df_orbit.loc[np.abs(fit[0][-1, :].T) < cut * SCALES[unit], :]
    mask = accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
    global_factor = np.sum(df_orbit.loc[mask, 'NDXMDL'].values) / np.sum(df_orbit.loc[mask, 'NDX_unscaled'].values)
    df_orbit['NDX'] = global_factor * df_orbit.loc[:, 'NDX_unscaled']
    df_orbit['STDNDX'] = global_factor * df_orbit.loc[:, 'STDNDX_unscaled']
    df_orbit['DX'] = df_orbit.loc[:, 'NDX'] * np.sqrt(df_orbit.loc[:, 'BETX_phase'])
    df_orbit['STDDX'] = df_orbit.loc[:, 'STDNDX'] * np.sqrt(df_orbit.loc[:, 'BETX_phase'])
    df_orbit['DPX'] = _calculate_dp(model, df_orbit.loc[:, ['DX', 'STDDX']], "X")
    df_orbit['DELTANDX'] = df_orbit.loc[:, 'NDX'] - df_orbit.loc[:, 'NDXMDL']
    df_orbit['DELTADX'] = df_orbit.loc[:, 'DX'] - df_orbit.loc[:, 'DXMDL']
    output_df = df_orbit.loc[:, ['S', 'COUNT', 'NDX', 'STDNDX', 'DX', 'DPX',
                                 'NDXMDL', 'DXMDL', 'DPXMDL', 'MUXMDL', 'DELTANDX', 'DELTADX']]
    tfs.write(join(output, header['FILENAME']), output_df, header, save_index='NAME')
    return output_df


def calculate_ndx_from_3d(meas_input, input_files, model, driven_model, beta, header):
    """It computes horizontal normalised dispersion from 3 D kicks,
    it performs model based compensation, i.e. as in _free2 files"""
    output, accelerator = meas_input.outputdir, meas_input.accelerator
    df_orbit = pd.DataFrame(model).loc[:, ['S', 'MUX', 'DPX', 'DX', 'X', 'BETX']]
    df_orbit['NDXMDL'] = df_orbit.loc[:, 'DX'] / np.sqrt(df_orbit.loc[:, 'BETX'])
    df_orbit.rename(columns={'MUX': 'MUXMDL', 'DPX': 'DPXMDL', 'DX': 'DXMDL', 'X': 'XMDL'},
                    inplace=True)
    df_orbit['COUNT'] = len(input_files.get_columns(df_orbit, 'AMPZ'))
    df_orbit = pd.merge(df_orbit, input_files.joined_frame("X", ['AMPZ', 'MUZ']),
                        how='inner', left_index=True, right_index=True)
    df_orbit = pd.merge(df_orbit, beta.loc[:, ['BETX', 'ERRBETX']], how='inner', left_index=True,
                        right_index=True, suffixes=('', '_phase'))
    df_orbit = pd.merge(df_orbit, driven_model.loc[:, ['BETX']], how='inner', left_index=True,
                        right_index=True, suffixes=('', '_driven'))
    mask = accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
    compensation = np.sqrt(df_orbit.loc[:, 'BETX_driven'].values / df_orbit.loc[:, 'BETX'].values)
    global_factors = np.sum(df_orbit.loc[mask, 'NDXMDL'].values) / np.sum(df_orbit.loc[mask, input_files.get_columns(df_orbit, 'AMPZ')].values * compensation[mask, None], axis=0)
    # scaling to the model, and getting the synchrotron phase in the arcs
    scaled_amps = (df_orbit.loc[:, input_files.get_columns(df_orbit, 'AMPZ')].values * global_factors) * compensation[:, None]
    same_interval_phase = np.angle(np.exp(PI2I * df_orbit.loc[:, input_files.get_columns(df_orbit, 'MUZ')].values)) / (2 * np.pi)
    phase_wrt_arcs = same_interval_phase - stats.circular_mean(same_interval_phase[mask, :], period=1, axis=0)
    phase_wrt_arcs = np.abs(np.where(np.abs(phase_wrt_arcs) > 0.5, phase_wrt_arcs - np.sign(phase_wrt_arcs), phase_wrt_arcs))
    if len(input_files.get_columns(df_orbit, 'AMPZ')) > 1:
        # resolving the sign of dispersion
        scaled_signed_amps = scaled_amps * np.sign(
            0.25 - np.abs(stats.circular_mean(phase_wrt_arcs, period=1, axis=1)))[:, None]
        # final calculation
        df_orbit['STDNDX'] = np.std(scaled_signed_amps, axis=1) * stats.t_value_correction(scaled_signed_amps.shape[1])
        df_orbit['NDX'] = np.mean(scaled_signed_amps, axis=1)
    else:
        df_orbit['STDNDX'] = 0.0
        df_orbit['NDX'] = scaled_amps * np.sign(0.25 - np.abs(phase_wrt_arcs))
    df_orbit['DX'] = df_orbit.loc[:, 'NDX'] * np.sqrt(df_orbit.loc[:, 'BETX_phase'])
    df_orbit['STDDX'] = df_orbit.loc[:, 'STDNDX'] * np.sqrt(df_orbit.loc[:, 'BETX_phase'])
    df_orbit['DPX'] = _calculate_dp(model, df_orbit.loc[:, ['DX', 'STDDX']], "X")
    df_orbit['DELTANDX'] = df_orbit.loc[:, 'NDX'] - df_orbit.loc[:, 'NDXMDL']
    df_orbit['DELTADX'] = df_orbit.loc[:, 'DX'] - df_orbit.loc[:, 'DXMDL']
    output_df = df_orbit.loc[:, ['S', 'COUNT', 'NDX', 'STDNDX', 'DX', 'STDDX', 'DPX',
                                 'NDXMDL', 'DXMDL', 'DPXMDL', 'MUXMDL', 'DELTANDX', 'DELTADX']]
    tfs.write(join(output, header['FILENAME']), output_df, header, save_index='NAME')
    return output_df


def calculate_dx_from_3d(meas_input, input_files, model, header_dict, tune_dict):
    """It computes  dispersion from 3 D kicks"""
    output, accelerator = meas_input.outputdir, meas_input.accelerator
    dispersion_dfs = {}
    global_factors = None
    for plane in PLANES:
        df_orbit = pd.DataFrame(model).loc[:, ['S', 'MU' + plane, 'DP' + plane, 'D' + plane, plane]]
        df_orbit.rename(columns={'MU' + plane: 'MU' + plane + 'MDL', 'DP' + plane: 'DP' + plane + 'MDL',
                                 'D' + plane: 'D' + plane + 'MDL', plane: plane + 'MDL'}, inplace=True)
        df_orbit = pd.merge(df_orbit, input_files.joined_frame(plane, ['AMPZ', 'MUZ', 'AMP' + plane]),
                            how='inner', left_index=True, right_index=True)
        df_orbit['COUNT'] = len(input_files.get_columns(df_orbit, 'AMPZ'))
        unscaled_amps = (df_orbit.loc[:, input_files.get_columns(df_orbit, 'AMPZ')].values *
                         df_orbit.loc[:, input_files.get_columns(df_orbit, 'AMP' + plane)].values)

        mask = accelerator.get_element_types_mask(df_orbit.index, ["arc_bpm"])
        if plane == "X":
            # global factors are 1 / dppamp ... or up to factor 2 if complex spectra
            global_factors = np.sum(df_orbit.loc[mask, 'D'+plane+'MDL'].values) / np.sum(unscaled_amps[mask, :], axis=0)
            print(1 / global_factors)
        # scaling to the model, and getting the synchrotron phase in the arcs
        dispersion_amps = (unscaled_amps * global_factors)
        same_interval_phase = np.angle(np.exp(PI2I * df_orbit.loc[:, input_files.get_columns(df_orbit, 'MUZ')].values)) / (2 * np.pi)
        phase_wrt_arcs = same_interval_phase - stats.circular_mean(same_interval_phase[mask, :], period=1, axis=0)
        phase_wrt_arcs = np.abs(np.where(np.abs(phase_wrt_arcs) > 0.5, phase_wrt_arcs - np.sign(phase_wrt_arcs), phase_wrt_arcs))
        if len(input_files.get_columns(df_orbit, 'AMPZ')) > 1:
            # resolving the sign of dispersion
            dispersions = dispersion_amps * np.sign(
                0.25 - np.abs(stats.circular_mean(phase_wrt_arcs, period=1, axis=1)))[:, None]
            # final calculation
            df_orbit['STDD' + plane] = np.std(dispersions, axis=1) * stats.t_value_correction(dispersions.shape[1])
            df_orbit['D' + plane] = np.mean(dispersions, axis=1)
        else:
            df_orbit['STDD' + plane] = 0.0
            df_orbit['D' + plane] = dispersion_amps * np.sign(0.25 - np.abs(phase_wrt_arcs))
        df_orbit['DP' + plane] = _calculate_dp(model, df_orbit.loc[:, ['D' + plane, 'STDD' + plane]], plane)
        df_orbit['DELTAD' + plane] = df_orbit.loc[:, 'D' + plane] - df_orbit.loc[:, 'D' + plane + 'MDL']
        output_df = df_orbit.loc[:, ['S', 'COUNT', 'D' + plane, 'STDD' + plane, 'DP' + plane,
                                     'D' + plane + 'MDL', 'DP' + plane + 'MDL', 'MU' + plane + 'MDL',
                                     'DELTAD' + plane]]
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
