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
from optics_measurements.compensate_excitation import get_lambda


def calculate_beta_from_amplitude(measure_input, input_files, tune_dict, phase_dict, beta_phase, header_dict):
    """
    Calculates beta and fills the following TfsFiles:
        getampbetax.out        getampbetax_free.out        getampbetax_free2.out
        getampbetay.out        getampbetay_free.out        getampbetay_free2.out

    Parameters:
        'measure_input': OpticsInput object
        'input_files': InputFiles object contains measurement files
        'tune_d': TuneDict contains measured tunes
        'phase_d': PhaseDict contains measured phase advances
        'beta_phase': contains beta functions from measured from phase
        'header_dict': dictionary of header items common for all output files

    Returns:
    """
    mad_twiss = measure_input.accelerator.get_model_tfs()
    if measure_input.accelerator.excitation:
        mad_ac = measure_input.accelerator.get_driven_tfs()
    else:
        mad_ac = mad_twiss
    ratios = {}
    for plane in ['X', 'Y']:
        beta_amp = beta_from_amplitude(measure_input, input_files, mad_ac, plane)
        ratio = pd.merge(beta_phase[plane].loc[:, ['BET' + plane]], beta_amp.loc[:, ['BET' + plane]],
                         how='inner', left_index=True, right_index=True, suffixes=('phase', 'amp'))
        ratio['Ratio'] = ratio.loc[:, 'BET' + plane + 'phase'].values / ratio.loc[:, 'BET' + plane + 'amp'].values
        mask = (np.array(0.1 < np.abs(ratio.loc[:, 'Ratio'].values)) &
                np.array(np.abs(ratio.loc[:, 'Ratio'].values) < 10.0) &
                np.array(measure_input.accelerator.get_element_types_mask(ratio.index, ["arc_bpm"])))
        x_ratio = np.mean(ratio.loc[mask, 'Ratio'].values)
        beta_amp['BET' + plane + 'RES'] = beta_amp.loc[:, 'BET' + plane] * x_ratio
        beta_amp['BET' + plane + 'STDRES'] = beta_amp.loc[:, 'BET' + plane + 'STD'] * x_ratio
        header_d = _get_header(header_dict, tune_dict, np.std(
            beta_amp.loc[:, 'DELTABET' + plane].values), x_ratio, 'getampbeta' + plane.lower() + '.out', free=False)
        tfs.write(join(measure_input.outputdir, header_d['FILENAME']), beta_amp, header_d, save_index='NAME')
        ratios[plane] = x_ratio
        # -- ac to free amp beta
        if measure_input.accelerator.excitation:
            beta_amp_f = beta_from_amplitude(measure_input, input_files, mad_twiss, plane,
                                             (tune_dict[plane]["Q"], tune_dict[plane]["QF"], phase_dict[plane]["ac2bpm"]))
            x_ratio_f = x_ratio
            header_f = _get_header(header_dict, tune_dict, np.std(
                    beta_amp_f.loc[:, 'DELTABET' + plane].values), x_ratio_f,
                                   'getampbeta' + plane.lower() + '_free.out', free=True)

            beta_amp_f['BET' + plane + 'RES'] = beta_amp_f.loc[:, 'BET' + plane] * x_ratio_f
            beta_amp_f['BET' + plane + 'STDRES'] = beta_amp_f.loc[:, 'BET' + plane + 'STD'] * x_ratio_f
            tfs.write(join(measure_input.outputdir, header_f['FILENAME']), beta_amp_f, header_f, save_index='NAME')
            # FREE2 calculation
            beta_amp_f2 = pd.DataFrame(beta_amp)
            beta_amp_f2['BET' + plane] = _get_free_amp_beta(beta_amp_f2.loc[:, ['BET' + plane]],
                                                            mad_ac, mad_twiss, plane)
            beta_amp_f2['BET' + plane + 'RES'] = _get_free_amp_beta(
                    beta_amp_f2.loc[:, ['BET' + plane + 'RES']].rename(
                        columns={'BET' + plane + 'RES': 'BET' + plane}), mad_ac, mad_twiss, plane)
            header_f2 = _get_header(header_dict, tune_dict, np.std(
                    beta_amp_f2.loc[:, 'DELTABET' + plane].values), x_ratio,
                                    'getampbeta' + plane.lower() + '_free2.out', free=True)
            tfs.write(join(measure_input.outputdir, header_f2['FILENAME']), beta_amp_f2, header_f2)
    return ratios


def _get_free_amp_beta(df_meas,  mad_ac, mad_twiss, plane):
    df = pd.merge(pd.DataFrame(df_meas), mad_ac.loc[:, ['BET' + plane]], how='inner',
                  left_index=True, right_index=True, suffixes=('', 'ac'))
    df = pd.merge(df, mad_twiss.loc[:, ['BET' + plane]], how='inner', left_index=True,
                  right_index=True, suffixes=('', 'f'))
    return df.loc[:, "BET" + plane] * df.loc[:, "BET" + plane + 'f'] / df.loc[:, "BET" + plane + 'ac']


def beta_from_amplitude(meas_input, input_files, model, plane, compensate=None):
    df_amp_beta = pd.DataFrame(model).loc[:, ['S', 'MU' + plane, 'BET' + plane]]
    df_amp_beta.rename(columns={'MU' + plane: 'MU' + plane + 'MDL',
                                'BET' + plane: 'BET' + plane + 'MDL'}, inplace=True)
    df_amp_beta = pd.merge(df_amp_beta, input_files.joined_frame(plane, ['AMP' + plane, 'MU' + plane]),
                           how='inner', left_index=True, right_index=True)
    df_amp_beta['COUNT'] = len(input_files.get_columns(df_amp_beta, 'AMP' + plane))
    df_amp_beta['AMP' + plane] = np.mean(input_files.get_data(df_amp_beta, 'AMP' + plane), axis=1)

    if compensate is not None:
        phases_meas = input_files.get_data(df_amp_beta, 'MU' + plane) * meas_input.accelerator.get_beam_direction()
        driven_tune, free_tune, ac2bpmac = compensate
        k_bpmac = ac2bpmac[2]
        phase_corr = ac2bpmac[1] - phases_meas[k_bpmac] + (0.5 * driven_tune)
        phases_meas = phases_meas + phase_corr[np.newaxis, :]
        r = get_lambda(driven_tune % 1.0, free_tune % 1.0)
        phases_meas[k_bpmac:, :] = phases_meas[k_bpmac:, :] - driven_tune
        for_sqrt2j = input_files.get_data(df_amp_beta, 'AMP' + plane) / np.sqrt(
            df_amp_beta.loc[:, 'BET' + plane + 'MDL'].values[:, np.newaxis])
        sqrt2j = np.mean(for_sqrt2j[meas_input.accelerator.get_element_types_mask(df_amp_beta.index, ["arc_bpm"])], axis=0)
        betall = (np.square(
            (input_files.get_data(df_amp_beta, 'AMP' + plane).T / sqrt2j[:, np.newaxis]).T) *
                  (1 + r ** 2 + 2 * r * np.cos(4 * np.pi * phases_meas)) / (1 - r ** 2))
        df_amp_beta['BET' + plane] = np.mean(betall, axis=1)
        df_amp_beta['BET' + plane + 'STD'] = np.std(betall, axis=1)
    else:
        # amplitudes are first averaged over files then squared and averaged over BPMs
        kick = np.mean(np.square(df_amp_beta.loc[:, 'AMP' + plane].values) /
                       df_amp_beta.loc[:, 'BET' + plane + 'MDL'].values)
        # amplitudes are first squared then averaged
        kick2 = np.mean(np.square(input_files.get_data(df_amp_beta, 'AMP' + plane)) /
                        df_amp_beta.loc[:, 'BET' + plane + 'MDL'].values[:, np.newaxis], axis=0)
        df_amp_beta['BET' + plane] = np.square(df_amp_beta.loc[:, 'AMP' + plane].values) / kick
        df_amp_beta['BET' + plane + 'STD'] = np.std((np.square(
            input_files.get_data(df_amp_beta, 'AMP' + plane)).T / kick2[:, np.newaxis]).T, axis=1)
    df_amp_beta['DELTABET' + plane] = ((df_amp_beta.loc[:, 'BET' + plane] -
                                       df_amp_beta.loc[:, 'BET' + plane + 'MDL']) /
                                       df_amp_beta.loc[:, 'BET' + plane + 'MDL'])
    return df_amp_beta.loc[:, ['S', 'COUNT', 'BET' + plane, 'BET' + plane + 'STD',
                               'BET' + plane + 'MDL', 'MU' + plane + 'MDL', 'DELTABET' + plane]]


def _get_header(header_dict, tune_d, rmsbbeat, scaling_factor, file_name, free=False):
    header = header_dict.copy()
    if free:
        header['Q1'] = tune_d["X"]["QF"]
        header['Q2'] = tune_d["Y"]["QF"]
    else:
        header['Q1'] = tune_d["X"]["Q"]
        header['Q2'] = tune_d["Y"]["Q"]
    header['RMSbetabeat'] = rmsbbeat
    header['RescalingFactor'] = scaling_factor
    header['FILENAME'] = file_name
    return header
