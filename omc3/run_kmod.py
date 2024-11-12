"""
Run Kmod
--------

Top-level script to analyse Kmod-results from the ``LHC`` and creating files for GUI and plotting
as well as returning beta-star and waist shift.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tfs
from generic_parser import EntryPointParameters, entrypoint

from omc3.kmod import analysis, helper
from omc3.kmod.constants import (BETA, ERR, EXT, FIT_PLOTS_NAME, INSTRUMENTS_FILE_NAME,
                                 LSA_FILE_NAME, RESULTS_FILE_NAME, SEQUENCES_PATH, STAR)
from omc3.utils import iotools, logging_tools

LOG = logging_tools.get_logger(__name__)

LSA_COLUMNS = ['NAME', f'{BETA}X', f'{ERR}{BETA}X', f'{BETA}Y', f'{ERR}{BETA}Y']


def kmod_params():
    parser = EntryPointParameters()
    parser.add_parameter(name='betastar_and_waist',
                         type=float,
                         required=True, nargs='+',
                         help='Estimated beta star of measurements and waist shift',)
    parser.add_parameter(name='working_directory',
                         type=Path,
                         required=True,
                         help='path to working directory with stored KMOD measurement files',)
    parser.add_parameter(name='beam',
                         type=int,
                         choices=[1, 2], required=True,
                         help='define beam used: 1 or 2',)
    parser.add_parameter(name='cminus', 
                         type=float,                         
                         help='C Minus',)
    parser.add_parameter(name='misalignment',
                         type=float,
                         help='misalignment of the modulated quadrupoles in m',)
    parser.add_parameter(name='errorK',
                         type=float,
                         help='error in K of the modulated quadrupoles, relative to gradient',)
    parser.add_parameter(name='errorL',
                         type=float,                         
                         help='error in length of the modulated quadrupoles, unit m',)
    parser.add_parameter(name='tune_uncertainty',
                         type=float,
                         default=2.5e-5,
                         help='tune measurement uncertainty')
    parser.add_parameter(name='instruments',
                         type=str,
                         default='MONITOR,SBEND,TKICKER,INSTRUMENT',
                         help='define instruments (use keywords from twiss) at which beta should '
                              'be calculated , separated by comma, e.g. MONITOR,RBEND,INSTRUMENT,TKICKER',)
    parser.add_parameter(name='simulation',
                         action='store_true',                         
                         help='flag for enabling simulation mode',)
    parser.add_parameter(name='log',
                         action='store_true',                         
                         help='flag for creating a log file')
    parser.add_parameter(name='no_autoclean',
                         action='store_true',
                         help='flag for manually cleaning data')
    parser.add_parameter(name='no_sig_digits',
                         action='store_true',                         
                         help='flag to not use significant digits')
    parser.add_parameter(name='no_plots',
                         action='store_true',                         
                         help='flag to not create any plots')
    parser.add_parameter(name='circuits',
                         type=str,
                         nargs=2,
                         help='circuit names of the modulated quadrupoles')
    parser.add_parameter(name='interaction_point',
                         type=str,
                         choices=['ip1', 'ip2', 'ip5', 'ip8', 'IP1', 'IP2', 'IP5', 'IP8'],
                         help='define interaction point')
    parser.add_parameter(name='measurement_dir',
                         type=Path,
                         help='give an optics measurement directory to include phase constraint in penalty function')
    parser.add_parameter(name='phase_weight',
                         type=float,
                         default=0.0,
                         help='weight in penalty function between phase and beta.'
                              'If weight=0 phase is not used as a constraint.')
    parser.add_parameter(name='model_dir',
                         type=Path,
                         help='twiss model that contains phase')
    parser.add_parameter(name="outputdir",
                         type=Path,
                         help="Path where outputfiles will be stored, defaults "
                                                "to the given working_directory")

    return parser


@entrypoint(kmod_params(), strict=True)
def analyse_kmod(opt):
    """
    Run Kmod analysis.

    Kmod Keyword Arguments:
        *--Required--*

        - **beam** *(int)*:

            define beam used: 1 or 2

            choices: ``[1, 2]``


        - **betastar_and_waist** *(float)*:

            Estimated beta star of measurements and waist shift


        - **working_directory** *(Path)*:

            path to working directory with stored KMOD measurement files


        *--Optional--*

        - **circuits** *(str)*:

            circuit names of the modulated quadrupoles


        - **cminus** *(float)*:

            C Minus


        - **errorK** *(float)*:

            error in K of the modulated quadrupoles, relative to gradient


        - **errorL** *(float)*:

            error in length of the modulated quadrupoles, unit m


        - **instruments** *(str)*:

            define instruments (use keywords from twiss) at which beta should be
            calculated , separated by comma, e.g. MONITOR,RBEND,INSTRUMENT,TKICKER

            default: ``MONITOR,SBEND,TKICKER,INSTRUMENT``


        - **interaction_point** *(str)*:

            define interaction point

            choices: ``['ip1', 'ip2', 'ip5', 'ip8', 'IP1', 'IP2', 'IP5', 'IP8']``


        - **log**:

            flag for creating a log file

            action: ``store_true``


        - **measurement_dir** *(Path)*:

            give an optics measurement directory to include phase constraint in
            penalty function


        - **misalignment** *(float)*:

            misalignment of the modulated quadrupoles in m


        - **model_dir** *(Path)*:

            twiss model that contains phase


        - **no_autoclean**:

            flag for manually cleaning data

            action: ``store_true``


        - **no_plots**:

            flag to not create any plots

            action: ``store_true``


        - **no_sig_digits**:

            flag to not use significant digits

            action: ``store_true``


        - **outputdir** *(Path)*:

            Path where outputfiles will be stored, defaults to the given
            working_directory


        - **phase_weight** *(float)*:

            weight in penalty function between phase and beta.If weight=0 phase is
            not used as a constraint.

            default: ``0.0``


        - **simulation**:

            flag for enabling simulation mode

            action: ``store_true``


        - **tune_uncertainty** *(float)*:

            tune measurement uncertainty

            default: ``2.5e-05``
    """
    LOG.info('Getting input parameter')
    if opt.interaction_point is None and opt.circuits is None:
        raise AttributeError('No IP or circuits specified, stopping analysis')
    if opt.interaction_point is not None and opt.circuits is not None:
        raise AttributeError('Both IP and circuits specified, choose only one, stopping analysis')
    if not 1 < len(opt.betastar_and_waist) < 5:
        raise AttributeError("Option betastar_and_waist has to consist of 2 to 4 floats")
    opt.betastar_and_waist = convert_betastar_and_waist(opt.betastar_and_waist)
    for error in ("cminus", "errorK", "errorL", "misalignment"):
        opt = check_default_error(opt, error)
    if opt.measurement_dir is None and opt.model_dir is None and opt.phase_weight:
        raise AttributeError("Cannot use phase advance without measurement or model")
    if opt.outputdir is None:
        opt.outputdir = opt.working_directory

    LOG.info(f"{'IP trim' if opt.interaction_point is not None else 'Individual magnets'} analysis")
    opt['magnets'] = MAGNETS_IP[opt.interaction_point.upper()] if opt.interaction_point is not None else [
        find_magnet(opt.beam, circuit) for circuit in opt.circuits]
    opt['label'] = f'{opt.interaction_point}B{opt.beam:d}' if opt.interaction_point is not None else f'{opt.magnets[0]}-{opt.magnets[1]}'
    opt['instruments'] = list(map(str.upper, opt.instruments.split(",")))

    output_dir = opt.outputdir / opt.label
    iotools.create_dirs(output_dir)

    LOG.info('Get inputfiles')
    magnet1_df, magnet2_df = helper.get_input_data(opt)
    opt, magnet1_df, magnet2_df, betastar_required = define_params(opt, magnet1_df, magnet2_df)

    LOG.info('Run simplex')
    magnet1_df, magnet2_df, results_df, instrument_beta_df = analysis.analyse(magnet1_df, magnet2_df, opt, betastar_required)

    LOG.info('Plot tunes and fit')
    if opt.no_plots:
        helper.plot_cleaned_data([magnet1_df, magnet2_df], output_dir / FIT_PLOTS_NAME, interactive_plot=False)

    LOG.info('Write magnet dataframes and results')
    for magnet_df in [magnet1_df, magnet2_df]:
        tfs.write(output_dir / f"{magnet_df.headers['QUADRUPOLE']}{EXT}", magnet_df)

    tfs.write(output_dir / f'{RESULTS_FILE_NAME}{EXT}', results_df)

    if opt.instruments_found:
        tfs.write(output_dir / f'{INSTRUMENTS_FILE_NAME}{EXT}', instrument_beta_df)

    create_lsa_results_file(betastar_required, opt.instruments_found, results_df, instrument_beta_df, output_dir)


def create_lsa_results_file(betastar_required, instruments_found, results_df, instrument_beta_df, output_dir):
    lsa_results_df = pd.DataFrame(columns=LSA_COLUMNS)
    if betastar_required:
        exporting_columns=['LABEL', f'{BETA}{STAR}X', f'{ERR}{BETA}{STAR}X', f'{BETA}{STAR}Y', f'{ERR}{BETA}{STAR}Y']
        lsa_results_df=results_df[exporting_columns].rename(columns=dict(zip(exporting_columns, LSA_COLUMNS)))
    if instruments_found:
        # We first make sure we don't try a concat operation if a df is empty
        # (otherwise pandas complains with a FutureWarning since 2.1.1)
        dfs_to_concat = [lsa_results_df, instrument_beta_df]
        dfs_to_concat = [df for df in dfs_to_concat if not df.empty]

        # We will raise for the user if there is no data in the DFs
        if not len(dfs_to_concat):
            msg = "All dfs are empty! Check your Kmod inputs."
            raise ValueError(msg)
        lsa_results_df = pd.concat(dfs_to_concat, axis="index", sort=False, ignore_index=True)

    if not lsa_results_df.empty:
        tfs.write(output_dir / f'{LSA_FILE_NAME}{EXT}', lsa_results_df)


def convert_betastar_and_waist(bs):
    if len(bs) == 2:
        return dict(X=np.array([bs[0], bs[1]]), Y=np.array([bs[0], bs[1]]))
    if len(bs) == 3:
        return dict(X=np.array([bs[0], bs[2]]), Y=np.array([bs[1], bs[2]]))
    return dict(X=np.array([bs[0], bs[2]]), Y=np.array([bs[1], bs[3]]))


def check_default_error(options, error):
    if options[error] is None:
        options[error] = DEFAULTS_IP[error] if options.interaction_point is not None else DEFAULTS_CIRCUITS[error]
    return options


def find_magnet(beam, circuit):
    sequence = tfs.read(SEQUENCES_PATH / f"twiss_lhcb{beam:d}.dat")
    circuit = circuit.split('.')
    magnetname = sequence[sequence['NAME'].str.contains(r'MQ\w+\.{:s}{:s}{:s}\.\w+'.format(circuit[0][-1], circuit[1][0], circuit[1][1]))]['NAME'].to_numpy()[0]
    return magnetname


def define_params(options, magnet1_df, magnet2_df):
    LOG.debug(' adding additional parameters to header ')
    beta_star_required = False
    sequence = tfs.read(SEQUENCES_PATH / f"twiss_lhcb{options.beam:d}.dat", index='NAME')

    for magnet_df in [magnet1_df, magnet2_df]:
        magnet_df.headers['LENGTH'] = sequence.loc[magnet_df.headers['QUADRUPOLE'], 'L']
        magnet_df.headers['POLARITY'] = np.sign(sequence.loc[magnet_df.headers['QUADRUPOLE'], 'K1L'])

    magnet1_position_center = sequence.loc[magnet1_df.headers['QUADRUPOLE'], 'S'] - magnet1_df.headers['LENGTH']/2.
    magnet2_position_center = sequence.loc[magnet2_df.headers['QUADRUPOLE'], 'S'] - magnet2_df.headers['LENGTH']/2.

    ip_position = (magnet1_position_center + magnet2_position_center)/2.

    for magnet_df, magnet_position_center in zip([magnet1_df, magnet2_df], [magnet1_position_center, magnet2_position_center]):
        magnet_df.headers['LSTAR'] = np.abs(ip_position - magnet_position_center) - magnet_df.headers['LENGTH']/2.

    if magnet1_position_center < magnet2_position_center:
        left_magnet_df = magnet1_df
        right_magnet_df = magnet2_df
    elif magnet2_position_center < magnet1_position_center:
        left_magnet_df = magnet2_df
        right_magnet_df = magnet1_df

    between_magnets_df = sequence.reset_index().truncate(before=sequence.index.get_loc(left_magnet_df.headers['QUADRUPOLE']), after=sequence.index.get_loc(right_magnet_df.headers['QUADRUPOLE']), axis='index')

    if between_magnets_df.isin(['OMK']).any().loc['PARENT']:
        beta_star_required = True
    instruments = []
    for instrument in options.instruments:
        if between_magnets_df.isin([instrument]).any().loc['KEYWORD']:
            instruments.append(instrument)
            options[instrument] = dict(zip(between_magnets_df.loc[between_magnets_df['KEYWORD'] == instrument]['NAME'].to_numpy(),
                                           (between_magnets_df.loc[between_magnets_df['KEYWORD'] == instrument]['S'].to_numpy() - ip_position)))
    options.instruments_found = instruments
    return options, magnet1_df, magnet2_df, beta_star_required


DEFAULTS_IP = {
    "cminus": 1E-3,
    "misalignment": 0.006,
    "errorK": 0.001,
    "errorL": 0.001,
}

DEFAULTS_CIRCUITS = {
    "cminus": 1E-3,
    "misalignment": 0.001,
    "errorK": 0.001,
    "errorL": 0.001,
}

MAGNETS_IP = {
    "IP1": ['MQXA.1L1', 'MQXA.1R1'],
    "IP2": ['MQXA.1L2', 'MQXA.1R2'],
    "IP5": ['MQXA.1L5', 'MQXA.1R5'],
    "IP8": ['MQXA.1L8', 'MQXA.1R8']
}


if __name__ == '__main__':
    analyse_kmod()
