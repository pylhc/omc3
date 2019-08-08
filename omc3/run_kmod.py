from os.path import join
from utils import logging_tools, iotools
import numpy as np
import datetime
import tfs
from kmod import analysis, helper
from kmod.constants import EXT, FIT_PLOTS_NAME, SEQUENCES_PATH
from generic_parser.entrypoint import entrypoint, EntryPointParameters
from definitions import formats

LOG = logging_tools.get_logger(__name__)


def kmod_params():
    parser = EntryPointParameters()
    parser.add_parameter(flags='--betastar_and_waist', type=float, name='betastar_and_waist', required=True,
                         nargs='+', help='Estimated beta star of measurements and waist shift',)
    parser.add_parameter(flags='--working_directory', type=str, name='working_directory', required=True,
                         help='path to working directory with stored KMOD measurement files',)
    parser.add_parameter(flags='--beam', type=str, name='beam', choices=['B1', 'B2'],
                         required=True, help='define beam used: B1 or B2',)
    parser.add_parameter(flags='--cminus', type=float, name='cminus', help='C Minus',)
    parser.add_parameter(flags='--misalignment', type=float, name='misalignment',
                         help='misalignment of the modulated quadrupoles in m',)
    parser.add_parameter(flags='--errorK', type=float, name='errorK',
                         help='error in K of the modulated quadrupoles, relative to gradient',)
    parser.add_parameter(flags='--errorL', type=float, name='errorL',
                         help='error in length of the modulated quadrupoles, unit m',)
    parser.add_parameter(flags='--tune_uncertainty', type=float, name='tune_uncertainty', default=2.5e-5,
                         help='tune measurement uncertainty')
    parser.add_parameter(flags='--instruments', type=str, name='instruments', default='MONITOR,SBEND,TKICKER,INSTRUMENT',
                         help='define instruments (use keywords from twiss) at which beta should '
                              'be calculated , separated by comma, e.g. MONITOR,RBEND,INSTRUMENT,TKICKER',)
    parser.add_parameter(flags='--simulation', action='store_true', name='simulation',
                         help='flag for enabling simulation mode',)
    parser.add_parameter(flags='--log', action='store_true', name='log',
                         help='flag for creating a log file')
    parser.add_parameter(flags='--no_autoclean', action='store_true', name='no_autoclean',
                         help='flag for manually cleaning data')
    parser.add_parameter(flags='--no_sig_digits', action='store_true', name='no_sig_digits',
                         help='flag to not use significant digits')
    parser.add_parameter(flags='--no_plots', action='store_true', name='no_plots',
                         help='flag to not create any plots')
    parser.add_parameter(flags='--circuit', type=str, name='circuits', nargs=2,
                         help='circuit names of the modulated quadrupoles')
    parser.add_parameter(flags='--interaction_point', type=str, name='ip', choices=['ip1', 'ip2', 'ip5', 'ip8', 'IP1', 'IP2', 'IP5', 'IP8'],
                         help='define interaction point')
    return parser


@entrypoint(kmod_params(), strict=True)
def analyse_kmod(opt):
    """
    Run Kmod analysis
    """
    LOG.info('Getting input parameter')
    if opt.ip is None and opt.circuits is None:
        raise AttributeError('No IP or circuits specified, stopping analysis')
    if opt.ip is not None and opt.circuits is not None:
        raise AttributeError('Both IP and circuits specified, choose only one, stopping analysis')
    if not 1 < len(opt.betastar_and_waist) < 5:
        raise AttributeError("Option betastar_and_waist has to consist of 2 to 4 floats")
    opt.betastar_and_waist = convert_betastar_and_waist(opt.betastar_and_waist)
    for error in ("cminus", "errorK", "errorL", "misalignment"):
        opt = check_default_error(opt, error)

    LOG.info(f"{'IP trim' if opt.ip is not None else 'Individual magnets'} analysis")
    opt['magnets'] = MAGNETS_IP[opt.ip.upper()] if opt.ip is not None else [
        find_magnet(opt.beam, circuit) for circuit in opt.circuits]
    opt['label'] = f'{opt.ip}{opt.beam}' if opt.ip is not None else f'{opt.magnets[0]}-{opt.magnets[1]}'
    opt['instruments'] = list(map(str.upper, opt.instruments.split(",")))

    output_dir = join(opt.working_directory, opt.label)
    iotools.create_dirs(output_dir)

    LOG.info('Get inputfiles')
    magnet1_df, magnet2_df = helper.get_input_data(opt)
    opt, magnet1_df, magnet2_df, betastar_required = define_params(opt, magnet1_df, magnet2_df)

    LOG.info('Run simplex')
    magnet1_df, magnet2_df, results_df = analysis.analyse(magnet1_df, magnet2_df, opt)

    LOG.info('Plot tunes and fit')
    if opt.no_plots:
        helper.plot_cleaned_data([magnet1_df, magnet2_df], join(output_dir, FIT_PLOTS_NAME), interactive_plot=False)

    LOG.info('Calculate betastar')
    if betastar_required:
        results_df = analysis.calc_betastar(opt, results_df, magnet1_df)

    results_df.loc[:, 'TIME'] = ('{0:formats.TIME}'.format(datetime.datetime.now()))

    LOG.info('Calculate beta at instruments')
    if opt.instruments_found:
        instrument_beta_df = analysis.calc_beta_at_instruments(opt, results_df, magnet1_df, magnet2_df)
        tfs.write(join(output_dir, 'beta_instrument.tfs'), instrument_beta_df)

    LOG.info('Write magnet dataframes and results')
    for magnet_df in [magnet1_df, magnet2_df]:
        tfs.write(join(output_dir, f"{magnet_df.headers['QUADRUPOLE']}{EXT}"), magnet_df)
    tfs.write(join(output_dir, 'results.tfs'), results_df)


def convert_betastar_and_waist(bs):
    if len(bs) == 2:
        return dict(X=np.array([bs[0], bs[1]]), Y=np.array([bs[0], bs[1]]))
    if len(bs) == 3:
        return dict(X=np.array([bs[0], bs[2]]), Y=np.array([bs[1], bs[2]]))
    return dict(X=np.array([bs[0], bs[2]]), Y=np.array([bs[1], bs[3]]))


def check_default_error(options, error):
    if options[error] is None:
        options[error] = DEFAULTS_IP[error] if options.ip is not None else DEFAULTS_CIRCUITS[error]
    return options


def find_magnet(beam, circuit):
    sequence = tfs.read(join(SEQUENCES_PATH, f"twiss_lhc{beam.lower()}.dat"))
    circuit = circuit.split('.')
    magnetname = sequence[sequence['NAME'].str.contains(r'MQ\w+\.{:s}{:s}{:s}\.\w+'.format(circuit[0][-1], circuit[1][0], circuit[1][1]))]['NAME'].values[0]
    return magnetname


def define_params(options, magnet1_df, magnet2_df):
    LOG.debug(' adding additional parameters to header ')
    beta_star_required = False
    sequence = tfs.read(join(SEQUENCES_PATH, f"twiss_lhc{options.beam.lower()}.dat"), index='NAME')

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
            options[instrument] = dict(zip(between_magnets_df.loc[between_magnets_df['KEYWORD'] == instrument]['NAME'].values,
                                           (between_magnets_df.loc[between_magnets_df['KEYWORD'] == instrument]['S'].values - ip_position)))
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
    with logging_tools.DebugMode(active=True, log_file=""):
        analyse_kmod()
