import os
from utils import logging_tools, iotools
import datetime
import tfs
from kmod import kmod_input, kmod_get_files, kmod_utils, kmod_cleaning, kmod_analysis, kmod_constants
from generic_parser.entrypoint import entrypoint, EntryPointParameters

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
LOG = logging_tools.get_logger(__name__)


def kmod_params():

    parser = EntryPointParameters()
    parser.add_parameter(flags='--betastar_and_waist',
                         help='Estimated beta star of measurements and waist shift',
                         type=float,
                         name='betastar',
                         required=True,
                         nargs=2)
    parser.add_parameter(flags='--working_directory',
                         help='path to working directory with stored KMOD measurement files',
                         type=str,
                         name='work_dir',
                         required=True)
    parser.add_parameter(flags='--beam',
                         help='define beam used: b1 or b2',
                         type=str,
                         name='beam',
                         choices=['b1', 'b2', 'B1', 'B2'],
                         required=True)

    parser.add_parameter(flags='--cminus',
                         help='C Minus',
                         type=float,
                         name='cminus')
    parser.add_parameter(flags='--misalignment',
                         help='misalignment of the modulated quadrupoles in m',
                         type=float,
                         name='misalignment')
    parser.add_parameter(flags='--errorK',
                         help='error in K of the modulated quadrupoles, relative to gradient',
                         type=float,
                         name='errorK')
    parser.add_parameter(flags='--errorL',
                         help='error in length of the modulated quadrupoles, unit m',
                         type=float,
                         name='errorL')

    parser.add_parameter(flags='--tune_uncertainty',
                         help='tune measurement uncertainty',
                         type=float,
                         name='tunemeasuncertainty',
                         default=2.5e-5)
    parser.add_parameter(flags='--instruments',
                         help='define instruments (use keywords from twiss) at which beta should be calculated , separated by comma, e.g. MONITOR,RBEND,INSTRUMENT,TKICKER',
                         type=str,
                         name='instruments',
                         default='MONITOR,SBEND,TKICKER,INSTRUMENT')

    parser.add_parameter(flags='--simulation',
                         help='flag for enabling simulation mode',
                         action='store_true',
                         name='simulation')    
    parser.add_parameter(flags='--log',
                         help='flag for creating a log file',
                         action='store_true',
                         name='log')
    parser.add_parameter(flags='--no_autoclean',
                         help='flag for manually cleaning data',
                         action='store_true',
                         name='a_clean')
    parser.add_parameter(flags='--circuit',
                         help='circuit names of the modulated quadrupoles',
                         type=str,
                         name='circuits',
                         nargs=2)
    parser.add_parameter(flags='--interaction_point',
                         help='define interaction point',
                         type=str,
                         name='ip',
                         choices=['ip1', 'ip2', 'ip5', 'ip8', 'IP1', 'IP2', 'IP5', 'IP8'])

    return parser


@entrypoint(kmod_params(), strict=True)
def analyse_kmod(opt):
    """
    Run Kmod analysis
    """
    LOG.info('Getting input parameter')
    kmod_input_params = kmod_input.get_input(opt)
    iotools.create_dirs(kmod_constants.get_working_directory(kmod_input_params))

    LOG.info('Get inputfiles')
    if kmod_input_params.simulation:
        magnet1_df, magnet2_df = kmod_get_files.get_simulation_files(kmod_input_params)
    else:
        magnet1_df, magnet2_df = kmod_get_files.merge_data(kmod_input_params)

    magnet1_df, magnet2_df = kmod_utils.define_params(kmod_input_params, magnet1_df, magnet2_df)

    magnet1_df = kmod_utils.add_tuneuncertainty(magnet1_df, kmod_input_params)
    magnet2_df = kmod_utils.add_tuneuncertainty(magnet2_df, kmod_input_params)

    LOG.info('Clean data')
    magnet1_df = kmod_cleaning.clean_data(kmod_input_params, magnet1_df)
    magnet2_df = kmod_cleaning.clean_data(kmod_input_params, magnet2_df)

    LOG.info('Run simplex')
    magnet1_df, magnet2_df, results_df = kmod_analysis.analyse(magnet1_df,
                                                               magnet2_df,
                                                               kmod_input_params)

    LOG.info('Plot tunes and fit')
    kmod_utils.plot_cleaned_data(magnet1_df, magnet2_df, kmod_input_params, interactive_plot=False)

    LOG.info('Calculate betastar')
    if kmod_input_params.betastar_required:
        results_df = kmod_analysis.calc_betastar(kmod_input_params,
                                                 results_df,
                                                 magnet1_df,
                                                 magnet2_df)

    results_df.loc[:, 'TIME'] = ('{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    LOG.info('Calculate beta at instruments')
    if kmod_input_params.instruments_found != []:
        instrument_beta_df = kmod_analysis.calc_beta_at_instruments(kmod_input_params,
                                                                    results_df,
                                                                    magnet1_df,
                                                                    magnet2_df)

    LOG.info('Write magnet dataframes and results')

    for magnet_df in [magnet1_df, magnet2_df]:
        tfs.write_tfs(
            os.path.join(kmod_constants.get_working_directory(kmod_input_params),
                         '{:s}.tfs'.format(magnet_df.headers['QUADRUPOLE'])),
            magnet_df)

    tfs.write_tfs(os.path.join(kmod_constants.get_working_directory(
        kmod_input_params), 'results.tfs'), results_df)
    if kmod_input_params.instruments_found != []:
        tfs.write_tfs(os.path.join(kmod_constants.get_working_directory(
            kmod_input_params), 'beta_instrument.tfs'), instrument_beta_df)


if __name__ == '__main__':
    with logging_tools.DebugMode(active=True, log_file=""):
        analyse_kmod()
