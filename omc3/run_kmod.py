import os
from utils import logging_tools, iotools
import datetime
import tfs
from kmod import kmod_input, kmod_get_files, kmod_utils, kmod_cleaning, kmod_analysis, kmod_constants


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
LOG = logging_tools.get_logger(__name__)


def analyse_kmod():
    """
    Run Kmod analysis 
    """
    LOG.info('Getting input parameter')
    kmod_input_params = kmod_input.get_input()
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
    magnet1_df, magnet2_df, results_df = kmod_analysis.analyse(magnet1_df, magnet2_df, kmod_input_params)

    LOG.info('Plot tunes and fit')
    kmod_utils.plot_cleaned_data(magnet1_df, magnet2_df, kmod_input_params, interactive_plot=False)

    LOG.info('Calculate betastar')
    if kmod_input_params.betastar_required:
        results_df = kmod_analysis.calc_betastar(kmod_input_params, results_df, magnet1_df, magnet2_df)

    results_df.loc[:, 'TIME'] = ('{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    LOG.info('Calculate beta at instruments')
    if kmod_input_params.instruments_found != []:
        instrument_beta_df = kmod_analysis.calc_beta_at_instruments(kmod_input_params, results_df, magnet1_df, magnet2_df)

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
