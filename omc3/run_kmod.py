import os
from utils import logging_tools, iotools
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

    iotools.create_dirs( kmod_constants.get_working_directory( kmod_input_params ) )

    LOG.info('get inputfiles')
    
    magnet1_df, magnet2_df = kmod_get_files.merge_data( kmod_input_params )

    magnet1_df, magnet2_df = kmod_utils.define_params( kmod_input_params, magnet1_df, magnet2_df )
    
    magnet1_df = kmod_utils.add_tuneuncertainty(magnet1_df, kmod_input_params)
    magnet2_df = kmod_utils.add_tuneuncertainty(magnet2_df, kmod_input_params)
    
    LOG.info('clean data')
    
    magnet1_df = kmod_cleaning.clean_data( kmod_input_params, magnet1_df )
    magnet2_df = kmod_cleaning.clean_data( kmod_input_params, magnet2_df )
    
    LOG.info('run simplex')
    
    magnet1_df, magnet2_df = kmod_analysis.analyse(magnet1_df, magnet2_df, kmod_input_params)

    kmod_utils.plot_cleaned_data( magnet1_df, magnet2_df, kmod_input_params, interactive_plot=True )    

    tfs.write_tfs( os.path.join( kmod_constants.get_working_directory( kmod_input_params ), '{:s}.tfs'.format( magnet1_df.headers['QUADRUPOLE'] ) ) , magnet1_df )
    tfs.write_tfs( os.path.join( kmod_constants.get_working_directory( kmod_input_params ), '{:s}.tfs'.format( magnet2_df.headers['QUADRUPOLE'] ) ) , magnet2_df )

    LOG.info('calc betastar')
    
    # results file format index magnet columns timestamp betastarX betastarY betawaistX betawaistY etc.

    LOG.info('calc beta at inst')
    



if __name__ == '__main__':
    with logging_tools.DebugMode(active=True, log_file=""):
        analyse_kmod()
