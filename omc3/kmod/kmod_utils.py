import numpy as np
from utils import logging_tools, outliers
import matplotlib.pyplot as plt 
import tfs
from kmod import kmod_constants

LOG = logging_tools.get_logger(__name__)

def define_params(kmod_input_params, magnet1_df, magnet2_df):
    LOG.debug(' adding additional parameters to header ')
    
    sequence = tfs.read( kmod_constants.get_sequence_filename( kmod_input_params.beam ), index='NAME' )

    magnet1_df.headers['LENGTH'] = sequence.loc[ magnet1_df.headers['QUADRUPOLE'], 'L' ]
    magnet2_df.headers['LENGTH'] = sequence.loc[ magnet2_df.headers['QUADRUPOLE'], 'L' ]

    magnet1_df.headers['POLARITY'] = np.sign(sequence.loc[ magnet1_df.headers['QUADRUPOLE'], 'K1L' ])
    magnet2_df.headers['POLARITY'] = np.sign(sequence.loc[ magnet2_df.headers['QUADRUPOLE'], 'K1L' ])

    magnet1_position_center = sequence.loc[ magnet1_df.headers['QUADRUPOLE'], 'S' ] - magnet1_df.headers['LENGTH']/2.
    magnet2_position_center = sequence.loc[ magnet2_df.headers['QUADRUPOLE'], 'S' ] - magnet2_df.headers['LENGTH']/2.

    ip_position = ( magnet1_position_center + magnet2_position_center )/2.

    magnet1_df.headers['LSTAR'] = np.abs( ip_position - magnet1_position_center ) - magnet1_df.headers['LENGTH']/2.
    magnet2_df.headers['LSTAR'] = np.abs( ip_position - magnet2_position_center ) - magnet2_df.headers['LENGTH']/2.

    return magnet1_df, magnet2_df

def add_tuneuncertainty( magnet_df,  kmod_input_params ):
    LOG.debug('adding {} units tune measurement uncertainty'.format(kmod_input_params.tune_uncertainty))
    magnet_df['TUNEX_ERR'] = np.sqrt(magnet_df['TUNEX_ERR']**2 + kmod_input_params.tune_uncertainty**2) 
    magnet_df['TUNEY_ERR'] = np.sqrt(magnet_df['TUNEY_ERR']**2 + kmod_input_params.tune_uncertainty**2) 

    return magnet_df

def plot_cleaned_data( magnet1_df, magnet2_df, fit_param ):

    pass

