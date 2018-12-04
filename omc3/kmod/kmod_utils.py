import numpy as np
from utils import logging_tools, outliers
import matplotlib.pyplot as plt 

LOG = logging_tools.get_logger(__name__)



def add_tuneuncertainty( magnet_df,  kmod_input_params ):
    LOG.debug('adding {} units tune measurement uncertainty'.format(kmod_input_params.tune_uncertainty))
    magnet_df['TUNEX_ERR'] = np.sqrt(magnet_df['TUNEX_ERR']**2 + kmod_input_params.tune_uncertainty**2) 
    magnet_df['TUNEY_ERR'] = np.sqrt(magnet_df['TUNEY_ERR']**2 + kmod_input_params.tune_uncertainty**2) 

    return magnet_df

def plot_cleaned_data( magnet1_df, magnet2_df, fit_param ):

    pass

