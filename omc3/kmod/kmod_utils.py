import numpy as np
from utils import logging_tools, outliers
import matplotlib.pyplot as plt 
import tfs
from kmod import kmod_constants, kmod_analysis

plt.rc('text', usetex=True)

LOG = logging_tools.get_logger(__name__)

def find_magnet(kmod_input_params, magnet1_df, magnet2_df):
    pass

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

def ax_plot(ax, magnet_df, plane ):

    ax.set_title( magnet_df.headers['QUADRUPOLE'], fontsize=15 )
    ax.errorbar( 
        (magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==True )[kmod_constants.get_k_col()].dropna() - magnet_df.headers[kmod_constants.get_k_col()] )*1E3,  
        magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==True )[kmod_constants.get_tune_col( plane )].dropna(),
        yerr = magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==True )[kmod_constants.get_tune_err_col( plane )].dropna(),
        color ='blue',
        fmt='o',
        zorder=1
        )
    ax.errorbar( 
        (magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==False )[kmod_constants.get_k_col()].dropna() - magnet_df.headers[kmod_constants.get_k_col()]  )*1E3,  
        magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==False )[kmod_constants.get_tune_col( plane )].dropna(),
        yerr = magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==False )[kmod_constants.get_tune_err_col( plane )].dropna(),
        color ='orange',
        fmt='o',
        zorder=2
        )


    ax.plot( 
        (magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==True )[kmod_constants.get_k_col()].dropna() - magnet_df.headers[kmod_constants.get_k_col()]  )*1E3,
        kmod_analysis.fit_prec( kmod_analysis.return_fit_input( magnet_df, plane )  , magnet_df.headers[kmod_constants.get_av_beta_col( plane )]) + magnet_df.headers[kmod_constants.get_tune_col( plane )] ,
        color='red',
        zorder=3
        )

    ax.set_xlabel( r'$ \Delta K $', fontsize=15 )    
    ax.set_ylabel( r'$ Q_{{{:s}}} $'.format(plane.upper()) , fontsize=15)    

    return

def plot_cleaned_data( magnet1_df, magnet2_df, interactive_plot=False ):

    fig, ax = plt.subplots( nrows=2, ncols=2, figsize=(10,10) )


    ax_plot( ax[0,0], magnet1_df, 'X' )
    ax_plot( ax[1,0], magnet1_df, 'Y' )
    ax_plot( ax[0,1], magnet2_df, 'X' )
    ax_plot( ax[1,1], magnet2_df, 'Y' )
              

    plt.tight_layout()
    plt.savefig('fit_plots.pdf')
    if interactive_plot == True:
        plt.show()
    
    return

    

