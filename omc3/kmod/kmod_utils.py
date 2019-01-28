import numpy as np
import os
from utils import logging_tools
import matplotlib.pyplot as plt 
import tfs
from kmod import kmod_constants, kmod_analysis

plt.rc('text', usetex=True)

LOG = logging_tools.get_logger(__name__)
PLANES = ['X', 'Y']

def find_magnet( beam, circuit):
    
    sequence = tfs.read( kmod_constants.get_sequence_filename( beam ) )

    circuit = circuit.split('.')

    magnetname = sequence[ sequence['NAME'].str.contains( r'MQ\w+\.{:s}{:s}{:s}\.\w+'.format( circuit[0][-1], circuit[1][0], circuit[1][1]  ) ) ]['NAME'].values[0] 

    return magnetname

def define_params(kmod_input_params, magnet1_df, magnet2_df):
    LOG.debug(' adding additional parameters to header ')
    
    sequence = tfs.read( kmod_constants.get_sequence_filename( kmod_input_params.beam ), index='NAME' )

    for magnet_df in [magnet1_df, magnet2_df]:

        magnet_df.headers['LENGTH'] = sequence.loc[ magnet_df.headers['QUADRUPOLE'], 'L' ]
        magnet_df.headers['POLARITY'] = np.sign(sequence.loc[ magnet_df.headers['QUADRUPOLE'], 'K1L' ])
        magnet_df.headers['LSTAR'] = np.abs( ip_position - magnet_position_center ) - magnet_df.headers['LENGTH']/2.
    

    magnet1_position_center = sequence.loc[ magnet1_df.headers['QUADRUPOLE'], 'S' ] - magnet1_df.headers['LENGTH']/2.
    magnet2_position_center = sequence.loc[ magnet2_df.headers['QUADRUPOLE'], 'S' ] - magnet2_df.headers['LENGTH']/2.

    ip_position = ( magnet1_position_center + magnet2_position_center )/2.


    if magnet1_position_center < magnet2_position_center:
        left_magnet_df = magnet1_df
        right_magnet_df = magnet2_df
    elif magnet2_position_center< magnet1_position_center:
        left_magnet_df = magnet2_df
        right_magnet_df = magnet1_df

    between_magnets_df = sequence.reset_index().truncate( before= sequence.index.get_loc( left_magnet_df.headers['QUADRUPOLE'] ), after= sequence.index.get_loc( right_magnet_df.headers['QUADRUPOLE'] ) , axis='index' )

    if between_magnets_df.isin( ['OMK'] ).any().loc['PARENT']:
        kmod_input_params.set_betastar_required()

    for instrument in kmod_input_params.instruments:
        if between_magnets_df.isin([instrument]).any().loc['KEYWORD']:
            kmod_input_params.set_instruments_found(instrument)
            kmod_input_params.set_instrument_position( instrument , dict( zip( between_magnets_df.loc[ between_magnets_df['KEYWORD'] == instrument ]['NAME'].values, (between_magnets_df.loc[ between_magnets_df['KEYWORD'] == instrument ]['S'].values - ip_position) ) ) )

    return magnet1_df, magnet2_df

def add_tuneuncertainty( magnet_df,  kmod_input_params ):
    LOG.debug('adding {} units tune measurement uncertainty'.format(kmod_input_params.tune_uncertainty))

    for plane in PLANES:
        magnet_df[kmod_constants.get_tune_err_col( plane )] = np.sqrt(magnet_df[kmod_constants.get_tune_err_col( plane )]**2 + kmod_input_params.tune_uncertainty**2) 
    

    return magnet_df

def ax_plot(ax, magnet_df, plane ):

    ax.set_title( magnet_df.headers['QUADRUPOLE'], fontsize=15 )
    ax.errorbar( 
        (magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==True )[kmod_constants.get_k_col()].dropna() - magnet_df.headers[kmod_constants.get_k_col()] )*1E3,  
        magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==True )[kmod_constants.get_tune_col( plane )].dropna(),
        yerr = magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==True )[kmod_constants.get_tune_err_col( plane )].dropna(),
        color ='blue',
        fmt='o',
        label='Data',
        zorder=1
        )
    ax.errorbar( 
        (magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==False )[kmod_constants.get_k_col()].dropna() - magnet_df.headers[kmod_constants.get_k_col()]  )*1E3,  
        magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==False )[kmod_constants.get_tune_col( plane )].dropna(),
        yerr = magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==False )[kmod_constants.get_tune_err_col( plane )].dropna(),
        color ='orange',
        fmt='o',
        label='Cleaned',
        zorder=2
        )


    ax.plot( 
        (magnet_df.where( magnet_df[kmod_constants.get_cleaned_col( plane )]  ==True )[kmod_constants.get_k_col()].dropna() - magnet_df.headers[kmod_constants.get_k_col()]  )*1E3,
        kmod_analysis.fit_prec( kmod_analysis.return_fit_input( magnet_df, plane )  , magnet_df.headers[kmod_constants.get_av_beta_col( plane )]) + magnet_df.headers[kmod_constants.get_tune_col( plane )] ,
        color='red',
        label='Fit',
        zorder=3
        )

    ax.set_xlabel( r'$ \Delta K $', fontsize=15 )    
    ax.set_ylabel( r'$ Q_{{{:s}}} $'.format(plane.upper()) , fontsize=15)    
    
    return

def plot_cleaned_data( magnet1_df, magnet2_df, kmod_input_params, interactive_plot=False ):

    fig, ax = plt.subplots( nrows=2, ncols=2, figsize=(10,10) )

    ax_plot( ax[0,0], magnet1_df, 'X' )
    ax_plot( ax[1,0], magnet1_df, 'Y' )
    ax_plot( ax[0,1], magnet2_df, 'X' )
    ax_plot( ax[1,1], magnet2_df, 'Y' )              

    ax[1,1].legend()

    plt.tight_layout()
    plt.savefig(  os.path.join(  kmod_constants.get_working_directory( kmod_input_params ) , 'fit_plots.pdf' ) )
    if interactive_plot == True:
        plt.show()
    
    return

    

