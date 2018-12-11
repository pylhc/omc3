import pandas as pd
import tfs
import os
from utils import logging_tools
import numpy as np
import datetime
from kmod import kmod_constants

LOG = logging_tools.get_logger(__name__)
SIDES=('L', 'R')

def return_filename( kmod_input_params ):
    # TODO rewrite to make less cluttered
        if kmod_input_params.ip is not None:
            LOG.debug('Setting IP trim file names')
            for side in SIDES:        
                path_tunex = os.path.join( kmod_input_params.working_directory, '{:s}{:s}{:s}X.tfs'.format(kmod_input_params.ip.lower(), kmod_input_params.beam.lower(), side) )
                path_tuney = os.path.join( kmod_input_params.working_directory, '{:s}{:s}{:s}Y.tfs'.format(kmod_input_params.ip.lower(), kmod_input_params.beam.lower(), side) )
                path_k = os.path.join( kmod_input_params.working_directory, '{:s}{:s}K.tfs'.format(kmod_input_params.ip.lower(), side) )

                yield path_tunex, path_tuney, path_k
        elif kmod_input_params.circuits is not None:
            LOG.debug('Setting Circuit trim file names')
            for circuit in [kmod_input_params.circuit1, kmod_input_params.circuit2]:          
                path_tunex = os.path.join( kmod_input_params.working_directory, '{:s}_tune_x_{:s}.tfs'.format(circuit, kmod_input_params.beam.lower()) )
                path_tuney = os.path.join( kmod_input_params.working_directory, '{:s}_tune_y_{:s}.tfs'.format(circuit, kmod_input_params.beam.lower()) )
                path_k = os.path.join( kmod_input_params.working_directory, '{:s}_k.tfs'.format(circuit) )

                yield path_tunex, path_tuney, path_k

def return_magnet( kmod_input_params ):
    # a = (x for x in [1,2,3,4])
    for x in [kmod_input_params.magnet1,kmod_input_params.magnet2]:
        yield x

def return_mean_of_binned_data( bins, tune_df ):

    digitize = np.digitize( tune_df['TIME'] , bins )

    mean = [ tune_df['TUNE'][ digitize==i ].mean() for i in range( 1, len(bins) )  ]
    std = np.nan_to_num([ tune_df['TUNE'][ digitize==i ].std() for i in range( 1, len(bins) )  ])

    return mean, std

def headers_for_df( magnet, k_df ):
    LOG.debug('creating headers for DF')
    head = {}

    head['QUADRUPOLE'] = magnet
    head['DELTA_I'] = (np.max( k_df['CURRENT'] ) - np.min( k_df['CURRENT'] ))/2.
    head['START_TIME'] =  (datetime.datetime.fromtimestamp( k_df['TIME'].iloc[0] /1000.0 )).strftime( '%Y-%m-%d %H:%M:%S' )
    head['END_TIME'] =  (datetime.datetime.fromtimestamp( k_df['TIME'].iloc[-1] /1000.0 )).strftime( '%Y-%m-%d %H:%M:%S' )

    # add starting tunes/tunesplit, number of cycles, ... to header

    return head

def bin_tunes_and_k( tunex_df, tuney_df, k_df, magnet ):     

    # create bins, centered around each time step in k with width eq half distance to the next timestep
    bins = np.append( (k_df['TIME']-k_df.diff()['TIME']/2.).fillna(value=0).values, k_df['TIME'].iloc[-1] )

    tunex, tunex_err = return_mean_of_binned_data( bins, tunex_df )
    tuney, tuney_err = return_mean_of_binned_data( bins, tuney_df )

    magnet_df = tfs.TfsDataFrame( headers=headers_for_df( magnet, k_df ) ,columns= [kmod_constants.get_k_col(), kmod_constants.get_tune_col('X'), kmod_constants.get_tune_err_col('X'), kmod_constants.get_tune_col('Y'), kmod_constants.get_tune_err_col('Y')], data= np.column_stack( ( k_df['K'], tunex, tunex_err, tuney, tuney_err ) ) )

    return magnet_df

def merge_data( kmod_input_params ):

    magnet_df = []

    for (filepaths, magnet) in zip(return_filename(kmod_input_params), return_magnet(kmod_input_params)) :
        
        LOG.debug('loading tune from {:s}'.format( filepaths[0] ))
        tunex_df = tfs.read( filepaths[0] )

        LOG.debug('loading tune from {:s}'.format( filepaths[1] ))
        tuney_df = tfs.read( filepaths[1] )

        LOG.debug('loading k from {:s}'.format( filepaths[2] ))
        k_df = tfs.read( filepaths[2] )

        LOG.debug('binning data')
        magnet_df.append( bin_tunes_and_k( tunex_df, tuney_df, k_df, magnet ))

        
    return magnet_df
