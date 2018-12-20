import scipy.optimize
import numpy as np
import math
from utils import logging_tools
from kmod import kmod_constants
import tfs

LOG = logging_tools.get_logger(__name__)

PLANES = ['X', 'Y']

def calc_betastar( kmod_input_params, results_df):
    
    if kmod_input_params.betastar_required:

        sign = np.array( [[0,0],[1,0],[-1,0],[0,1],[0,-1]] )

        for plane in PLANES:

            betastar = \
            (float(results_df.loc[:, kmod_constants.get_betawaist_col(plane)].values) + sign[:,0] * float(results_df.loc[:, kmod_constants.get_betawaist_err_col(plane)].values) )\
            + (float(results_df.loc[:, kmod_constants.get_waist_col(plane)].values)+ sign[:,1]* float(results_df.loc[:, kmod_constants.get_waist_err_col(plane)].values) )**2\
            /(float(results_df.loc[:, kmod_constants.get_betawaist_col(plane)].values) + sign[:,0] * float(results_df.loc[:, kmod_constants.get_betawaist_err_col(plane)].values) )

            betastar_err = np.sqrt(np.sum(np.maximum(np.absolute(betastar[1::2]-betastar[0]),abs(betastar[1::2]-betastar[0]))**2))

            results_df[ kmod_constants.get_betastar_col(plane) ] = betastar[0]
            results_df[ kmod_constants.get_betastar_err_col(plane) ] = betastar_err

    cols = results_df.columns.tolist()
    cols = [cols[0]]+cols[-4:]+cols[1:-4]
    results_df = results_df[cols]

    return results_df

def fit_prec(x, beta_av):
    
    dQ = (1/(2.*np.pi)) * np.arccos( np.cos(2 * np.pi * np.modf(x[1])[0] ) - 0.5 * beta_av * x[0] * np.sin( 2 * np.pi * np.modf(x[1])[0] )  ) - np.modf(x[1])[0]
    return dQ

np.vectorize(fit_prec)

def fit_approx(x, beta_av):
    
    dQ = beta_av*x[0]/(4*np.pi)
    return dQ

np.vectorize(fit_approx)

def average_beta_from_Tune(Q, TdQ, l, Dk):
    """Calculates average beta function in quadrupole from Tunechange TdQ and delta K """
    
    beta_av = 2 * (1 / math.tan(2 * math.pi * Q) * (1 - math.cos(2 * math.pi * TdQ)) + math.sin(2 * math.pi * TdQ)) / ( l * Dk)
    return abs(beta_av)

def average_beta_focussing_quadrupole(b, w, L, K, Lstar):

    beta0 = b + ((Lstar - w) ** 2 / (b))
    alpha0 = -(Lstar - w) / b
    average_beta =   (beta0/2.) * ( 1 + ( ( np.sin(2 * np.sqrt(K) * L ) ) / ( 2 * np.sqrt(K) * L ) ) ) \
                    - alpha0 * ( ( np.sin( np.sqrt(K) * L )**2 ) / ( K * L ) ) \
                    + (1/(2*K)) * ( (1 + alpha0**2)/(beta0) ) * ( 1 - ( ( np.sin(2 * np.sqrt(K) * L) ) / ( 2 * np.sqrt(K) * L ) ) )

    return average_beta
np.vectorize(average_beta_focussing_quadrupole) 

def average_beta_defocussing_quadrupole(b, w, L, K, Lstar):
    beta0 = b + ((Lstar - w) ** 2 / (b))
    alpha0 = -(Lstar - w) / b
    average_beta =   (beta0/2.) * ( 1 + ( ( np.sinh(2 * np.sqrt(K) * L ) ) / ( 2 * np.sqrt(K) * L ) ) ) \
                    - alpha0 * ( ( np.sinh( np.sqrt(K) * L )**2 ) / ( K * L ) ) \
                    + (1/(2*K)) * ( (1 + alpha0**2)/(beta0) ) * ( ( ( np.sinh(2 * np.sqrt(K) * L) ) / ( 2 * np.sqrt(K) * L ) ) - 1 )

    return average_beta
np.vectorize(average_beta_defocussing_quadrupole)


def calc_tune( magnet_df ):    
    
    magnet_df.headers[kmod_constants.get_tune_col('X')] = np.average( magnet_df.where( magnet_df[kmod_constants.get_cleaned_col('X')]  ==True )[kmod_constants.get_tune_col('X')].dropna() )
    magnet_df.headers[kmod_constants.get_tune_col('Y')] = np.average( magnet_df.where( magnet_df[kmod_constants.get_cleaned_col('Y')]  ==True )[kmod_constants.get_tune_col('Y')].dropna() )
    
    return magnet_df

def calc_k( magnet_df ):    
    
    magnet_df.headers[kmod_constants.get_k_col()] = np.average(  magnet_df.where( magnet_df[kmod_constants.get_cleaned_col('X')]  ==True )[kmod_constants.get_k_col()].dropna() )
    magnet_df.headers[kmod_constants.get_k_col()] = np.average(  magnet_df.where( magnet_df[kmod_constants.get_cleaned_col('Y')]  ==True )[kmod_constants.get_k_col()].dropna() )
    
    return magnet_df
    
def return_fit_input( magnet_df, plane ):

    x = np.zeros( ( 2, len( magnet_df.where( magnet_df[kmod_constants.get_cleaned_col(plane)]  ==True )[kmod_constants.get_k_col()].dropna() ) ) )
    sign = magnet_df.headers['POLARITY'] if plane=='X' else -1*magnet_df.headers['POLARITY']        
    x[0, : ] = sign*( magnet_df.where( magnet_df[kmod_constants.get_cleaned_col(plane)]  ==True )[kmod_constants.get_k_col()].dropna() - magnet_df.headers[kmod_constants.get_k_col()] ) * magnet_df.headers['LENGTH']
    x[1, : ] = magnet_df.headers[ kmod_constants.get_tune_col(plane) ]

    return x

def do_fit( magnet_df, plane, use_approx=False ):
    if not use_approx:
        av_beta, av_beta_err = scipy.optimize.curve_fit(
            fit_prec,
            xdata= return_fit_input(magnet_df, plane),
            ydata = magnet_df.where( magnet_df[kmod_constants.get_cleaned_col(plane)]  ==True )[kmod_constants.get_tune_col(plane)].dropna() - magnet_df.headers[ kmod_constants.get_tune_col(plane) ],
            p0= 1
        )     

    elif use_approx:

        av_beta, av_beta_err = scipy.optimize.curve_fit(
            fit_approx,
            xdata= return_fit_input(magnet_df, plane),
            ydata = magnet_df.where( magnet_df[kmod_constants.get_cleaned_col(plane)]  ==True )[kmod_constants.get_tune_col(plane)].dropna() - magnet_df.headers[ kmod_constants.get_tune_col(plane) ],
            p0= 1
        )

    return av_beta[0], np.sqrt(np.diag(av_beta_err))[0]
    

def get_av_beta(magnet_df):

    magnet_df.headers[ kmod_constants.get_av_beta_col( 'X') ], magnet_df.headers[ kmod_constants.get_av_beta_err_col( 'X') ] = do_fit( magnet_df, 'X' )
    magnet_df.headers[ kmod_constants.get_av_beta_col( 'Y') ], magnet_df.headers[ kmod_constants.get_av_beta_err_col( 'Y') ] = do_fit( magnet_df, 'Y' )

    return magnet_df

def return_df(magnet1_df, magnet2_df, plane):

    if plane =='X':
        if magnet1_df.headers['POLARITY'] == 1 and magnet2_df.headers['POLARITY'] == -1:
            return magnet1_df, magnet2_df
        elif magnet1_df.headers['POLARITY'] == -1 and magnet2_df.headers['POLARITY'] == 1:
            return magnet2_df, magnet1_df

    elif plane =='Y':
        if magnet1_df.headers['POLARITY'] == -1 and magnet2_df.headers['POLARITY'] == 1:
            return magnet1_df, magnet2_df
        elif magnet1_df.headers['POLARITY'] == 1 and magnet2_df.headers['POLARITY'] == -1:
            return magnet2_df, magnet1_df

def chi2(x, magnet1_df, magnet2_df, plane  ):

    b = x[0]
    w = x[1]

    foc_magnet_df, def_magnet_df = return_df( magnet1_df, magnet2_df, plane )
    
    c2=\
    (average_beta_focussing_quadrupole(b, w, foc_magnet_df.headers[ 'LENGTH' ], foc_magnet_df.headers[ kmod_constants.get_k_col() ], foc_magnet_df.headers[ 'LSTAR' ]) - foc_magnet_df.headers[ kmod_constants.get_av_beta_col( plane ) ] ) ** 2 \
    + (average_beta_defocussing_quadrupole(b, -w, def_magnet_df.headers[ 'LENGTH' ], def_magnet_df.headers[ kmod_constants.get_k_col() ], def_magnet_df.headers[ 'LSTAR' ]) - def_magnet_df.headers[ kmod_constants.get_av_beta_col( plane ) ]  ) ** 2

    return c2

def get_beta_waist( magnet1_df, magnet2_df, kmod_input_params, plane ):
    fun = lambda x: chi2(x, magnet1_df, magnet2_df, plane)

    results = scipy.optimize.minimize( fun, kmod_input_params.return_guess(plane), method='nelder-mead', tol=1E-9 )

    return results.x[0], results.x[1]



def analyse( magnet1_df, magnet2_df, kmod_input_params ):

    LOG.info('get tune')

    magnet1_df = calc_tune(magnet1_df)
    magnet2_df = calc_tune(magnet2_df)

    LOG.info('get k')

    magnet1_df = calc_k(magnet1_df)
    magnet2_df = calc_k(magnet2_df)

    LOG.info('fit average beta')

    magnet1_df = get_av_beta( magnet1_df )
    magnet2_df = get_av_beta( magnet2_df )

    LOG.info('simplex to determine beta waist')

    results_x = get_beta_waist(magnet1_df, magnet2_df, kmod_input_params, 'X')
    results_y = get_beta_waist(magnet1_df, magnet2_df, kmod_input_params, 'Y')

    results_df = tfs.TfsDataFrame( columns=['LABEL', kmod_constants.get_betawaist_col('X'), kmod_constants.get_betawaist_err_col('X'), kmod_constants.get_waist_col('X'), kmod_constants.get_waist_err_col('X'), kmod_constants.get_betawaist_col('Y'), kmod_constants.get_betawaist_err_col('Y'), kmod_constants.get_waist_col('Y'), kmod_constants.get_waist_err_col('Y')] , data=[np.hstack( (kmod_constants.get_label(kmod_input_params), results_x[0], 1E-4, results_x[1], 1E-4, results_y[0], 1E-4, results_y[1], 1E-4) )]  )


    return magnet1_df, magnet2_df, results_df
