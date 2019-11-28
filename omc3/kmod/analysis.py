import scipy.optimize
from os.path import join
import os
import numpy as np
import tfs
import datetime
from tfs import tools as tfstools
from utils import logging_tools
from kmod import helper
from kmod.constants import CLEANED, PLANES, K, TUNE, ERR, BETA, STAR, WAIST, PHASEADV, AVERAGE
from definitions import formats

LOG = logging_tools.get_logger(__name__)


def return_sign_for_err(n):
    """
    creates an array of form
    [[ 0.  0.  0.]
    [ 1.  0.  0.]
    [-1. -0. -0.]
    [ 0.  1.  0.]
    [-0. -1. -0.]
    [ 0.  0.  1.]
    [-0. -0. -1.]] for error calculation
    columns corresponds to error i.e. first column for dQ etc.
    """
    sign = np.zeros((2*n+1, n))

    sign[1::2] = np.eye(n)
    sign[2::2] = -np.eye(n)
    return sign


def propagate_beta_in_drift(beta_waist, drift):
    beta = beta_waist + drift**2/beta_waist
    return beta


def calc_betastar(kmod_input_params, results_df, l_star):
    sign = return_sign_for_err(2)
    for plane in PLANES:
        betastar = propagate_beta_in_drift((results_df.loc[:, f"{BETA}{WAIST}{plane}"].values + sign[:, 0] * results_df.loc[:, f"{ERR}{BETA}{WAIST}{plane}"].values),
                                           (results_df.loc[:, f"{WAIST}{plane}"].values + sign[:, 1] * results_df.loc[:, f"{ERR}{WAIST}{plane}"].values))
        betastar_err = get_err(betastar[1::2]-betastar[0])

        if kmod_input_params.no_sig_digits:
            results_df[f"{BETA}{STAR}{plane}"], results_df[f"{ERR}{BETA}{STAR}{plane}"] = (betastar[0], betastar_err)
        else:
            results_df[f"{BETA}{STAR}{plane}"], results_df[f"{ERR}{BETA}{STAR}{plane}"] = tfstools.significant_numbers(betastar[0], betastar_err)

    # reindex df to put betastar first
    cols = results_df.columns.tolist()
    cols = [cols[0]]+cols[-4:]+cols[1:-4]
    results_df = results_df.reindex(columns=cols)

    for plane in PLANES:
        results_df[f"{PHASEADV}{plane}"], results_df[f"{ERR}{PHASEADV}{plane}"] = phase_adv_from_kmod(
            l_star, betastar[0], betastar_err,
            results_df.loc[:, f"{WAIST}{plane}"].values,
            results_df.loc[:, f"{ERR}{WAIST}{plane}"].values)

    return results_df


def phase_adv_from_kmod(lstar, betastar, ebetastar, waist, ewaist):
    return _phase_adv_from_kmod_value(lstar, betastar, waist),\
           _phase_adv_from_kmod_err(lstar, betastar, ebetastar, waist, ewaist)


def _phase_adv_from_kmod_value(lstar, betastar, waist):
    return (np.arctan((lstar - waist) / betastar) +
            np.arctan((lstar + waist) / betastar)) / (2 * np.pi)


def _phase_adv_from_kmod_err(lstar, betastar, ebetastar, waist, ewaist):
    numer = (2 * lstar * (betastar ** 2 + lstar ** 2 - waist ** 2) * ebetastar) ** 2
    numer = numer + (4 * betastar * lstar * waist * ewaist) ** 2
    denom = (betastar ** 2 + (lstar - waist) ** 2) ** 2
    denom = denom * (betastar ** 2 + (lstar + waist) ** 2) ** 2
    return np.sqrt(numer / denom) / (2 * np.pi)


def calc_beta_inst(name, position, results_df, magnet1_df, magnet2_df, kmod_input_params):
    betas = np.zeros((2, 2))
    sign = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    for i, plane in enumerate(PLANES):
        waist = results_df.loc[:, f"{WAIST}{plane}"].values
        if magnet1_df.headers['POLARITY'] == 1 and magnet2_df.headers['POLARITY'] == -1:
            waist = -waist
        if plane == 'Y':
            waist = -waist

        beta = propagate_beta_in_drift((results_df.loc[:, f"{BETA}{WAIST}{plane}"].values + sign[:, 0] * results_df.loc[:, f"{ERR}{BETA}{WAIST}{plane}"].values),
                                       ((waist - position) + sign[:, 1] * results_df.loc[:, f"{ERR}{WAIST}{plane}"].values))
        beta_err = get_err(beta[1::2]-beta[0])
        if kmod_input_params.no_sig_digits:
            betas[i, 0], betas[i, 1] = beta[0], beta_err
        else:
            betas[i, 0], betas[i, 1] = tfstools.significant_numbers(beta[0], beta_err)
    return name, betas[0, 0], betas[0, 1], betas[1, 0], betas[1, 1]


def calc_beta_at_instruments(kmod_input_params, results_df, magnet1_df, magnet2_df):

    beta_instr = []

    for instrument in kmod_input_params.instruments_found:
        positions = getattr(kmod_input_params, instrument)

        for name, position in positions.items():
            beta_instr.append(calc_beta_inst(
                name, position, results_df, magnet1_df, magnet2_df, kmod_input_params))

    instrument_beta_df = tfs.TfsDataFrame(
        columns=['INSTRUMENT',
                 f"{BETA}{'X'}",
                 f"{ERR}{BETA}{'X'}",
                 f"{BETA}{'Y'}",
                 f"{ERR}{BETA}{'Y'}",
                 ],
        data=beta_instr)
    return instrument_beta_df


def fit_prec(x, beta_av):
    twopiQ = 2 * np.pi * np.modf(x[1])[0]
    dQ = (1/(2.*np.pi)) * np.arccos(np.cos(twopiQ) -
                                    0.5 * beta_av * x[0] * np.sin(twopiQ)) - np.modf(x[1])[0]
    return dQ


np.vectorize(fit_prec)


def fit_approx(x, beta_av):
    dQ = beta_av*x[0]/(4*np.pi)
    return dQ


np.vectorize(fit_approx)


def average_beta_from_Tune(Q, TdQ, l, Dk):
    """Calculates average beta function in quadrupole from Tunechange TdQ and delta K """

    beta_av = 2 * (1 / np.tan(2 * np.pi * Q) *
              (1 - np.cos(2 * np.pi * TdQ)) + np.sin(2 * np.pi * TdQ)) / (l * Dk)
    return abs(beta_av)


def average_beta_focussing_quadrupole(b, w, L, K, Lstar):

    beta0 = b + ((Lstar - w) ** 2 / b)
    alpha0 = -(Lstar - w) / b
    average_beta = (beta0/2.) * (1 + ((np.sin(2 * np.sqrt(abs(K)) * L)) / (2 * np.sqrt(abs(K)) * L))) \
                   - alpha0 * ((np.sin(np.sqrt(abs(K)) * L)**2) / (abs(K) * L)) \
                   + (1/(2*abs(K))) * ((1 + alpha0**2) / beta0) * \
                   (1 - ((np.sin(2 * np.sqrt(abs(K)) * L)) / (2 * np.sqrt(abs(K)) * L)))

    return average_beta


np.vectorize(average_beta_focussing_quadrupole)


def average_beta_defocussing_quadrupole(b, w, L, K, Lstar):
    beta0 = b + ((Lstar - w) ** 2 / b)
    alpha0 = -(Lstar - w) / b
    average_beta = (beta0/2.) * (1 + ((np.sinh(2 * np.sqrt(abs(K)) * L)) / (2 * np.sqrt(abs(K)) * L))) \
                   - alpha0 * ((np.sinh(np.sqrt(abs(K)) * L)**2) / (abs(K) * L)) \
                   + (1/(2*abs(K))) * ((1 + alpha0**2) / beta0) * \
                   (((np.sinh(2 * np.sqrt(abs(K)) * L)) / (2 * np.sqrt(abs(K)) * L)) - 1)

    return average_beta


np.vectorize(average_beta_defocussing_quadrupole)


def calc_tune(magnet_df):
    for plane in PLANES:
        magnet_df.headers[f"{TUNE}{plane}"] = np.average(magnet_df.where(
            magnet_df[f"{CLEANED}{plane}"])[f"{TUNE}{plane}"].dropna())
    return magnet_df


def calc_k(magnet_df):
    magnet_df.headers[K] = np.average(magnet_df.where(magnet_df[f"{CLEANED}X"])[K].dropna())
    return magnet_df


def return_fit_input(magnet_df, plane):

    x = np.zeros((2, len(magnet_df.where(magnet_df[f"{CLEANED}{plane}"])[K].dropna())))

    sign = magnet_df.headers['POLARITY'] if plane == 'X' else -1 * magnet_df.headers['POLARITY']
    x[0, :] = sign*(
            magnet_df.where(magnet_df[f"{CLEANED}{plane}"])[K].dropna() -
            magnet_df.headers[K]) * magnet_df.headers['LENGTH']
    x[1, :] = magnet_df.headers[f"{TUNE}{plane}"]

    return x


def do_fit(magnet_df, plane, use_approx=False):
    if not use_approx:
        fun = fit_prec
    elif use_approx:
        fun = fit_approx

    if not np.any(magnet_df.where(magnet_df[f"{CLEANED}{plane}"])[f"{ERR}{TUNE}{plane}"].dropna()):
        sigma = None
        absolute_sigma = False
    else:
        sigma = magnet_df.where(magnet_df[f"{CLEANED}{plane}"])[
            f"{ERR}{TUNE}{plane}"].dropna()
        absolute_sigma = True

    av_beta, av_beta_err = scipy.optimize.curve_fit(
        fun,
        xdata=return_fit_input(magnet_df, plane),
        ydata=magnet_df.where(magnet_df[f"{CLEANED}{plane}"])[
            f"{TUNE}{plane}"].dropna() - magnet_df.headers[f"{TUNE}{plane}"],
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        p0=1
    )
    return np.abs(av_beta[0]), np.sqrt(np.diag(av_beta_err))[0]


def get_av_beta(magnet_df):
    for plane in PLANES:
        magnet_df.headers[f"{AVERAGE}{BETA}{plane}"], magnet_df.headers[f"{ERR}{AVERAGE}{BETA}{plane}"] = do_fit(magnet_df, plane)
    return magnet_df


def check_polarity(magnet1_df, magnet2_df, sign):
    left, right = sign
    return magnet1_df.headers['POLARITY'] == left and magnet2_df.headers['POLARITY'] == right


def return_df(magnet1_df, magnet2_df, plane):

    sign = {'X': np.array([1, -1]), 'Y': np.array([-1, 1])}

    if check_polarity(magnet1_df, magnet2_df, sign[plane]):
        return magnet1_df, magnet2_df
    elif check_polarity(magnet1_df, magnet2_df, -sign[plane]):
        return magnet2_df, magnet1_df


def chi2(x, foc_magnet_df, def_magnet_df, plane, kmod_input_params, sign):

    b = x[0]
    w = x[1]

     # Replace LSTAR by BPM distance
    #delta_s = def_magnet_df.headers['LSTAR']
    #phase_adv = phase_adv_from_kmod(delta_s,b,0.0,w,0.0)[0]

    # Last BPMs left and right
    BPML = 'BPMSW.1L' + kmod_input_params.ip[-1] + '.' + kmod_input_params.beam
    BPMR = 'BPMSW.1R' + kmod_input_params.ip[-1] + '.' + kmod_input_params.beam
    
    # Position s of BPML
    twiss_df = tfs.read(os.path.join(f'{kmod_input_params.twiss_model_dir}', f'twiss.dat'), index='NAME')
    pos_1L = twiss_df.loc[BPML,'S']
    pos_1R = twiss_df.loc[BPMR,'S']
    BPM_distance = np.abs(pos_1R - pos_1L)/2.0

    # phase from kmod using beta and waist guess

    # Using LSTAR from headers
    # phase_adv = phase_adv_from_kmod(def_magnet_df.headers['LSTAR'],b,0.0,w,0.0)[0]

    # Using real distance between BPMs
    phase_adv = phase_adv_from_kmod(BPM_distance,b,0.0,w,0.0)[0]

    # Make it more general for any BPM

    phase_adv_model = 0.0
    phase_adv_err = 0.0

    weight = kmod_input_params.phase_weight
    scale = kmod_input_params.phase_scale

    if os.path.exists(os.path.join(f'{kmod_input_params.meas_directory}',f'getphase{plane.lower()}.out')):
        # get measured phase from getphase[x/y].out
        phase_df = tfs.read( os.path.join(f'{kmod_input_params.meas_directory}',f'getphase{plane.lower()}.out'), index='NAME')
        phase_adv_model = phase_df.loc[BPML,'PHASE'+plane]
        phase_adv_err = phase_df.lco[BPML,'STDPH'+plane]
        # getphase is python2, python3 is phase_x/y


    elif (os.path.exists(os.path.join(f'{kmod_input_params.twiss_model_dir}', f'twiss.dat')) and weight!=0):
        # get phase from twiss model
        twiss_df = tfs.read(os.path.join(f'{kmod_input_params.twiss_model_dir}', f'twiss.dat'), index='NAME')
        phase_1L = twiss_df.loc[BPML,'MU'+plane]
        phase_1R = twiss_df.loc[BPMR,'MU'+plane]
        phase_adv_model = abs(phase_1R - phase_1L)
        phase_adv_err = 0.5e-3 # this number is given by Andrea's estimations 

    #weight = kmod_input_params.phase_weight
    #scale = kmod_input_params.phase_scale

    c2 = (1-weight)*((average_beta_focussing_quadrupole(b, w, foc_magnet_df.headers['LENGTH'] +
        sign[0] * kmod_input_params.errorL, foc_magnet_df.headers[K] +
        sign[1] * kmod_input_params.errorK * foc_magnet_df.headers[K],
        foc_magnet_df.headers['LSTAR'] +
        sign[2] * kmod_input_params.misalignment) -
        foc_magnet_df.headers[f"{AVERAGE}{BETA}{plane}"] +
        sign[3] * foc_magnet_df.headers[f"{ERR}{AVERAGE}{BETA}{plane}"]) ** 2 +
        (average_beta_defocussing_quadrupole(b, -w, def_magnet_df.headers['LENGTH'] +
        sign[4] * kmod_input_params.errorL, def_magnet_df.headers[K] +
        sign[5] * kmod_input_params.errorK * def_magnet_df.headers[K],
        def_magnet_df.headers['LSTAR'] +
        sign[6] * kmod_input_params.misalignment) -
        def_magnet_df.headers[f"{AVERAGE}{BETA}{plane}"] +
        sign[7] * foc_magnet_df.headers[f"{ERR}{AVERAGE}{BETA}{plane}"]) ** 2) + scale*weight*((phase_adv - (phase_adv_model+sign[8]*phase_adv_err))**2)

    return c2


def get_beta_waist(magnet1_df, magnet2_df, kmod_input_params, plane):
    n = 9
    sign = return_sign_for_err(n)
    foc_magnet_df, def_magnet_df = return_df(magnet1_df, magnet2_df, plane)
    results = np.zeros((2*n+1, 2))
    
    for i, s in enumerate(sign):
        def fun(x): return chi2(x, foc_magnet_df, def_magnet_df, plane, kmod_input_params, s)
        fitresults = scipy.optimize.minimize(fun=fun,
                                             x0=kmod_input_params.betastar_and_waist[plane],
                                             method='nelder-mead',
                                             tol=1E-9)
        results[i, :] = fitresults.x[0], fitresults.x[1]

    beta_waist_err = get_err(results[1::2, 0]-results[0, 0])
    waist_err = get_err(results[1::2, 1]-results[0, 1])

    return results[0, 0], beta_waist_err, results[0, 1], waist_err


def get_err(diff_array):
    return np.sqrt(np.sum(np.square(diff_array)))


def analyse(magnet1_df, magnet2_df, opt, betastar_required):

    for magnet_df in (magnet1_df, magnet2_df):
        LOG.info(f'Analysing magnet {magnet_df.headers["QUADRUPOLE"]}')
        magnet_df = helper.add_tune_uncertainty(magnet_df, opt.tune_uncertainty)
        magnet_df = helper.clean_data(magnet_df, opt.no_autoclean)
        magnet_df = calc_tune(magnet_df)
        magnet_df = calc_k(magnet_df)
        magnet_df = get_av_beta(magnet_df)

    LOG.info('Simplex to determine beta waist')
    results = {plane: get_beta_waist(magnet1_df, magnet2_df, opt, plane) for plane in PLANES}

    results_df = tfs.TfsDataFrame(
        columns=['LABEL',
                 "TIME"],
        data=[np.hstack((opt.label,
                         datetime.datetime.now().strftime(formats.TIME)))])

    for plane in PLANES:
        results_df[f"{BETA}{WAIST}{plane}"] = results[plane][0]
        results_df[f"{ERR}{BETA}{WAIST}{plane}"] = results[plane][1]
        results_df[f"{WAIST}{plane}"] = results[plane][2]
        results_df[f"{ERR}{WAIST}{plane}"] = results[plane][3]

    LOG.info('Calculate betastar')
    if betastar_required:
        results_df = calc_betastar(opt, results_df, magnet1_df.headers['LSTAR'])

    LOG.info('Calculate beta at instruments')
    if opt.instruments_found:
        instrument_beta_df = calc_beta_at_instruments(opt, results_df, magnet1_df, magnet2_df)    

    return magnet1_df, magnet2_df, results_df, instrument_beta_df
