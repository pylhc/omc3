"""
Analysis
--------

This module contains the analysis functionality of ``kmod``.
It provides functions to calculate beta functions at different locations from K-modulation data.
"""

import datetime

import numpy as np
import scipy.optimize
import tfs
from tfs import tools as tfstools

from omc3.definitions import formats
from omc3.definitions.constants import PLANES
from omc3.kmod import helper
from omc3.kmod.constants import AVERAGE, BETA, CLEANED, ERR, PHASEADV, SEQUENCES_PATH, STAR, TUNE, WAIST, K
from omc3.model.constants import TWISS_DAT
from omc3.optics_measurements.constants import EXT, PHASE_NAME
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


def return_sign_for_err(n):
    """
    Creates an array for error calculation, of form:
    [[ 0.  0.  0.]
    [ 1.  0.  0.]
    [-1. -0. -0.]
    [ 0.  1.  0.]
    [-0. -1. -0.]
    [ 0.  0.  1.]
    [-0. -0. -1.]]
    Columns corresponds to error, i.e. first column for `dQ` etc.
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
        betastar = propagate_beta_in_drift((results_df.loc[:, f"{BETA}{WAIST}{plane}"].to_numpy() + sign[:, 0] * results_df.loc[:, f"{ERR}{BETA}{WAIST}{plane}"].to_numpy()),
                                           (results_df.loc[:, f"{WAIST}{plane}"].to_numpy() + sign[:, 1] * results_df.loc[:, f"{ERR}{WAIST}{plane}"].to_numpy()))
        betastar_err = get_err(betastar[1::2]-betastar[0])

        if kmod_input_params.no_sig_digits:
            results_df[f"{BETA}{STAR}{plane}"], results_df[f"{ERR}{BETA}{STAR}{plane}"] = (betastar[0], betastar_err)
        else:
            results_df[f"{BETA}{STAR}{plane}"], results_df[f"{ERR}{BETA}{STAR}{plane}"] = tfstools.significant_digits(betastar[0], betastar_err, return_floats=True)


    # reindex df to put betastar first
    cols = results_df.columns.tolist()
    cols = [cols[0]]+cols[-4:]+cols[1:-4]
    results_df = results_df.reindex(columns=cols)

    for plane in PLANES:
        results_df[f"{PHASEADV}{plane}"], results_df[f"{ERR}{PHASEADV}{plane}"] = phase_adv_from_kmod(
            l_star, betastar[0], betastar_err,
            results_df.loc[:, f"{WAIST}{plane}"].to_numpy(),
            results_df.loc[:, f"{ERR}{WAIST}{plane}"].to_numpy())

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
        waist = results_df.loc[:, f"{WAIST}{plane}"].to_numpy()
        if magnet1_df.headers['POLARITY'] == 1 and magnet2_df.headers['POLARITY'] == -1:
            waist = -waist
        if plane == 'Y':
            waist = -waist

        beta = propagate_beta_in_drift((results_df.loc[:, f"{BETA}{WAIST}{plane}"].to_numpy() + sign[:, 0] * results_df.loc[:, f"{ERR}{BETA}{WAIST}{plane}"].to_numpy()),
                                       ((waist - position) + sign[:, 1] * results_df.loc[:, f"{ERR}{WAIST}{plane}"].to_numpy()))
        beta_err = get_err(beta[1::2]-beta[0])
        if kmod_input_params.no_sig_digits:
            betas[i, 0], betas[i, 1] = beta[0], beta_err
        else:
            betas[i, 0], betas[i, 1] = tfstools.significant_digits(beta[0], beta_err, return_floats=True)

    return name, betas[0, 0], betas[0, 1], betas[1, 0], betas[1, 1]


def calc_beta_at_instruments(kmod_input_params, results_df, magnet1_df, magnet2_df):

    beta_instr = []

    for instrument in kmod_input_params.instruments_found:
        positions = getattr(kmod_input_params, instrument)

        for name, position in positions.items():
            beta_instr.append(calc_beta_inst(
                name, position, results_df, magnet1_df, magnet2_df, kmod_input_params))

    instrument_beta_df = tfs.TfsDataFrame(
        columns=['NAME',
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


def fit_approx(x, beta_av):
    dQ = beta_av*x[0]/(4*np.pi)
    return dQ


def average_beta_from_Tune(Q, TdQ, length, Dk):
    """Calculates average beta function in quadrupole from tune change ``TdQ`` and ``delta K``."""

    beta_av = 2 * (1 / np.tan(2 * np.pi * Q) *
              (1 - np.cos(2 * np.pi * TdQ)) + np.sin(2 * np.pi * TdQ)) / (length * Dk)
    return abs(beta_av)


def average_beta_focussing_quadrupole(b, w, L, K, Lstar):

    beta0 = b + ((Lstar - w) ** 2 / b)
    alpha0 = -(Lstar - w) / b
    average_beta = (beta0/2.) * (1 + ((np.sin(2 * np.sqrt(abs(K)) * L)) / (2 * np.sqrt(abs(K)) * L))) \
                   - alpha0 * ((np.sin(np.sqrt(abs(K)) * L)**2) / (abs(K) * L)) \
                   + (1/(2*abs(K))) * ((1 + alpha0**2) / beta0) * \
                   (1 - ((np.sin(2 * np.sqrt(abs(K)) * L)) / (2 * np.sqrt(abs(K)) * L)))

    return average_beta


def average_beta_defocussing_quadrupole(b, w, L, K, Lstar):
    beta0 = b + ((Lstar - w) ** 2 / b)
    alpha0 = -(Lstar - w) / b
    average_beta = (beta0/2.) * (1 + ((np.sinh(2 * np.sqrt(abs(K)) * L)) / (2 * np.sqrt(abs(K)) * L))) \
                   - alpha0 * ((np.sinh(np.sqrt(abs(K)) * L)**2) / (abs(K) * L)) \
                   + (1/(2*abs(K))) * ((1 + alpha0**2) / beta0) * \
                   (((np.sinh(2 * np.sqrt(abs(K)) * L)) / (2 * np.sqrt(abs(K)) * L)) - 1)

    return average_beta


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
    import warnings

    from scipy.optimize import OptimizeWarning

    if not use_approx:
        fun = fit_prec
    elif use_approx:
        fun = fit_approx
    
    sigma = magnet_df.where(magnet_df[f"{CLEANED}{plane}"])[f"{ERR}{TUNE}{plane}"].dropna()
    if not np.any(sigma):
        sigma = 1.E-22 * np.ones(len(sigma))

    # We filter out the "Covariance of the parameters could not be estimated" warning
    # If the warning is issued we relay it as a logged message, which allows us to
    # avoid polluting the stderr and allows the user to not see it depending on log level
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("ignore", category=OptimizeWarning)
        av_beta, av_beta_err = scipy.optimize.curve_fit(
            fun,
            xdata=return_fit_input(magnet_df, plane),
            ydata=magnet_df.where(magnet_df[f"{CLEANED}{plane}"])[
                f"{TUNE}{plane}"].dropna() - magnet_df.headers[f"{TUNE}{plane}"],
            sigma=sigma,
            absolute_sigma=True,
            p0=1
        )

        # We log any captured warning at warning level
        for warning in records:
            LOG.warning(f"Curve fit warning: {warning.message}")

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

      
def get_BPM(kmod_input_params):

    # listing the BPMs of the last quadrupole BPMs
    BPM_dict = {
        "IP1": ["BPMSW.1L1", "BPMSW.1R1"],
        "IP2": ["BPMSW.1L2", "BPMSW.1R2"],
        "IP3": ["BPMW.4L3", "BPMW.4R3"],
        "IP4": ["BPMWA.A5L4", "BPMWA.A5R4"],
        "IP5": ["BPMSW.1L5", "BPMSW.1R5"],
        "IP6": ["BPMSE.4L6", "BPMSA.4R6"],
        "IP7": ["BPMW.4L7", "BPMW.4R7"],
        "IP8": ["BPMSW.1L8", "BPMSW.1R8"],
    }
    if kmod_input_params.interaction_point:
        return [f"{bpm}.B{kmod_input_params.beam:d}"
                for bpm in BPM_dict[kmod_input_params.interaction_point.upper()]]
    if kmod_input_params.circuits:
        return [f"{bpm}.B{kmod_input_params.beam:d}"
                for bpm in BPM_dict[f"IP{kmod_input_params.circuits[0][-3]}"]]
    raise AttributeError("Should not have happened, was checked in analyse_kmod")


def get_BPM_distance(kmod_input_params, BPML, BPMR):
    twiss_df = tfs.read(
        SEQUENCES_PATH / f"twiss_lhcb{kmod_input_params.beam:d}.dat", index='NAME'
    )
    return np.abs(twiss_df.loc[BPMR, 'S'] - twiss_df.loc[BPML, 'S']) / 2


def get_phase_from_model(kmod_input_params, plane):
    """Get the phase from twiss model."""
    twiss_df = tfs.read(kmod_input_params.model_dir / TWISS_DAT, index='NAME')
    BPML, BPMR = get_BPM(kmod_input_params)[0], get_BPM(kmod_input_params)[1]
    phase_adv_model = abs(twiss_df.loc[BPMR, f'MU{plane}'] - twiss_df.loc[BPML, f'MU{plane}'])
    phase_adv_err = 0.5e-3  # this number is given by Andrea's estimations

    return phase_adv_model, phase_adv_err


def get_phase_from_measurement(kmod_input_params, plane):
    phase_df = tfs.read(
        kmod_input_params.measurement_dir / f'{PHASE_NAME}{plane.lower()}{EXT}', index='NAME'
    )
    bpms_lr = get_BPM(kmod_input_params)
    for bpm in bpms_lr:
        if bpm not in phase_df.index.to_numpy():
            raise ValueError(f"BPM {bpm} not found in the measurement")
    # both BPMs are in measurement and there should be no other BPM in between
    return (phase_df.loc[bpms_lr[0], f'PHASE{plane}'],
            phase_df.loc[bpms_lr[0], f'{ERR}PHASE{plane}'])


def phase_constraint(kmod_input_params, plane):
    if kmod_input_params.measurement_dir:
        return get_phase_from_measurement(kmod_input_params, plane)
    # model is taken (if exists) in case no measurement data is provided
    if kmod_input_params.model_dir:
        return get_phase_from_model(kmod_input_params, plane)
    return [1.0, 1.0]


def chi2(x, foc_magnet_df, def_magnet_df, plane, kmod_input_params, sign, BPM_distance, phase_adv_constraint):

    b = x[0]
    w = x[1]

    if kmod_input_params.interaction_point:

        phase_adv = phase_adv_from_kmod(BPM_distance, b, 0.0, w, 0.0)[0]
        weight = kmod_input_params.phase_weight
    else:
        phase_adv = 0.0
        weight = 0

    c2 = (1-weight)*(((average_beta_focussing_quadrupole(b, w, foc_magnet_df.headers['LENGTH'] +
        sign[0] * kmod_input_params.errorL, foc_magnet_df.headers[K] +
        sign[1] * kmod_input_params.errorK * foc_magnet_df.headers[K],
        foc_magnet_df.headers['LSTAR'] +
        sign[2] * kmod_input_params.misalignment) -
        foc_magnet_df.headers[f"{AVERAGE}{BETA}{plane}"] +
        sign[3] * foc_magnet_df.headers[f"{ERR}{AVERAGE}{BETA}{plane}"]) / (def_magnet_df.headers[f"{AVERAGE}{BETA}{plane}"] + foc_magnet_df.headers[f"{AVERAGE}{BETA}{plane}"]) / 2.0) ** 2 +
        ((average_beta_defocussing_quadrupole(b, -w, def_magnet_df.headers['LENGTH'] +
        sign[4] * kmod_input_params.errorL, def_magnet_df.headers[K] +
        sign[5] * kmod_input_params.errorK * def_magnet_df.headers[K],
        def_magnet_df.headers['LSTAR'] +
        sign[6] * kmod_input_params.misalignment) -
        def_magnet_df.headers[f"{AVERAGE}{BETA}{plane}"] +
        sign[7] * def_magnet_df.headers[f"{ERR}{AVERAGE}{BETA}{plane}"])/(foc_magnet_df.headers[f"{AVERAGE}{BETA}{plane}"] + def_magnet_df.headers[f"{AVERAGE}{BETA}{plane}"])/2.0) ** 2) + \
        weight*(((phase_adv - (phase_adv_constraint[0]+sign[8]*phase_adv_constraint[1]))/phase_adv_constraint[0])**2)

    return c2


def get_beta_waist(magnet1_df, magnet2_df, kmod_input_params, plane):

    n = 9
    sign = return_sign_for_err(n)
    foc_magnet_df, def_magnet_df = return_df(magnet1_df, magnet2_df, plane)
    results = np.zeros((2*n+1, 2))
    BPML, BPMR = get_BPM(kmod_input_params)
    BPM_distance = get_BPM_distance(kmod_input_params, BPML, BPMR)
    phase_adv_constraint = phase_constraint(kmod_input_params, plane)
    for i, s in enumerate(sign):

        def fun(x): return chi2(x, foc_magnet_df, def_magnet_df, plane, kmod_input_params, s, BPM_distance, phase_adv_constraint)
        fitresults = scipy.optimize.minimize(fun=fun,
                                             x0=kmod_input_params.betastar_and_waist[plane],
                                             method='nelder-mead',
                                             tol=1E-22)

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
