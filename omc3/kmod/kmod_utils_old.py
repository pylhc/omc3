import numpy as np
import scipy.optimize
import argparse


def equations(x, cminus, q1, q2):
    qx, qy = x

    return (0.5 * (qx + qy + np.sqrt((qx - qy) ** 2 + cminus ** 2)) - q1,
            0.5 * (qx + qy - np.sqrt((qx - qy) ** 2 + cminus ** 2)) - q2)


def tune_error_from_coupling(cminus, Q1, Q2, dQ):

    if Q1 < Q2:
        q1 = Q2
        q2 = Q1
    else:
        q1 = Q1
        q2 = Q2
        
    q1 -= dQ
    q2 += dQ    
    
    if q1 < q2:
        qtemp = q2
        q2 = q1
        q1 = qtemp

    initial_conditions = np.array((q1, q2))
    qx1, qy1 = scipy.optimize.fsolve(lambda x: equations(x, cminus, q1, q2), initial_conditions)

    return ((np.abs(q1-q2) - np.abs(qx1 - qy1))/(np.abs(qx1-qy1)))


def func_beta_foc_av(b, L, w, KL, K):
    """calc beta av in foc magnet
    b ... beta at waist
    L ...  L*
    w ... waist shift
    KL ... Sqrt of Quadrupole Gradient times Quadrupole length
    K ... Sqrt of Quadrupole Gradient
    """
    beta0 = b + ((L - w) ** 2 / (b))
    alpha0 = (L - w) / b
    sin2KL = ((np.sin(2 * KL)) / (2 * KL))
    beta_foc_av = 0.5 * beta0 * (1 + sin2KL) + alpha0 * ((np.sin(KL) ** 2) / (KL * K)) + (1 - sin2KL) / (
        2 * b * K ** 2)
    return beta_foc_av


def func_beta_def_av(b, L, w, KL, K):
    """calc beta av in def magnet
    b ... beta at waist
    L ...  L*
    w ... waist shift
    KL ... Sqrt of Quadrupole Gradient times Quadrupole length
    K ... Sqrt of Quadrupole Gradient
    """
    beta0 = b + ((L - w) ** 2 / (b))
    alpha0 = (L - w) / b
    sinh2KL = ((np.sinh(2 * KL)) / (2 * KL))
    beta_def_av = 0.5 * beta0 * (1 + sinh2KL) + alpha0 * ((np.sinh(KL) ** 2) / (KL * K)) + (sinh2KL - 1) / (
        2 * b * K ** 2)
    return beta_def_av


def chi2(d, L_star_left, L_star_right, KL_foc, K_foc, KL_def, K_def, betaavfocquad, betaavdefquad):
    """error function for simplex algorithm"""
    b = d[0]
    w = d[1]
    c2 = (func_beta_foc_av(b, L_star_left, w, KL_foc, K_foc) - betaavfocquad) ** 2 + (func_beta_def_av(b, L_star_right, -w, KL_def, K_def) - betaavdefquad) ** 2
    return c2


def beta_from_Tune(Q, TdQ, l, Dk):
    """Calculates average beta function in quadrupole from Tunechange TdQ and delta K """
    beta_av = 2 * (1 / np.tan(2 * np.pi * Q) * (1 - np.cos(2 * np.pi * TdQ)) + np.sin(2 * np.pi * TdQ)) / (
        l * Dk)
    return np.abs(beta_av)


def simplex(Q_foc, Q_def, dq_foc, dq_def, l_foc, l_def, k_foc, k_def, dk_foc, dk_def, L_star_left, L_star_right, guess):
    beta_av_foc = beta_from_Tune(Q_foc, dq_foc, l_foc, dk_foc)
    beta_av_def = beta_from_Tune(np.abs(Q_def), np.abs(dq_def), np.abs(l_def), np.abs(dk_def))

    fun = lambda x: chi2(x, L_star_left, L_star_right, np.sqrt(np.abs(k_foc)) * l_foc, np.sqrt(np.abs(k_foc)), np.sqrt(np.abs(k_def)) * l_def, np.sqrt(np.abs(k_def)), beta_av_foc,
                         beta_av_def)
    res = scipy.optimize.minimize(fun, guess, method='nelder-mead', tol=0.00000001)

    return res.x[0], res.x[1], beta_av_foc, beta_av_def


def analysis(Q1_foc, Q1_def, Q2, L_star, m, k_foc, dk_foc, l_foc, k_def, dk_def, l_def, dq_foc, edq_foc, dq_def, edq_def, ek_foc, ek_def, cminus, beta_star_guess, waist_guess, label, log, logfile):

    guess = [beta_star_guess, waist_guess]


    DQs = np.zeros([17, 6])

    DQs[0] = dq_foc, dq_def, k_foc, k_def, L_star, L_star
    DQs[1] = dq_foc + edq_foc, dq_def, k_foc, k_def, L_star, L_star
    DQs[2] = dq_foc - edq_foc,  dq_def, k_foc, k_def, L_star, L_star
    DQs[3] = dq_foc, dq_def + edq_def, k_foc, k_def, L_star, L_star
    DQs[4] = dq_foc, dq_def - edq_def, k_foc, k_def, L_star, L_star

    DQs[5] = dq_foc, dq_def, k_foc + k_foc*ek_foc, k_def, L_star, L_star
    DQs[6] = dq_foc, dq_def, k_foc - k_foc*ek_foc, k_def, L_star, L_star
    DQs[7] = dq_foc, dq_def, k_foc, k_def + k_def*ek_def, L_star, L_star
    DQs[8] = dq_foc, dq_def, k_foc, k_def - k_def*ek_def, L_star, L_star

    DQs[9] = dq_foc, dq_def, k_foc, k_def, L_star + m, L_star
    DQs[10] = dq_foc, dq_def, k_foc, k_def, L_star - m, L_star
    DQs[11] = dq_foc, dq_def, k_foc, k_def, L_star, L_star + m
    DQs[12] = dq_foc, dq_def, k_foc, k_def, L_star, L_star - m

    DQs[13] = dq_foc + dq_foc * tune_error_from_coupling(cminus, Q1_foc, Q2, np.abs(dq_foc)), dq_def, k_foc, k_def, L_star, L_star
    DQs[14] = dq_foc - dq_foc * tune_error_from_coupling(cminus, Q1_foc, Q2, np.abs(dq_foc)), dq_def, k_foc, k_def, L_star, L_star
    DQs[15] = dq_foc, dq_def + dq_def * tune_error_from_coupling(cminus, Q1_foc, Q2, np.abs(dq_def)), k_foc, k_def, L_star, L_star 
    DQs[16] = dq_foc, dq_def - dq_def * tune_error_from_coupling(cminus, Q1_foc, Q2, np.abs(dq_def)), k_foc, k_def, L_star, L_star 

    resb = np.zeros(17)
    resw = np.zeros(17)

    resbavf = np.zeros(17)
    resbavd = np.zeros(17)

    for i in range(17):
        resb[i], resw[i], resbavf[i], resbavd[i] = simplex(Q1_foc, Q1_def, DQs[i, 0], DQs[i, 1], l_foc, l_def, DQs[i, 2], DQs[i, 3],
                                                           dk_foc, dk_def, DQs[i, 4], DQs[i, 5], guess)

    stdb = np.sqrt(max(np.abs(resb[1] - resb[0]), np.abs(resb[2] - resb[0])) ** 2 +
                     max(np.abs(resb[3] - resb[0]), np.abs(resb[4] - resb[0])) ** 2 +
                     max(np.abs(resb[5] - resb[0]), np.abs(resb[6] - resb[0])) ** 2 +
                     max(np.abs(resb[7] - resb[0]), np.abs(resb[8] - resb[0])) ** 2 +
                     max(np.abs(resb[9] - resb[0]), np.abs(resb[10] - resb[0])) ** 2 +
                     max(np.abs(resb[11] - resb[0]), np.abs(resb[12] - resb[0])) ** 2 +
                     max(np.abs(resb[13] - resb[0]), np.abs(resb[14] - resb[0])) ** 2 +
                     max(np.abs(resb[15] - resb[0]), np.abs(resb[16] - resb[0])) ** 2 )

    stdw = np.sqrt(max(np.abs(resw[1] - resw[0]), np.abs(resw[2] - resw[0])) ** 2 +
                     max(np.abs(resw[3] - resw[0]), np.abs(resw[4] - resw[0])) ** 2 +
                     max(np.abs(resw[5] - resw[0]), np.abs(resw[6] - resw[0])) ** 2 +
                     max(np.abs(resw[7] - resw[0]), np.abs(resw[8] - resw[0])) ** 2 +
                     max(np.abs(resw[9] - resw[0]), np.abs(resw[10] - resw[0])) ** 2 +
                     max(np.abs(resw[11] - resw[0]), np.abs(resw[12] - resw[0])) ** 2 +
                     max(np.abs(resw[13] - resw[0]), np.abs(resw[14] - resw[0])) ** 2 +
                     max(np.abs(resw[15] - resw[0]), np.abs(resw[16] - resw[0])) ** 2 )

    stdbavf = np.sqrt(max(np.abs(resbavf[1] - resbavf[0]), np.abs(resbavf[2] - resbavf[0])) ** 2 +
                        max(np.abs(resbavf[3] - resbavf[0]), np.abs(resbavf[4] - resbavf[0])) ** 2 +
                        max(np.abs(resbavf[5] - resbavf[0]), np.abs(resbavf[6] - resbavf[0])) ** 2 +
                        max(np.abs(resbavf[7] - resbavf[0]), np.abs(resbavf[8] - resbavf[0])) ** 2 +
                        max(np.abs(resbavf[9] - resbavf[0]), np.abs(resbavf[10] - resbavf[0])) ** 2 +
                        max(np.abs(resbavf[11] - resbavf[0]), np.abs(resbavf[12] - resbavf[0])) ** 2 +
                        max(np.abs(resbavf[13] - resbavf[0]), np.abs(resbavf[14] - resbavf[0])) ** 2 +
                        max(np.abs(resbavf[15] - resbavf[0]), np.abs(resbavf[16] - resbavf[0])) ** 2 )

    stdbavd = np.sqrt(max(np.abs(resbavd[1] - resbavd[0]), np.abs(resbavd[2] - resbavd[0])) ** 2 +
                        max(np.abs(resbavd[3] - resbavd[0]), np.abs(resbavd[4] - resbavd[0])) ** 2 +
                        max(np.abs(resbavd[5] - resbavd[0]), np.abs(resbavd[6] - resbavd[0])) ** 2 +
                        max(np.abs(resbavd[7] - resbavd[0]), np.abs(resbavd[8] - resbavd[0])) ** 2 +
                        max(np.abs(resbavd[9] - resbavd[0]), np.abs(resbavd[10] - resbavd[0])) ** 2 +
                        max(np.abs(resbavd[11] - resbavd[0]), np.abs(resbavd[12] - resbavd[0])) ** 2 +
                        max(np.abs(resbavd[13] - resbavd[0]), np.abs(resbavd[14] - resbavd[0])) ** 2 +
                        max(np.abs(resbavd[15] - resbavd[0]), np.abs(resbavd[16] - resbavd[0])) ** 2 )

    if log == True:
        logfile.write('Label: %s \n' %label)
        logfile.write('L_star +/- misalignment : %s +/- %s, Length focussing/defocussing magnet: %s / %s  \n' % (L_star, m, l_foc, l_def))
        logfile.write('Tune: %s, dQ focussing magnet: %s +/- %s \n' % (Q1_foc, dq_foc, edq_foc))
        logfile.write('Tune: %s, dQ defocussing magnet: %s +/- %s \n' % (Q1_def, dq_def, edq_def))
        logfile.write('K focussing magnet: %s +/- %s,  K defocussing magnet: %s +/- %s, dK focussing / defocussing magnet: %s / %s \n' % (k_foc, ek_foc, k_def, ek_def, dk_foc, dk_def))
        logfile.write('C- : %s, Error from coupling in %% focussing/defocussing: %s / %s \n' %( cminus, tune_error_from_coupling(cminus, Q1_foc, Q2, np.abs(dq_foc)) * 100 , tune_error_from_coupling(cminus, Q1_foc, Q2, np.abs(dq_def)) * 100 ))
        logfile.write('Betastar guess: %s, Waistshift guess: %s \n' %(beta_star_guess, waist_guess))
        logfile.write('dQx/dK: %s, Error: %s \n' %(dq_foc/dk_foc, edq_foc/dk_foc))
        logfile.write('dQy/dK: %s, Error: %s \n' %(dq_def/dk_def, edq_def/dk_def))
        logfile.write('Average Beta focussing Quad: %s +/- %s \n' %(resbavf[0], stdbavf))
        logfile.write('Average Beta defocussing Quad: %s +/- %s \n' %(resbavd[0], stdbavd))
        logfile.write('\n')

    return label, resb[0], stdb, resw[0], stdw, resbavf[0], stdbavf, resbavd[0], stdbavd


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


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--Tunes',
                        help='Tune QX QY, separated by comma, default 0.31,0.32',
                        action='store', type=str, dest='tune', default='0.31,0.32')
    parser.add_argument('--Lstar',
                        help='L star in m, default= 22.965',
                        action='store', type=str, dest='lstar', default=22.965)
    parser.add_argument('--misalignment',
                        help='misalignment of the modulated quadrupoles in m',
                        action='store', type=float, dest='misalign', default=0)
    parser.add_argument('--K1',
                        help='K of the quadrupole, separated by comma',
                        action='store', type=str, dest='k1', default='0,0')
    parser.add_argument('--DeltaK',
                        help='DeltaK used during modulation, separated by comma',
                        action='store', type=str, dest='dk', default='0,0')
    parser.add_argument('--errorK',
                        help='error in K of the modulated quadrupoles, unit m^-2',
                        action='store', type=float, dest='ek', default=0)
    parser.add_argument('--quadlength',
                        help='length of the quadrupoles, separated by comma',
                        action='store', type=str, dest='l', default='0,0')

    parser.add_argument('--tuneshift',
                        help='tuneshifts from focussing and defocusing quadrupole, separated by comma',
                        action='store', type=str, dest='dq', default='0,0')

    parser.add_argument('--tuneuncertainty',
                        help='tune measurement uncertainty',
                        action='store', type=float, dest='edq', default=2.5e-5)

    parser.add_argument('--cminus',
                        help='Coupling C-',
                        action='store', type=float, dest='cmin', default=0)

    parser.add_argument('--betastar',
                        help='Guess for beta star',
                        action='store', type=float, dest='bstar', default=0)

    parser.add_argument('--waist',
                        help='waistshift',
                        action='store', type=float, dest='waist', default=0)

    parser.add_argument('--label',
                        help='measurement label',
                        action='store', type=float, dest='label', default='')

    options = parser.parse_args()

    return options


if __name__ == '__main__':
    options = parse_args()

    Q1, Q2 = options.tune.split(",")
    k_foc, k_def = options.k1.split(",")
    dk_foc, dk_def = options.dk.split(",")
    ek_foc, ek_def = options.ek.split(",")

    l_foc, l_def = options.l.split(",")
    dq_foc, dq_def = options.dq.split(",")
    edq_foc, edq_def = options.edq.split(",")

    print(analysis(Q1, Q2, options.lstar, options.misalign, k_foc, dk_foc, l_foc, k_def, dk_def, l_def, dq_foc, edq_foc, dq_def, edq_def, ek_foc, ek_def, options.cmin, options.bstar, options.waist, options.label, False, None))
