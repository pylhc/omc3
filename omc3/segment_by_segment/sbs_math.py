"""
Segment by Segment: Maths functions
-----------------------------------

This module provides mathematical helper functions, e.g. to propagate errors.
"""
from typing import Union

import numpy as np
from pandas import Series


def propagate_error_beta(errbeta0: float, erralpha0: float, beta0: float, alpha0: float,
                         dphi: Union[Series, np.array, float], beta: Union[Series, np.array, float]
                         ) -> np.array:
    """Propagates the beta-error from beta0 to beta with dphi phaseadvance.

        Args:
            errbeta0 (float): Error on the beta function at initial position
            erralpha0 (float): Error on the alpha function at initial position
            dphi (float, np.array, Series): Phase advances from initial to final positions
            beta (float, np.array, Series): Beta function at final positions
            beta0 (float): Beta function at the initial position
            alpha0 (float): Alpha function at the initial position
    """
    return np.sqrt(
        (
                beta * np.sin(4 * np.pi * dphi) * alpha0 / beta0 +
                beta * np.cos(4 * np.pi * dphi) / beta0
        ) ** 2 * errbeta0 ** 2 +
        (beta * np.sin(4 * np.pi * dphi)) ** 2 * erralpha0 ** 2
    )


def propagate_error_alfa(errbeta0: float, erralpha0: float, beta0: float, alpha0: float,
                         dphi: Union[Series, np.array, float], alpha: Union[Series, np.array, float]
                         ) -> np.array:
    """Propagates the alpha-error from alpha0 to alpha with dphi phaseadvance.

        Args:
            errbeta0 (float): Error on the beta function at initial position
            erralpha0 (float): Error on the alpha function at initial position
            beta0 (float): Beta function at the initial position
            alpha0 (float): Alpha function at the initial position
            dphi (float, np.array, Series): Phase advances from initial to final positions
            alpha (float, np.array, Series): alpha function at final positions
    """
    return np.sqrt(
        (
                (alpha * ((np.sin(4 * np.pi * dphi) * alpha0 / beta0) + (np.cos(4 * np.pi * dphi) / beta0))) -
                (np.cos(4*np.pi*dphi) * alpha0 / beta0) + (np.sin(4 * np.pi * dphi) / beta0)
        ) ** 2 * errbeta0 ** 2 +
        ((np.cos(4*np.pi*dphi)) - (alpha * np.sin(4 * np.pi * dphi))) ** 2 * erralpha0 ** 2
    )


def propagate_error_phase(errbeta0: float, erralpha0: float, beta0: float, alpha0: float,
                          dphi: Union[Series, np.array, float]
                          ) -> np.array:
    """Propagates the phase-error.

        Args:
            errbeta0 (float): Error on the beta function at initial position
            erralpha0 (float): Error on the alpha function at initial position
            beta0 (float): Beta function at the initial position
            alpha0 (float): Alpha function at the initial position
            dphi (float, np.array, Series): Phase advances from initial to final positions
    """
    return np.sqrt(
        (
                ((1 / 2. * np.cos(4*np.pi*dphi) * alpha0 / beta0) -
                 (1 / 2. * np.sin(4*np.pi*dphi) / beta0)
                 - (1 / 2. * alpha0 / beta0)
                 ) * errbeta0
        ) ** 2 +
        ((-(1/2.*np.cos(4*np.pi*dphi))+(1/2.)) * erralpha0) ** 2
    )/(2*np.pi)


def propagate_error_coupling_1001_re(f1001ab_ini, p1001_ini, phasex, phasey, f1001_std_ini, p1001_std_ini):
    return np.sqrt(
        (f1001_std_ini * np.cos(2 * np.pi * (p1001_ini - phasex + phasey)))**2 +
        (2 * np.pi * p1001_std_ini * f1001ab_ini *
         np.sin(2 * np.pi * (p1001_ini - phasex + phasey)))**2
    )


def propagate_error_coupling_1001_im(f1001ab_ini, p1001_ini, phasex, phasey, f1001_std_ini, p1001_std_ini):
    return np.sqrt(
        (f1001_std_ini * np.sin(2 * np.pi * (p1001_ini - phasex + phasey)))**2 +
        (2 * np.pi * p1001_std_ini * f1001ab_ini *
         np.cos(2 * np.pi * (p1001_ini - phasex + phasey)))**2
    )


def propagate_error_coupling_1010_re(f1010ab_ini, p1010_ini, phasex, phasey, f1010_std_ini, p1010_std_ini):
    return np.sqrt(
        (f1010_std_ini * np.cos(2 * np.pi * (p1010_ini - phasex - phasey)))**2 +
        (2 * np.pi * p1010_std_ini * f1010ab_ini *
         np.sin(2 * np.pi * (p1010_ini - phasex - phasey)))**2
    )


def propagate_error_coupling_1010_im(f1010ab_ini, p1010_ini, phasex, phasey, f1010_std_ini, p1010_std_ini):
    return np.sqrt(
        (f1010_std_ini * np.sin(2 * np.pi * (p1010_ini - phasex - phasey)))**2 +
        (2 * np.pi * p1010_std_ini * f1010ab_ini *
         np.cos(2 * np.pi * (p1010_ini - phasex - phasey)))**2
    )


def propagate_error_dispersion(std_D0, bet0, bets, dphi, alf0):
    return np.abs(
        std_D0 * np.sqrt(bets/bet0) *
        (np.cos(2*np.pi*dphi)+alf0*np.sin(2*np.pi*dphi))
    )


def weighted_average_for_SbS_elements(value1, sigma1, value2, sigma2):
    weighted_average = ((1/sigma1**2 * value1 + 1/sigma2**2 * value2) /
                        (1/sigma1**2 + 1/sigma2**2))
    uncertainty_of_average = np.sqrt(1 / (1/sigma1**2 + 1/sigma2**2))
    weighted_rms = np.sqrt(2 * (1/sigma1**2 * (value1 - weighted_average)**2 +
                                1/sigma2**2 * (value2 - weighted_average)**2) /
                           (1/sigma1**2 + 1/sigma2**2))
    final_error = np.sqrt(uncertainty_of_average**2 + weighted_rms**2)
    return weighted_average, final_error
