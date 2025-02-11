"""
Segment by Segment: Maths functions
-----------------------------------

This module provides mathematical helper functions, e.g. to propagate errors.
"""
from __future__ import annotations

import numpy as np
from pandas import Series
from typing import TYPE_CHECKING

from omc3.segment_by_segment.definitions import PropagableBoundaryConditions

if TYPE_CHECKING:
    NumericOrArray = float | np.array | Series


def propagate_error_phase(dphi: NumericOrArray, init: PropagableBoundaryConditions) -> NumericOrArray:
    """Propagates the phase-error.
       See Eq. (2) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ .
       This implementation has a minus-sign instead of the first plus-sign as in the reference,
       this seems to be the correct implementation, 
       as found e.g. in https://github.com/pylhc/MESS/tree/master/FODO_Test_Lattice/Phase_Error_Propagation

        Args:
            dphi (float, np.array, Series): Phase advances from initial to final positions
            init (PropagableBoundaryConditions): Initial conditions for alpha and beta and their uncertainties.
    """
    alpha0, erralpha0 = init.alpha.as_tuple()
    beta0, errbeta0 = init.beta.as_tuple()
    
    sin2phi = np.sin(4 * np.pi * dphi)
    cos2phi = np.cos(4 * np.pi * dphi)

    res = np.sqrt(
        (0.5 * (((cos2phi - 1) * alpha0) - sin2phi) * errbeta0/beta0) ** 2 +
        (0.5 * (cos2phi - 1) * erralpha0) ** 2
    ) / (2 * np.pi)
    return res


def propagate_error_beta(beta: NumericOrArray, dphi: NumericOrArray, init: PropagableBoundaryConditions) -> NumericOrArray:
    """Propagates the beta-error from beta0 to beta with dphi phaseadvance.
       See Eq. (3) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ .

        Args:
            beta (float, np.array, Series): Beta function at final positions
            dphi (float, np.array, Series): Phase advances from initial to final positions
            init (PropagableBoundaryConditions): Initial conditions for alpha and beta and their uncertainties.
    """
    alpha0, erralpha0 = init.alpha.as_tuple()
    beta0, errbeta0 = init.beta.as_tuple()
    
    sin2phi = np.sin(4 * np.pi * dphi)
    cos2phi = np.cos(4 * np.pi * dphi)

    res = np.sqrt(
        (beta * (sin2phi * alpha0 + cos2phi) * errbeta0 / beta0) ** 2 +
        (beta * sin2phi * erralpha0) ** 2
    )
    return res


def propagate_error_alpha(alpha: NumericOrArray, dphi: NumericOrArray, init: PropagableBoundaryConditions) -> NumericOrArray:
    """Propagates the alpha-error from alpha0 to alpha with dphi phaseadvance.
       See Eq. (4) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ .

       Args:
           alpha (float, np.array, Series): alpha function at final positions
           dphi (float, np.array, Series): Phase advances from initial to final positions
           init (PropagableBoundaryConditions): Initial conditions for alpha and beta and their uncertainties.
    """
    alpha0, erralpha0 = init.alpha.as_tuple()
    beta0, errbeta0 = init.beta.as_tuple()

    sin2phi = np.sin(4 * np.pi * dphi)
    cos2phi = np.cos(4 * np.pi * dphi)

    res = np.sqrt(
        (
            (((sin2phi * alpha0) + cos2phi) * alpha -
              (cos2phi * alpha0) + sin2phi
            ) * errbeta0/beta0
        ) ** 2 +
        ((cos2phi - (alpha * sin2phi)) * erralpha0) ** 2
    )
    return res


def propagate_error_coupling_1001_re(dphix: NumericOrArray, dphiy: NumericOrArray, init: PropagableBoundaryConditions) -> NumericOrArray:
    """Propagates the error on the real part of f1001 through dphix and dphiy phaseadvance,
       based on the initial amplitude and phase error of f1001.
       See Eq. (5) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ .

       Args:
           dphix (float, np.array, Series): Phase advances in x from initial to final positions
           dphiy (float, np.array, Series): Phase advances in y from initial to final positions
           init (PropagableBoundaryConditions): Initial conditions for f1001 amplitude and phase and their uncertainties.
    """
    
    amp0, erramp0 = init.f1001_amplitude.as_tuple()
    phase0, errphase0 = init.f1001_phase.as_tuple()

    errphase0 = 2 * np.pi * errphase0

    phase = 2 * np.pi * (phase0 - dphix + dphiy)

    res = np.sqrt(
        (erramp0 * np.cos(phase))**2 +
        (errphase0 * amp0 * np.sin(phase))**2
    )
    return res


def propagate_error_coupling_1001_im(dphix: NumericOrArray, dphiy: NumericOrArray, init: PropagableBoundaryConditions) -> NumericOrArray:
    """Propagates the error on the imagary part of f1001 through dphix and dphiy phaseadvance,
       based on the initial amplitude and phase error of f1001.
       See Eq. (6) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ .

       Args:
           dphix (float, np.array, Series): Phase advances in x from initial to final positions
           dphiy (float, np.array, Series): Phase advances in y from initial to final positions
           init (PropagableBoundaryConditions): Initial conditions for f1001 amplitude and phase and their uncertainties.
    """
    amp0, erramp0 = init.f1001_amplitude.as_tuple()
    phase0, errphase0 = init.f1001_phase.as_tuple()
    errphase0 = 2 * np.pi * errphase0

    phase = 2 * np.pi * (phase0 - dphix + dphiy)

    res = np.sqrt(
        (erramp0 * np.sin(phase))**2 +
        (errphase0 * amp0 * np.cos(phase))**2
    )
    return res


def propagate_error_coupling_1010_re(dphix: NumericOrArray, dphiy: NumericOrArray, init: PropagableBoundaryConditions) -> NumericOrArray:
    """Propagates the error on the real part of f1010 through dphix and dphiy phaseadvance,
       based on the initial amplitude and phase error of f1010.
       See Eq. (7) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ , 
       yet the phases dphix and dphiy are subtracted from the initial rdt phase.

       Args:
           dphix (float, np.array, Series): Phase advances in x from initial to final positions
           dphiy (float, np.array, Series): Phase advances in y from initial to final positions
           init (PropagableBoundaryConditions): Initial conditions for f1010 amplitude and phase and their uncertainties.
    """
    amp0, erramp0 = init.f1010_amplitude.as_tuple()
    phase0, errphase0 = init.f1010_phase.as_tuple()
    errphase0 = 2 * np.pi * errphase0

    phase = 2 * np.pi * (phase0 - dphix - dphiy)

    res = np.sqrt(
        (erramp0 * np.cos(phase))**2 +
        (errphase0 * amp0 * np.sin(phase))**2
    )
    return res


def propagate_error_coupling_1010_im(dphix: NumericOrArray, dphiy: NumericOrArray, init: PropagableBoundaryConditions) -> NumericOrArray:
    """Propagates the error on the imaginary part of f1010 through dphix and dphiy phaseadvance,
       based on the initial amplitude and phase error of f1010.
       See Eq. (7) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ , 
       yet the phases dphix and dphiy are subtracted from the initial rdt phase.

       Args:
           dphix (float, np.array, Series): Phase advances in x from initial to final positions
           dphiy (float, np.array, Series): Phase advances in y from initial to final positions
           init (PropagableBoundaryConditions): Initial conditions for f1010 amplitude and phase and their uncertainties.
    """
    amp0, erramp0 = init.f1010_amplitude.as_tuple()
    phase0, errphase0 = init.f1010_phase.as_tuple()
    errphase0 = 2 * np.pi * errphase0

    phase = 2 * np.pi * (phase0 - dphix - dphiy)

    res = np.sqrt(
        (erramp0 * np.sin(phase))**2 +
        (errphase0 * amp0 * np.cos(phase))**2
    )
    return res


def propagate_error_dispersion(beta: NumericOrArray, dphi: NumericOrArray, init: PropagableBoundaryConditions) -> NumericOrArray:
    """Propagates the dispersion error with dphi phaseadvance.

    Args:
        beta (float, np.array, Series): Beta function at final positions
        dphi (float, np.array, Series): Phase advances from initial to final positions
        init (PropagableBoundaryConditions): Initial conditions for alpha and beta and the dispersion uncertainty.
    """
    _, errdispersion0 = init.dispersion.as_tuple()
    beta0, _ = init.beta.as_tuple()
    alpha0, _ = init.alpha.as_tuple()

    res = np.abs(
        errdispersion0 * np.sqrt(beta / beta0) * 
        (np.cos(2 * np.pi * dphi) + alpha0 * np.sin(2 * np.pi * dphi))
    )
    return res


def weighted_average_for_SbS_elements(value1, sigma1, value2, sigma2):
    weighted_average = ((1/sigma1**2 * value1 + 1/sigma2**2 * value2) /
                        (1/sigma1**2 + 1/sigma2**2))
    uncertainty_of_average = np.sqrt(1 / (1/sigma1**2 + 1/sigma2**2))
    weighted_rms = np.sqrt(2 * (1/sigma1**2 * (value1 - weighted_average)**2 +
                                1/sigma2**2 * (value2 - weighted_average)**2) /
                           (1/sigma1**2 + 1/sigma2**2))
    final_error = np.sqrt(uncertainty_of_average**2 + weighted_rms**2)
    return weighted_average, final_error

    
def phase_diff(phase_a: NumericOrArray, phase_b: NumericOrArray) -> NumericOrArray:
    """ Returns the phase difference between phase_a and phase_b, mapped to [-0.5, 0.5]. """
    phase_diff = (phase_a - phase_b) % 1
    return np.where(phase_diff > 0.5, phase_diff - 1, phase_diff)