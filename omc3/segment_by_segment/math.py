"""
Segment by Segment: Maths functions
-----------------------------------

This module provides mathematical helper functions, e.g. to propagate errors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omc3.segment_by_segment.definitions import PropagableBoundaryConditions
    from numpy.typing import ArrayLike


def propagate_error_phase(dphi: ArrayLike, init: PropagableBoundaryConditions) -> ArrayLike:
    """Propagates the phase-error.
       See Eq. (2) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ .
       This implementation has a minus-sign instead of the first plus-sign as in the reference,
       this seems to be the correct implementation, 
       as found e.g. in https://github.com/pylhc/MESS/tree/master/FODO_Test_Lattice/Phase_Error_Propagation

        Args:
            dphi (ArrayLike): Phase advances from the initial position to the observation point(s), in units of 2pi radians.
            init (PropagableBoundaryConditions): Initial conditions at the start of the segment for alpha and beta and their uncertainties.
    """
    alpha0, erralpha0 = init.alpha.as_tuple()
    beta0, errbeta0 = init.beta.as_tuple()
    
    sin2phi = np.sin(4 * np.pi * dphi)
    cos2phi = np.cos(4 * np.pi * dphi)

    return np.sqrt(
        (0.5 * (((cos2phi - 1) * alpha0) - sin2phi) * errbeta0/beta0) ** 2 +
        (0.5 * (cos2phi - 1) * erralpha0) ** 2
    ) / (2 * np.pi)


def propagate_error_beta(beta: ArrayLike, dphi: ArrayLike, init: PropagableBoundaryConditions) -> ArrayLike:
    """Propagates the beta-error from beta0 to beta with dphi phase-advance.
       See Eq. (3) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ .

        Args:
            beta (ArrayLike): Beta function at the observation point(s).
            dphi (ArrayLike): Phase advances from the initial position to the observation point(s), in units of 2pi radians.
            init (PropagableBoundaryConditions): Initial conditions at the start of the segment for alpha and beta and their uncertainties.
    """
    alpha0, erralpha0 = init.alpha.as_tuple()
    beta0, errbeta0 = init.beta.as_tuple()
    
    sin2phi = np.sin(4 * np.pi * dphi)
    cos2phi = np.cos(4 * np.pi * dphi)

    return np.sqrt(
        (beta * (sin2phi * alpha0 + cos2phi) * errbeta0 / beta0) ** 2 +
        (beta * sin2phi * erralpha0) ** 2
    )


def propagate_error_alpha(alpha: ArrayLike, dphi: ArrayLike, init: PropagableBoundaryConditions) -> ArrayLike:
    """Propagates the alpha-error from alpha0 to alpha with dphi phaseadvance.
       See Eq. (4) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ .

       Args:
           alpha (ArrayLike): Alpha function at the observation point(s).
           dphi (ArrayLike): Phase advances from the initial position to the observation point(s), in units of 2pi radians.
           init (PropagableBoundaryConditions): Initial conditions for alpha and beta and their uncertainties.
    """
    alpha0, erralpha0 = init.alpha.as_tuple()
    beta0, errbeta0 = init.beta.as_tuple()

    sin2phi = np.sin(4 * np.pi * dphi)
    cos2phi = np.cos(4 * np.pi * dphi)

    return np.sqrt(
        (
            (((sin2phi * alpha0) + cos2phi) * alpha -
              (cos2phi * alpha0) + sin2phi
            ) * errbeta0/beta0
        ) ** 2 +
        ((cos2phi - (alpha * sin2phi)) * erralpha0) ** 2
    )


def propagate_error_coupling_1001_re(dphix: ArrayLike, dphiy: ArrayLike, init: PropagableBoundaryConditions) -> ArrayLike:
    """Propagates the error on the real part of f1001 through dphix and dphiy phase-advance,
       based on the initial amplitude and phase error of f1001.
       See Eq. (5) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ .

       Args:
           dphix (ArrayLike): Phase advances in x from initial position to observation point(s).
           dphiy (ArrayLike): Phase advances in y from initial position to observation point(s).
           init (PropagableBoundaryConditions): Initial conditions for f1001 amplitude and phase and their uncertainties.
    """
    amp0, erramp0 = init.f1001_amplitude.as_tuple()
    phase0, errphase0 = init.f1001_phase.as_tuple()

    errphase0 = 2 * np.pi * errphase0

    phase = 2 * np.pi * (phase0 - dphix + dphiy)

    return np.sqrt(
        (erramp0 * np.cos(phase))**2 +
        (errphase0 * amp0 * np.sin(phase))**2
    )


def propagate_error_coupling_1001_im(dphix: ArrayLike, dphiy: ArrayLike, init: PropagableBoundaryConditions) -> ArrayLike:
    """Propagates the error on the imagary part of f1001 through dphix and dphiy phase-advance,
       based on the initial amplitude and phase error of f1001.
       See Eq. (6) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ .

       Args:
           dphix (ArrayLike): Phase advances in x from initial position to observation point(s).
           dphiy (ArrayLike): Phase advances in y from initial position to observation point(s).
           init (PropagableBoundaryConditions): Initial conditions for f1001 amplitude and phase and their uncertainties.
    """
    amp0, erramp0 = init.f1001_amplitude.as_tuple()
    phase0, errphase0 = init.f1001_phase.as_tuple()
    errphase0 = 2 * np.pi * errphase0

    phase = 2 * np.pi * (phase0 - dphix + dphiy)

    return np.sqrt(
        (erramp0 * np.sin(phase))**2 +
        (errphase0 * amp0 * np.cos(phase))**2
    )


def propagate_error_coupling_1010_re(dphix: ArrayLike, dphiy: ArrayLike, init: PropagableBoundaryConditions) -> ArrayLike:
    """Propagates the error on the real part of f1010 through dphix and dphiy phase-advance,
       based on the initial amplitude and phase error of f1010.
       See Eq. (7) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ , 
       yet the phases dphix and dphiy are subtracted from the initial rdt phase.

       Args:
           dphix (ArrayLike): Phase advances in x from initial position to observation point(s).
           dphiy (ArrayLike): Phase advances in y from initial position to observation point(s).
           init (PropagableBoundaryConditions): Initial conditions for f1010 amplitude and phase and their uncertainties.
    """
    amp0, erramp0 = init.f1010_amplitude.as_tuple()
    phase0, errphase0 = init.f1010_phase.as_tuple()
    errphase0 = 2 * np.pi * errphase0

    phase = 2 * np.pi * (phase0 - dphix - dphiy)

    return np.sqrt(
        (erramp0 * np.cos(phase))**2 +
        (errphase0 * amp0 * np.sin(phase))**2
    )


def propagate_error_coupling_1010_im(dphix: ArrayLike, dphiy: ArrayLike, init: PropagableBoundaryConditions) -> ArrayLike:
    """Propagates the error on the imaginary part of f1010 through dphix and dphiy phase-advance,
       based on the initial amplitude and phase error of f1010.
       See Eq. (7) in  [LangnerDevelopmentsSegmentbySegmentTechnique2015]_ , 
       yet the phases dphix and dphiy are subtracted from the initial rdt phase.

       Args:
           dphix (ArrayLike): Phase advances in x from initial position to observation point(s).
           dphiy (ArrayLike): Phase advances in y from initial position to observation point(s).
           init (PropagableBoundaryConditions): Initial conditions for f1010 amplitude and phase and their uncertainties.
    """
    amp0, erramp0 = init.f1010_amplitude.as_tuple()
    phase0, errphase0 = init.f1010_phase.as_tuple()
    errphase0 = 2 * np.pi * errphase0

    phase = 2 * np.pi * (phase0 - dphix - dphiy)

    return np.sqrt(
        (erramp0 * np.sin(phase))**2 +
        (errphase0 * amp0 * np.cos(phase))**2
    )


def propagate_error_dispersion(beta: ArrayLike, dphi: ArrayLike, init: PropagableBoundaryConditions) -> ArrayLike:
    """Propagates the dispersion error with dphi phase-advance.

    Args:
        beta (ArrayLike): Beta function at the observation point(s).
        dphi (ArrayLike): Phase advances from the initial position to the observation point(s), in units of 2pi radians.
        init (PropagableBoundaryConditions): Initial conditions for alpha and beta and the dispersion uncertainty.
    """
    _, errdispersion0 = init.dispersion.as_tuple()
    beta0, _ = init.beta.as_tuple()
    alpha0, _ = init.alpha.as_tuple()

    return np.abs(
        errdispersion0 * np.sqrt(beta / beta0) * 
        (np.cos(2 * np.pi * dphi) + alpha0 * np.sin(2 * np.pi * dphi))
    )

    
def phase_diff(phase_a: ArrayLike, phase_b: ArrayLike) -> ArrayLike:
    """ Returns the phase difference between phase_a and phase_b, mapped to [-0.5, 0.5]. """
    phase_diff = (phase_a - phase_b) % 1
    return phase_diff - np.where(phase_diff > 0.5, 1, 0)  # this way keeps input type as is (!`where` returns np.array)