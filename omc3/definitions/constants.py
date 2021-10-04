"""
Constants
---------

General constants to use throughout ``omc3``, to help with consistency.
"""
from typing import Dict, Tuple

import numpy as np

# File Names -------------------------------------------------------------------
PI: float = np.pi
PI2: float = 2 * np.pi
PI2I: float = 2j * np.pi

# File Names -------------------------------------------------------------------
EXT: str = ".tfs"
AMP_BETA_NAME: str = "beta_amplitude_"
BETA_NAME: str = "beta_phase_"
CHROM_BETA_NAME: str = "chrom_beta_"
PHASE_NAME: str = "phase_"
SPECIAL_PHASE_NAME: str = "special_phase_"
TOTAL_PHASE_NAME: str = "total_phase_"
DISPERSION_NAME: str = "dispersion_"
NORM_DISP_NAME: str = "normalised_dispersion_"
ORBIT_NAME: str = "orbit_"
KICK_NAME: str = "kick_"
IP_NAME: str = "interaction_point_"

# Column Names -----------------------------------------------------------------
# Prefixes and Suffixes
ERR: str = "ERR"  # Error of the measurement
RMS: str = "RMS"  # Root-Mean-Square
RES: str = "RES"  # Rescaled measurement
DELTA: str = "DELTA"  # Delta between measurement and model (sometimes beating)
MDL: str = "MDL"  # Model
REAL: str = "REAL"
IMAG: str = "IMAG"

# Names
S: str = "S"
NAME: str = "NAME"
NAME2: str = f"{NAME}2"
TUNE: str = "Q"
NAT_TUNE: str = "NATQ"
PEAK2PEAK: str = "PK2PK"
ALPHA: str = "ALF"
BETA: str = "BET"
DPP: str = "DPP"
DPPAMP: str = "DPPAMP"
AMPLITUDE: str = "AMP"
PHASE: str = "PHASE"
PHASE_ADV: str = "MU"
F1001: str = "F1001"
F1010: str = "F1010"
SECONDARY_AMPLITUDE_X: str = "AMP01_X"  # amplitude of secondary line in horizontal spectrum
SECONDARY_AMPLITUDE_Y: str = "AMP10_Y"  # amplitude of secondary line in vertical spectrum
SECONDARY_FREQUENCY_X: str = "PHASE01_X"  # frequency of secondary line in horizontal spectrum
SECONDARY_FREQUENCY_Y: str = "PHASE10_Y"  # frequency of secondary line in vertical spectrum
TIME: str = "TIME"
ACTION: str = "2J"
SQRT_ACTION: str = "sqrt2J"

# Headers ----------------------------------------------------------------------
RESCALE_FACTOR: str = "RescalingFactor"

# Miscellaneous ----------------------------------------------------------------
PLANES: Tuple[str] = ("X", "Y")
PLANE_TO_NUM: Dict[str, int] = dict(X=1, Y=2)
PLANE_TO_HV: Dict[str, str] = dict(X="H", Y="V")
UNIT_IN_METERS: Dict[str, float] = dict(
    km=1e3, m=1e0, mm=1e-3, um=1e-6, nm=1e-9, pm=1e-12, fm=1e-15, am=1e-18
)
