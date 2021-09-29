"""
Constants
---------

Specific constants to be used in optics_measurements, to help with consistency.
"""
# File Names -------------------------------------------------------------------
EXT = ".tfs"
AMP_BETA_NAME = "beta_amplitude_"
BETA_NAME = "beta_phase_"
CHROM_BETA_NAME = "chrom_beta_"
PHASE_NAME = "phase_"
SPECIAL_PHASE_NAME = "special_phase_"
TOTAL_PHASE_NAME = "total_phase_"
DISPERSION_NAME = "dispersion_"
NORM_DISP_NAME = "normalised_dispersion_"
ORBIT_NAME = "orbit_"
KICK_NAME = "kick_"
IP_NAME = "interaction_point_"

# Column Names -----------------------------------------------------------------
# Pre- and Suffixe
ERR = "ERR"  # Error of the measurement
RMS = "RMS"  # Root-Mean-Square
RES = "RES"  # Rescaled measurement
DELTA = "DELTA"  # Delta between measurement and model (sometimes beating)
MDL = "MDL"  # Model

# Names
S = "S"
NAME = "NAME"
NAME2 = f"{NAME}2"
TUNE = "Q"
NAT_TUNE = "NATQ"
PEAK2PEAK = "PK2PK"
ALPHA = "ALF"
BETA = "BET"
DPP = "DPP"
DPPAMP = "DPPAMP"
AMPLITUDE = "AMP"
PHASE = "PHASE"
PHASE_ADV = "MU"
REAL = "REAL"
IMAG = "IMAG"

SECONDARY_AMPLITUDE_X = "AMP01_X"  # amplitude of secondary line in horizontal spectrum
SECONDARY_AMPLITUDE_Y = "AMP10_Y"  # amplitude of secondary line in vertical spectrum
SECONDARY_FREQUENCY_X = "PHASE01_X"  # frequency of secondary line in horizontal spectrum
SECONDARY_FREQUENCY_Y = "PHASE10_Y"  # frequency of secondary line in vertical spectrum

TIME = "TIME"
ACTION = "2J"
SQRT_ACTION = "sqrt2J"


# Headers ----------------------------------------------------------------------
RESCALE_FACTOR = "RescalingFactor"
