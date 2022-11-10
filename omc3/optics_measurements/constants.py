"""
Constants
---------

Specific constants to be used in optics_measurements, to help with consistency.
"""
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
CALIBRATION_FILE: str = "calibration_{plane}.out"

# Column Names -----------------------------------------------------------------
# Pre- and Suffixe
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
NAT_AMPLITUDE: str = "NATAMP"
PHASE: str = "PHASE"
PHASE_ADV: str = "MU"
F1001: str = "F1001"
F1010: str = "F1010"
NOISE: str = "NOISE"
CLOSED_ORBIT: str = "CO"

SECONDARY_AMPLITUDE_X: str = "AMP01_X"  # amplitude of secondary line in horizontal spectrum
SECONDARY_AMPLITUDE_Y: str = "AMP10_Y"  # amplitude of secondary line in vertical spectrum
SECONDARY_FREQUENCY_X: str = "PHASE01_X"  # frequency of secondary line in horizontal spectrum
SECONDARY_FREQUENCY_Y: str = "PHASE10_Y"  # frequency of secondary line in vertical spectrum

# Kick files
TIME: str = "TIME"
ACTION: str = "2J"
SQRT_ACTION: str = "sqrt2J"

# Calibration files
CALIBRATION = "CALIBRATION"
ERR_CALIBRATION = "ERROR_CALIBRATION"

# Headers ----------------------------------------------------------------------
RESCALE_FACTOR: str = "RescalingFactor"
BPM_RESOLUTION: str = "BPMResolution"
