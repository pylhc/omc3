"""
Constants
---------

Specific constants to be used in optics_measurements, to help with consistency.
"""
# File Names -------------------------------------------------------------------
EXT: str = ".tfs"
AMP_BETA_NAME: str = "beta_amplitude_"
BETA_NAME: str = "beta_phase_"
KMOD_BETA_NAME: str = "beta_kmod_"  # TODO Check in Michis repo
KMOD_IP_NAME: str = "interaction_point_kmod_"  # TODO Check in Michis repo
CHROM_BETA_NAME: str = "chrom_beta_"
PHASE_NAME: str = "phase_"
SPECIAL_PHASE_NAME: str = "special_phase_"
TOTAL_PHASE_NAME: str = "total_phase_"
DRIVEN_PHASE_NAME: str = f"{PHASE_NAME}driven_"
DRIVEN_TOTAL_PHASE_NAME: str = f"{TOTAL_PHASE_NAME}driven_"
DISPERSION_NAME: str = "dispersion_"
NORM_DISP_NAME: str = "normalised_dispersion_"
ORBIT_NAME: str = "orbit_"
KICK_NAME: str = "kick_"
IP_NAME: str = "interaction_point_"
CALIBRATION_FILE: str = "calibration_{plane}.out"
F1001_NAME: str = "f1001"
F1010_NAME: str = "f1010"

RDT_FOLDER: str = "rdt"
CRDT_FOLDER: str = "crdt"

# Column Names -----------------------------------------------------------------
# Pre- and Suffixe
ERR: str = "ERR"  # Error of the measurement
RMS: str = "RMS"  # Root-Mean-Square
RES: str = "RES"  # Rescaled measurement
DELTA: str = "DELTA"  # Delta between measurement and model (sometimes beating)
MDL: str = "MDL"  # Model
REAL: str = "REAL"
IMAG: str = "IMAG"
MASKED: str = "MASKED"

# Names
S: str = "S"
S2: str = f"{S}2"
NAME: str = "NAME"
NAME2: str = f"{NAME}2"
TUNE: str = "Q"
NAT_TUNE: str = "NATQ"
PEAK2PEAK: str = "PK2PK"
COUNT: str = "COUNT"
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
ORBIT: str = ""  # Column is plane (X or Y) in files
CLOSED_ORBIT: str = "CO"
DISPERSION: str = "D"
NORM_DISPERSION: str = "ND"

MEASUREMENT: str = "MEAS"
MODEL: str = "MODEL"

SECONDARY_AMPLITUDE_X: str = "AMP01_X"  # amplitude of secondary line in horizontal spectrum
SECONDARY_AMPLITUDE_Y: str = "AMP10_Y"  # amplitude of secondary line in vertical spectrum
SECONDARY_FREQUENCY_X: str = "PHASE01_X"  # frequency of secondary line in horizontal spectrum
SECONDARY_FREQUENCY_Y: str = "PHASE10_Y"  # frequency of secondary line in vertical spectrum

# Kick files
TIME: str = "TIME"  # also in K-Mod
ACTION: str = "2J"
SQRT_ACTION: str = "sqrt2J"

# Calibration files
CALIBRATION = "CALIBRATION"
ERR_CALIBRATION = "ERROR_CALIBRATION"

# Headers ----------------------------------------------------------------------
RESCALE_FACTOR: str = "RescalingFactor"
BPM_RESOLUTION: str = "BPMResolution"


# K-Modulation Specific --------------------------------------------------------

# Columns and Column-Prefixes
EFFECTIVE: str = "EFF"
LUMINOSITY: str = "LUMI"
IMBALANCE: str = "IMBALANCE"
S_LOCATION: str = "_S_LOCATION"

BEAM: str = "BEAM"

STAR: str = "STAR"
BETASTAR: str = f"{BETA}{STAR}"
WAIST: str = "WAIST"
BETAWAIST: str = f"{BETA}{WAIST}"
LABEL: str = "LABEL"
KMOD_PHASE_ADV: str = "PHASEADV"

# file names from kmod-application
BEAM_DIR: str = "B"
LSA_FILE_NAME: str = "lsa_results"  # contains beta-per-BPM (and IP) results
RESULTS_FILE_NAME: str = "results"  # contains betastar results

# file names defined by omc3
AVERAGED_BETASTAR_FILENAME: str = "averaged_ip{ip}_beta{betastar_x:.2f}m{betastar_y:.2f}m"
AVERAGED_BPM_FILENAME: str = "averaged_bpm_beam{beam}_ip{ip}_beta{betastar_x:.2f}m{betastar_y:.2f}m"
EFFECTIVE_BETAS_FILENAME: str = "effective_betas_beta{betastar_x:.2f}m{betastar_y:.2f}m"

BETA_KMOD_FILENAME: str = "beta_kmod_"
BETA_STAR_FILENAME: str = "betastar_"
