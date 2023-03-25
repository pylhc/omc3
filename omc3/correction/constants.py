"""
Constants
---------

Specific constants to be used in correction, to help with consistency.
"""
from omc3.model.constants import TWISS_DAT

# Column Names -----------------------------------------------------------------
# Pre- and Suffixe
ERR: str = "ERR"  # Error of the measurement
RMS: str = "RMS"  # Root-Mean-Square
RES: str = "RES"  # Rescaled measurement
DELTA: str = "DELTA"  # Delta between measurement and model (sometimes beating)
MDL: str = "MDL"  # Model
VALUE: str = "VALUE"
WEIGHT: str = "WEIGHT"
ERROR: str = "ERROR"
MODEL: str = "MODEL"
DIFF: str = "DIFF"

# Names
NAME: str = "NAME"
NAME2: str = "NAME2"
S: str = "S"
BETA: str = "BET"
BETABEAT: str = "BB"
DISP: str = "D"
F1001: str = "F1001"
F1010: str = "F1010"
INCR: str = "incr"
NORM_DISP: str = "ND"
PHASE: str = "PHASE"
PHASE_ADV: str = "MU"
TUNE: str = "Q"

# Correction Test Constants ----------------------------------------------------
MODEL_MATCHED_FILENAME: str = "twiss_matched.tfs"
NOMINAL_MEASUREMENT: str = "CorrectionTest_Nominal"
UNCORRECTED_LABEL: str = "Uncorrected"
CORRECTED_LABEL: str = "Corrected"

