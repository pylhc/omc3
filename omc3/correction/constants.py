"""
Constants
---------

Specific constants to be used in correction, to help with consistency.
"""
# Same as in optics-measurements:

from omc3.optics_measurements.constants import (
    BETA_NAME,
    AMP_BETA_NAME,
    ORBIT_NAME,
    DISPERSION_NAME,
    NORM_DISP_NAME,
    PHASE_NAME,
    TOTAL_PHASE_NAME,
    F1001_NAME,
    F1010_NAME,

    # Column Suffixes
    ERR,  # Error of the measurement
    RMS,  # Root-Mean-Square
    RES,  # Rescaled measurement
    DELTA,  # Delta between measurement and model (sometimes beating)
    MDL,  # Model

    # Column Names
    NAME,
    NAME2,
    S,
    BETA,
    DISPERSION,
    NORM_DISPERSION,
    F1001,
    F1010,
    PHASE,
    PHASE_ADV,
    TUNE,
    AMPLITUDE,
    REAL,
    IMAG,
)

# Column Names -----------------------------------------------------------------
VALUE: str = "VALUE"
WEIGHT: str = "WEIGHT"
ERROR: str = "ERROR"
MODEL: str = "MODEL"
DIFF: str = "DIFF"
NCR: str = "incr"

# Correction Test Constants ----------------------------------------------------
MODEL_NOMINAL_FILENAME: str = "twiss_nominal.tfs"
MODEL_MATCHED_FILENAME: str = "twiss_matched.tfs"
NOMINAL_MEASUREMENT: str = "CorrectionTest_Nominal.tfs"
UNCORRECTED_LABEL: str = "Uncorrected"
CORRECTED_LABEL: str = "Corrected"
EXPECTED: str = "EXP"

COUPLING_NAME_TO_MODEL_COLUMN_SUFFIX = {  # I know, I know ... (jdilly, 2023)
    F1001_NAME: F1001,
    F1010_NAME: F1010,
}
