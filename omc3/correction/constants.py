"""
Constants
---------

Specific constants to be used in correction, to help with consistency.
"""
# Trying to decouple input from output columns, but maybe not?
# Same as in optics-measurements:
from omc3.optics_measurements.constants import (
    F1001_NAME,
    F1010_NAME,
    F1001,
    F1010,
)

# Column Names -----------------------------------------------------------------
VALUE: str = "VALUE"
WEIGHT: str = "WEIGHT"
ERROR: str = "ERROR"
MODEL: str = "MODEL"
DIFF: str = "DIFF"
EXPECTED: str = "EXP"

COUPLING_NAME_TO_MODEL_COLUMN_SUFFIX = {  # I know, I know ... (jdilly, 2023)
    F1001_NAME: F1001,
    F1010_NAME: F1010,
}

# For FullResponse
INCR: str = "incr"
ORBIT_DPP: str = "orbit_dpp"

# Correction Test Constants ----------------------------------------------------
MODEL_NOMINAL_FILENAME: str = "twiss_nominal.tfs"  # using twiss from model for now
MODEL_MATCHED_FILENAME: str = "twiss_matched.tfs"

# Plotting Labels
UNCORRECTED_LABEL: str = "Measurement"
CORRECTED_LABEL: str = "Corrected"  # default label if none given
EXPECTED_LABEL: str = "Expected"
CORRECTION_LABEL: str = "Correction"


