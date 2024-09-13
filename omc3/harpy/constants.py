"""
Constants
---------

Specific constants to be used in ``harpy``, to help with consistency.
"""
# Output Fileextensions --------------------------------------------------------

FILE_AMPS_EXT: str = ".amps{plane}"
FILE_FREQS_EXT: str = ".freqs{plane}"
FILE_LIN_EXT: str = ".lin{plane}"

LINFILES_SUBFOLDER: str = "lin_files"

# Column Names -----------------------------------------------------------------
# TODO use these everywhere (jdilly)

# Basic ---
COL_NAME: str = "NAME"

# Lin Files ---
COL_TUNE: str = "TUNE"
COL_AMP: str = "AMP"
COL_MU: str = "MU"

COL_NATTUNE: str = "NATTUNE"
COL_NATAMP: str = "NATAMP"
COL_NATMU: str = "NATMU"

COL_FREQ: str = "FREQ"
COL_PHASE: str = "PHASE"
COL_ERR: str = "ERR"
