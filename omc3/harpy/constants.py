"""
Constants
---------

Specific constants to be used in ``harpy``, to help with consistency.
"""
# Output Fileextensions --------------------------------------------------------

FILE_AMPS_EXT = ".amps{plane}"  # add
FILE_FREQS_EXT = ".freqs{plane}"  # add
FILE_LIN_EXT = ".lin{plane}"  # add

# Column Names -----------------------------------------------------------------
# Basic ---
NAME = "NAME"

# Lin Files ---
COL_TUNE = "TUNE"  # add
AMPLITUDE = "AMP"
PHASE_ADV = "MU"

COL_NATTUNE = f"NAT{COL_TUNE}"  # add
COL_NATAMP = f"NAT{AMPLITUDE}"  # add
COL_NATMU = f"NAT{PHASE_ADV}"  # add

FREQ = "FREQ"  # add
PHASE = "PHASE"
ERR = "ERR"
