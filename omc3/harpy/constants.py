"""
Constants
---------

Specific constants to be used in ``harpy``, to help with consistency.
"""
# Output Fileextensions --------------------------------------------------------

FILE_AMPS_EXT = ".amps{plane}"
FILE_FREQS_EXT = ".freqs{plane}"
FILE_LIN_EXT = ".lin{plane}"

# Column Names -----------------------------------------------------------------
# TODO use these everywhere (jdilly)

# Basic ---
COL_NAME = "NAME"

# Lin Files ---
COL_TUNE = "TUNE"
COL_AMP = "AMP"
COL_MU = "MU"

COL_NATTUNE = "NATTUNE"
COL_NATAMP = "NATAMP"
COL_NATMU = "NATMU"

COL_FREQ = "FREQ"
COL_PHASE = "PHASE"
COL_ERR = "ERR"

# Defocussing Monitors ---------------------------------------------------------
LIST_1 = ["BPM.20L3.B", "BPM.18L3.B",
          "BPM.21R3.B", "BPM.19R3.B",
          "BPM.20L7.B", "BPM.18L7.B",
          "BPM.21R7.B", "BPM.19R7.B",
          "BPM.20R2.B", "BPM.18R2.B",
          "BPM.21L4.B", "BPM.19L4.B",
          "BPM.20R6.B", "BPM.18R6.B",
          "BPM.21L8.B", "BPM.19L8.B",]

LIST_2 = ["BPM.21L3.B", "BPM.19L3.B",
          "BPM.20R3.B", "BPM.18R3.B",
          "BPM.21L7.B", "BPM.19L7.B",
          "BPM.20R7.B", "BPM.18R7.B",
          "BPM.21R2.B", "BPM.19R2.B",
          "BPM.20L4.B", "BPM.18L4.B",
          "BPM.21R6.B", "BPM.19R6.B",
          "BPM.20L8.B", "BPM.18L8.B",]

DEFOCUSSING_MONITORS = {
    1: {"X": LIST_1, "Y": LIST_2},
    2: {"X": LIST_2, "Y": LIST_1}
}