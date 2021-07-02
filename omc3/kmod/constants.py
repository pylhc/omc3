"""
Constants
---------

Specific constants to be used in ``kmod``, to help with consistency.
"""
from pathlib import Path

SEQUENCES_PATH = Path(__file__).parent / 'sequences'
EXT = ".tfs"
FIT_PLOTS_NAME = 'fit_plots.pdf'

SIDES = ("L", "R")

K = "K"
TUNE = "TUNE"
ERR = "ERR"

BETA = "BET"

STAR = "STAR"
WAIST = "WAIST"
CLEANED = "CLEANED_"
PHASEADV = "PHASEADV"
AVERAGE = "AVERAGE"
RESULTS_FILE_NAME = 'results'
LSA_FILE_NAME = 'lsa_results'
INSTRUMENTS_FILE_NAME = 'beta_instrument'
