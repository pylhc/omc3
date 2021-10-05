"""
Constants
---------

Specific constants to be used in ``kmod``, to help with consistency.
"""
from pathlib import Path  # dup
from typing import Tuple  # dup

SEQUENCES_PATH: Path = Path(__file__).parent / "sequences"
EXT = ".tfs"  # dup
FIT_PLOTS_NAME: str = "fit_plots.pdf"

SIDES: Tuple[str, str] = ("L", "R")

K: str = "K"
COL_TUNE: str = "TUNE"  # dup from harpy
ERR: str = "ERR"  # dup

BETA: str = "BET"  # dup

STAR: str = "STAR"
WAIST: str = "WAIST"
CLEANED: str = "CLEANED_"
PHASEADV = "PHASEADV"
AVERAGE: str = "AVERAGE"
RESULTS_FILE_NAME: str = "results"
LSA_FILE_NAME: str = "lsa_results"
INSTRUMENTS_FILE_NAME: str = "beta_instrument"
