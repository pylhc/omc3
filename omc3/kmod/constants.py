"""
Constants
---------

Specific constants to be used in ``kmod``, to help with consistency.
"""
from __future__ import annotations
from pathlib import Path

SEQUENCES_PATH: Path = Path(__file__).parent / 'sequences'
EXT: str = ".tfs"
FIT_PLOTS_NAME: str = 'fit_plots.pdf'

SIDES: tuple[str, str] = ("L", "R")

# columns
K: str = "K"
TUNE: str = "TUNE"
ERR: str = "ERR"
MDL: str = "MDL"
EFFECTIVE: str = "EFF"
LUMINOSITY: str = "LUMI"
IMBALACE: str = "IMB"

BEAM: str = "BEAM"
IP: str = "IP"

BETA: str = "BET"
STAR: str = "STAR"
BETASTAR: str = f"{BETA}{STAR}"
WAIST: str = "WAIST"
BETAWAIST: str = f"{BETA}{WAIST}"
CLEANED: str = "CLEANED_"
PHASEADV: str = "PHASEADV"
AVERAGE: str = "AVERAGE"
LABEL: str = "LABEL"
TIME: str = "TIME"

# file names from kmod-application
BEAM_DIR: str = 'B'
RESULTS_FILE_NAME: str = 'results'
LSA_FILE_NAME: str = 'lsa_results'
INSTRUMENTS_FILE_NAME: str = 'beta_instrument'

# file names for omc3
BETA_FILENAME: str = 'beta_kmod_'
AVERAGED_BETASTAR_FILENAME: str = 'averaged_ip{ip}_beta{betastar}m'
AVERAGED_BPM_FILENAME: str = 'averaged_bpm_beam{beam}_ip{ip}_beta{betastar}m'
EFFECTIVE_BETAS_FILENAME: str = 'effective_betas_beta{betastar}m'

