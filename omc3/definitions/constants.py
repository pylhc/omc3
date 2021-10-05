"""
Constants
---------

General constants to use throughout ``omc3``, to help with consistency.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Numbers -------------------------------------------------------------------
PI: float = np.pi
PI2: float = 2 * np.pi
PI2I: complex = 2j * np.pi

# File Names -------------------------------------------------------------------
# Extensions
EXT: str = ".tfs"
FILE_AMPS_EXT = ".amps{plane}"  # for harpy
FILE_FREQS_EXT = ".freqs{plane}"  # for harpy
FILE_LIN_EXT = ".lin{plane}"  # for harpy

# Names
AMP_BETA_NAME: str = "beta_amplitude_"  # for optics_measurements
BETA_NAME: str = "beta_phase_"  # for optics_measurements
CHROM_BETA_NAME: str = "chrom_beta_"  # for optics_measurements
DISPERSION_NAME: str = "dispersion_"  # for optics_measurements
FIT_PLOTS_NAME: str = "fit_plots.pdf"  # for kmod
INSTRUMENTS_FILE_NAME: str = "beta_instrument"  # for kmod
IP_NAME: str = "interaction_point_"  # for optics_measurements
KICK_NAME: str = "kick_"  # for optics_measurements
LSA_FILE_NAME: str = "lsa_results"  # for kmod
NORM_DISP_NAME: str = "normalised_dispersion_"  # for optics_measurements
ORBIT_NAME: str = "orbit_"  # for optics_measurements
PHASE_NAME: str = "phase_"  # for optics_measurements
RESULTS_FILE_NAME: str = "results"  # for kmod
SPECIAL_PHASE_NAME: str = "special_phase_"  # for optics_measurements
TOTAL_PHASE_NAME: str = "total_phase_"  # for optics_measurements

# Column Names -----------------------------------------------------------------
# Prefixes and Suffixes
CLEANED: str = "CLEANED_"  # for kmod
ERR: str = "ERR"  # Error of the measurement
ERROR: str = "ERROR"  # for omc3.correction
DELTA: str = "DELTA"  # Delta between measurement and model (sometimes beating)
DIFF: str = "DIFF"  # for omc3.correction
IMAG: str = "IMAG"  # Imaginary part of complex
MDL: str = "MDL"  # Model
MODEL: str = "MODEL"  # for omc3.correction
REAL: str = "REAL"  # Real part of complex
RES: str = "RES"  # Rescaled measurement
RMS: str = "RMS"  # Root-Mean-Square
STAR: str = "STAR"  # for kmod
VALUE: str = "VALUE"  # for omc3.correction
WAIST: str = "WAIST"  # for kmod
WEIGHT: str = "WEIGHT"  # for omc3.correction

# Names
ACTION: str = "2J"
ALPHA: str = "ALF"
AMPLITUDE: str = "AMP"
AVERAGE: str = "AVERAGE"  # for kmod
BETA: str = "BET"
BETABEAT: str = "BB"
DISP: str = "D"
DPP: str = "DPP"
DPPAMP: str = f"DPP{AMPLITUDE}"
F1001: str = "F1001"
F1010: str = "F1010"
INCR: str = "incr"
K: str = "K"  # for kmod
NAME: str = "NAME"
NAME2: str = f"{NAME}2"
NORM_DISP: str = f"N{DISP}"
PEAK2PEAK: str = "PK2PK"
PHASE: str = "PHASE"
PHASE_ADV: str = "MU"
PHASEADV = "PHASEADV"  # for kmod
S: str = "S"
SECONDARY_AMPLITUDE_X: str = f"{AMPLITUDE}01_X"  # amplitude of secondary line in horizontal spectrum
SECONDARY_AMPLITUDE_Y: str = f"{AMPLITUDE}10_Y"  # amplitude of secondary line in vertical spectrum
SECONDARY_FREQUENCY_X: str = f"{PHASE}01_X"  # frequency of secondary line in horizontal spectrum
SECONDARY_FREQUENCY_Y: str = f"{PHASE}10_Y"  # frequency of secondary line in vertical spectrum
SQRT_ACTION: str = f"sqrt{ACTION}"
TIME: str = "TIME"
TUNE: str = "Q"
NAT_TUNE: str = f"NAT{TUNE}"

# harpy specific, unfortunately
COL_TUNE = "TUNE"
COL_NATTUNE = f"NAT{COL_TUNE}"
COL_NATAMP = f"NAT{AMPLITUDE}"
COL_NATMU = f"NAT{PHASE_ADV}"
FREQ = "FREQ"

# Headers ----------------------------------------------------------------------
RESCALE_FACTOR: str = "RescalingFactor"

# Miscellaneous ----------------------------------------------------------------
PLANES: Tuple[str] = ("X", "Y")
PLANE_TO_NUM: Dict[str, int] = dict(X=1, Y=2)
PLANE_TO_HV: Dict[str, str] = dict(X="H", Y="V")
SIDES: Tuple[str, str] = ("L", "R")
UNIT_IN_METERS: Dict[str, float] = dict(km=1e3, m=1e0, mm=1e-3, um=1e-6, nm=1e-9, pm=1e-12, fm=1e-15, am=1e-18)

# Model creator specifics ------------------------------------------------------
MACROS_DIR: str = "macros"
OBS_POINTS: str = "observation_points.def"
MODIFIERS_MADX: str = "modifiers.madx"
MODIFIER_TAG: str = "!@modifier"
TWISS_BEST_KNOWLEDGE_DAT: str = "twiss_best_knowledge.dat"
TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT: str = "twiss_elements_best_knowledge.dat"
TWISS_ADT_DAT: str = "twiss_adt.dat"
TWISS_AC_DAT: str = "twiss_ac.dat"
TWISS_ELEMENTS_DAT: str = "twiss_elements.dat"
TWISS_DAT: str = "twiss.dat"
ERROR_DEFFS_TXT: str = "error_deffs.txt"
JOB_MODEL_MADX: str = "job.create_model.madx"

# Macros
GENERAL_MACROS: str = "general.macros.madx"
LHC_MACROS: str = "lhc.macros.madx"

# Settings files
B2_SETTINGS_MADX: str = "b2_settings.madx"
B2_ERRORS_TFS: str = "b2_errors.tfs"

# Important files locations ----------------------------------------------------
# Afs acc-models lhc repo
ACCELERATOR_MODEL_REPOSITORY: Path = Path("/afs/cern.ch/eng/acc-models/lhc")

# K-Mod sequences
KMOD_SEQUENCES_PATH: Path = Path(__file__).parent.parent / "kmod" / "sequences"

# Tune analysis specifics ------------------------------------------------------
ODR_PREF: str = "ODR_"
MOVING_AV: str = "MAV"
TOTAL: str = "TOT"
CORRECTED: str = "CORR"
COEFFICIENT: str = "COEFF{order:d}"
BBQ: str = "BBQ"


def get_timber_bbq_key(plane, beam) -> str:
    """ Key to extract bbq from timber. """
    return f"lhc.bofsu:eigen_freq_{PLANE_TO_NUM[plane] :d}_b{beam:d}"


def get_kick_out_name() -> str:
    return f"{KICK_NAME}ampdet_xy{EXT}"


def get_bbq_out_name() -> str:
    return f"bbq_ampdet.tfs"


# Kick File Headers
def get_tstart_head() -> str:
    """ Label for fill start time from header. """
    return "START_TIME"


def get_tend_head() -> str:
    """ Label for fill end time from header. """
    return "END_TIME"


def get_odr_header_default(q_plane: str, j_plane: str) -> str:
    return f"{ODR_PREF}dQ{q_plane.upper():s}d2J{j_plane.upper():s}"


def get_odr_header_coeff(q_plane: str, j_plane: str, order: int) -> str:
    """ Header key for odr coefficient for term of given order (i.e. beta[order]) """
    return f"{get_odr_header_default(q_plane, j_plane) :s}_{COEFFICIENT.format(order=order)}"


def get_odr_header_err_coeff(q_plane: str, j_plane: str, order: int) -> str:
    """ Header key for odr coefficient standard deviation for term of given order (i.e. sd_beta[order]) """
    return f"{get_odr_header_default(q_plane, j_plane) :s}_{ERR}{COEFFICIENT.format(order=order)}"


def get_odr_header_coeff_corrected(q_plane: str, j_plane: str, order: int) -> str:
    """ Header key for corrected odr coefficient for term of given order (i.e. beta[order]) """
    return f"{get_odr_header_default(q_plane, j_plane)}_{CORRECTED}{COEFFICIENT.format(order=order)}"


def get_odr_header_err_coeff_corrected(q_plane: str, j_plane: str, order: int) -> str:
    """ Header key for corrected odr coefficient standard deviation for term of given order (i.e. sd_beta[order]) """
    return f"{get_odr_header_default(q_plane, j_plane)}_{ERR}{CORRECTED}{COEFFICIENT.format(order=order)}"


def get_mav_window_header(plane: str) -> str:
    """ Header to store the moving average window length """
    return f"MOVINGAV_WINDOW_{plane.upper()}"


# Kick File Columns
def get_time_col() -> str:
    """ Label for the TIME column."""
    return TIME


def get_bbq_col(plane: str) -> str:
    """ Label for the BBQ column """
    return f"{BBQ}{plane.upper():s}"


def get_mav_col(plane: str) -> str:
    """ Label for the moving average BBQ column. """
    return f"{get_bbq_col(plane):s}{MOVING_AV}"


def get_used_in_mav_col(plane: str) -> str:
    """ Label for the column showing if BBQ value was used in moving average. """
    return f"{get_bbq_col(plane):s}IN{MOVING_AV}"


def get_mav_err_col(plane: str) -> str:
    """ Label for the standard deviation of the moving average data. """
    return f"{ERR}{get_bbq_col(plane):s}{MOVING_AV}"


def get_corr_natq_err_col(plane: str) -> str:
    """ Return the standard deviation for the corrected natural tune. """
    return f"{ERR}{get_natq_corr_col(plane):s}"


def get_natq_col(plane: str) -> str:
    """ Label for the natural tune column. """
    return f"{NAT_TUNE}{plane.upper():s}"


def get_natq_corr_col(plane: str) -> str:
    """ Label for the corrected natural tune column. """
    return f"{get_natq_col(plane):s}{CORRECTED}"


def get_natq_err_col(plane: str) -> str:
    """ Label for the natural tune error column. """
    return f"{ERR}{get_natq_col(plane):s}"


def get_action_col(plane: str) -> str:
    """ Label for the action column. """
    return f"{ACTION}{plane.upper():s}{RES}"


def get_action_err_col(plane: str) -> str:
    """ Label for the action error column. """
    return f"{ERR}{get_action_col(plane):s}"


# Plotting
def get_paired_labels(tune_plane: str, action_plane: str, tune_scale: int = None) -> Tuple[str, str]:
    """ Labels for the action/tune plots. """
    tune_unit = ""
    if tune_scale:
        tune_unit = f" \quad [10^{{{tune_scale:-d}}}]"

    return (fr"$2J_{action_plane.lower():s} \quad [\mu m]$", fr"$\Delta Q_{tune_plane.lower():s}{tune_unit}$")


def get_detuning_exponent_for_order(order: int) -> int:
    """ Returns the default exponent for detuning orders. Highly Empirical. """
    return {1: 3, 2: 9, 3: 15}[order]
