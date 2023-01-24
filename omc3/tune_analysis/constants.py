"""
Constants
---------

Specific constants and helpers to be used in ``tune_analysis``, to help with consistency.
"""
from dataclasses import dataclass
from typing import Tuple, Sequence

import pandas as pd
from omc3.definitions.constants import PLANE_TO_NUM
from omc3.optics_measurements.constants import ACTION, ERR, EXT, KICK_NAME, NAT_TUNE, RES, TIME

# Global 'Parameters' for easy editing #########################################

ODR_PREF: str = "ODR_"
MOVING_AV: str = "MAV"
TOTAL: str = "TOT"
CORRECTED: str = "CORR"
COEFFICIENT: str = "COEFF{order:d}"
BBQ: str = "BBQ"

INPUT_KICK = "kick"
INPUT_PREVIOUS = "previous"


def get_timber_bbq_key(plane: str, beam: int) -> str:
    """ Key to extract bbq from timber. """
    # return f"lhc.bofsu:eigen_freq_{PLANE_TO_NUM[plane]:d}_b{beam:d}".upper()  # pre run 3
    # return f"BFC.LHC:TuneFBAcq:TUNEB{beam:d}{PLANE_TO_HV[plane]}"  # contains less data
    return f"LHC.BQBBQ.CONTINUOUS_HS.B{beam:d}:EIGEN_FREQ_{PLANE_TO_NUM[plane]:d}"


def get_kick_out_name() -> str:
    return f"{KICK_NAME}ampdet_xy{EXT}"


def get_bbq_out_name() -> str:
    return f"bbq_ampdet.tfs"


@dataclass
class AmpDetData:
    tune_plane: str
    action_plane: str
    action: pd.Series
    action_err: pd.Series
    tune: pd.Series
    tune_err: pd.Series


@dataclass
class FakeOdrOutput:
    beta: Sequence[float]
    sd_beta: Sequence[float]


# Kick File Headers ###########################################################


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


def get_mav_window_header() -> str:
    """ Header to store the moving average window length """
    return "MOVINGAV_WINDOW"


def get_fine_window_header() -> str:
    """ Header to store the moving average fine window length """
    return "FINE_WINDOW"


def get_fine_cut_header() -> str:
    """ Header to store the fine cut. """
    return "FINE_CUT"


def get_outlier_limit_header() -> str:
    """ Header to store the outlier limit."""
    return "OUTLIER_LIMIT"


def get_min_tune_header(plane: str) -> str:
    """ Header to store the minimum tune cut. """
    return f"MIN_Q{plane.upper()}"


def get_max_tune_header(plane: str) -> str:
    """ Header to store the maximum tune cut. """
    return f"MAX_Q{plane.upper()}"


# Kick File Columns ###########################################################


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


# Plotting #####################################################################


def get_tune_label(plane: str, scale: int = None) -> str:
    """ Tune label for the action/tune plots. """
    unit = ""
    if scale:
        unit = f" \quad [10^{{{scale:d}}}]"
    return fr"$\Delta Q_{plane.lower():s}{unit}$"


def get_action_label(plane: str, unit: str) -> str:
    """ Action label for the action/tune plots. """
    if unit == "um":
        unit = r"\mu m"
    return fr"$2J_{plane.lower():s} \quad [{unit:s}]$"


def get_detuning_exponent_for_order(order: int) -> int:
    """ Returns the default exponent for detuning orders. Highly Empirical. """
    return {1: 3, 2: 11, 3: 15}[order]
