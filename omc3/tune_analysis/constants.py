"""
Module tune_analysis.constants
----------------------------------

"""
from optics_measurements.constants import (ERR, RES, EXT, KICK_NAME, ACTION, TIME, NAT_TUNE)


# Global 'Parameters' for easy editing #########################################

ODR_PREF = "ODR_"
MOVING_AV = "MAV"
TOTAL = "TOT"
CORRECTED = "CORR"
COEFFICIENT = "COEFF{order:d}"
BBQ = "BBQ"


def get_planes():
    """ Names for the planes."""
    return "XY"


def get_timber_bbq_key(plane, beam):
    """ Key to extract bbq from timber. """
    return f'lhc.bofsu:eigen_freq_{ {"X": 1, "Y": 2}[plane] :d}_b{beam:d}'


def get_kick_out_name():
    return f"{KICK_NAME}ampdet_xy{EXT}"


def get_bbq_out_name():
    return f"bbq_ampdet.tfs"


# Kick File Headers ###########################################################


def get_tstart_head():
    """ Label for fill start time from header. """
    return "START_TIME"


def get_tend_head():
    """ Label for fill end time from header. """
    return "END_TIME"


def get_odr_header_default(j_plane, q_plane):
    return f"{ODR_PREF}J{j_plane.upper():s}Q{q_plane.upper():s}"


def get_odr_header_coeff(j_plane, q_plane, order):
    """ Header key for odr coefficient for term of given order (i.e. beta[order]) """
    return f"{get_odr_header_default(j_plane, q_plane):s}_{COEFFICIENT.format(order=order)}"


def get_odr_header_err_coeff(j_plane, q_plane, order):
    """ Header key for odr coefficient standard deviation for term of given order (i.e. sd_beta[order]) """
    return f"{get_odr_header_default(j_plane, q_plane):s}_{ERR}{COEFFICIENT.format(order=order)}"


def get_odr_header_coeff_corrected(j_plane, q_plane, order):
    """ Header key for corrected odr coefficient for term of given order (i.e. beta[order]) """
    return f"{get_odr_header_default(j_plane, q_plane)}_{CORRECTED}{COEFFICIENT.format(order=order)}"


def get_odr_header_err_coeff_corrected(j_plane, q_plane, order):
    """ Header key for corrected odr coefficient standard deviation for term of given order (i.e. sd_beta[order]) """
    return f"{get_odr_header_default(j_plane, q_plane)}_{ERR}{CORRECTED}{COEFFICIENT.format(order=order)}"


def get_mav_window_header(plane):
    """ Header to store the moving average window length """
    return f"MOVINGAV_WINDOW_{plane.upper()}"

# Kick File Columns ###########################################################


def get_time_col():
    """ Label for the TIME column."""
    return TIME


def get_bbq_col(plane):
    """ Label for the BBQ column """
    return f'{BBQ}{plane.upper():s}'


def get_mav_col(plane):
    """ Label for the moving average BBQ column. """
    return f"{get_bbq_col(plane):s}{MOVING_AV}"


def get_used_in_mav_col(plane):
    """ Label for the column showing if BBQ value was used in moving average. """
    return f"{get_bbq_col(plane):s}IN{MOVING_AV}"


def get_mav_err_col(plane):
    """ Label for the standard deviation of the moving average data. """
    return f"{ERR}{get_bbq_col(plane):s}{MOVING_AV}"


def get_corr_natq_err_col(plane):
    """ Return the standard deviation for the corrected natural tune. """
    return f"{ERR}{get_natq_corr_col(plane):s}"


def get_natq_col(plane):
    """ Label for the natural tune column. """
    return f'{NAT_TUNE}{plane.upper():s}'


def get_natq_corr_col(plane):
    """ Label for the corrected natural tune column. """
    return f"{get_natq_col(plane):s}{CORRECTED}"


def get_natq_err_col(plane):
    """ Label for the natural tune error column. """
    return f"{ERR}{get_natq_col(plane):s}"


def get_action_col(plane):
    """ Label for the action column. """
    return f"{ACTION}{plane.upper():s}{RES}"


def get_action_err_col(plane):
    """ Label for the action error column. """
    return f"{ERR}{get_action_col(plane):s}"


# Plotting #####################################################################


def get_paired_lables(action_plane, tune_plane, tune_scale=None):
    """ Labels for the action/tune plots. """
    tune_unit = ''
    if tune_scale:
        tune_unit = f' \quad [10^{{{tune_scale:-d}}}]'

    return (fr'$2J_{action_plane.lower():s} \quad [\mu m]$',
            fr'$\Delta Q_{tune_plane.lower():s}{tune_unit}$')


def get_detuning_exponent_for_order(order):
    """ Returns the default exponent for detuning orders. Highly Empirical. """
    return {1: 3, 2: 9, 3: 15}[order]


