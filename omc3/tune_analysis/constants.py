"""
Module tune_analysis.constants
----------------------------------

"""
import pytz


# Global 'Parameters' for easy editing #########################################


def get_planes():
    """ Names for the planes."""
    return "XY"


def get_experiment_timezone():
    """ Get time zone for measurement data. """
    return pytz.timezone("Europe/Zurich")


def get_timber_bbq_key(plane, beam):
    """ Key to extract bbq from timber. """
    return f'lhc.bofsu:eigen_freq_{ {"X": 1, "Y": 2}[plane] :d}_b{beam:d}'


# Kickac Headers ###############################################################


def get_tstart_head():
    """ Label for fill start time from header. """
    return "START_TIME"


def get_tend_head():
    """ Label for fill end time from header. """
    return "END_TIME"


def get_odr_header_offset(j_plane, q_plane):
    """ Header key for odr offset (i.e. beta[0]) """
    return f"ODR_J{j_plane.upper():s}Q{q_plane.upper():s}_OFFSET"


def get_odr_header_slope(j_plane, q_plane):
    """ Header key for odr slope (i.e. beta[1]) """
    return f"ODR_J{j_plane.upper():s}Q{q_plane.upper():s}_SLOPE"


def get_odr_header_slope_std(j_plane, q_plane):
    """ Header key for odr slope standard deviation (i.e. sd_beta[1]) """
    return f"ODR_J{j_plane.upper():s}Q{q_plane.upper():s}_SLOPE_STD"


def get_odr_header_offset_corr(j_plane, q_plane):
    """ Header key for corrected odr offset (i.e. beta[0]) """
    return f"ODR_J{j_plane.upper():s}Q{q_plane.upper():s}_OFFSET_CORR"


def get_odr_header_slope_corr(j_plane, q_plane):
    """ Header key for corrected odr slope (i.e. beta[1]) """
    return f"ODR_J{j_plane.upper():s}Q{q_plane.upper():s}_SLOPE_CORR"


def get_odr_header_slope_std_corr(j_plane, q_plane):
    """ Header key for corrected odr slope standard deviation (i.e. sd_beta[1]) """
    return f"ODR_J{j_plane.upper():s}Q{q_plane.upper():s}_SLOPE_STD_CORR"


def get_mav_window_header(plane):
    """ Header to store the moving average window length """
    return f"MOVINGAV_WINDOW_{plane.upper()}"

# Kickac Columns ###############################################################


def get_time_col():
    """ Label for the TIME column."""
    return "TIME"


def get_bbq_col(plane):
    """ Label for the BBQ column """
    return f'BBQ{plane.upper():s}'


def get_mav_col(plane):
    """ Label for the moving average BBQ column. """
    return f"{get_bbq_col(plane):s}MAV"


def get_used_in_mav_col(plane):
    """ Label for the column showing if BBQ value was used in moving average. """
    return f"{get_bbq_col(plane):s}INMAV"


def get_mav_std_col(plane):
    """ Label for the standard deviation of the moving average data. """
    return f"ERR{get_bbq_col(plane):s}MAV"


def get_total_natq_std_col(plane):
    """ Return the total standard deviation for the natural tune. """
    return f"ERR{get_natq_col(plane):s}TOT"


def get_natq_col(plane):
    """ Label for the natural tune column. """
    return f'NATQ{plane.upper():s}'


def get_natq_corr_col(plane):
    """ Label for the corrected natural tune column. """
    return f"{get_natq_col(plane):s}CORR"


def get_natq_err_col(plane):
    """ Label for the natural tune error column. """
    return f"ERR{get_natq_col(plane):s}"


def get_action_col(plane):
    """ Label for the action column. """
    return f"2J{plane.upper():s}RES"


def get_action_err_col(plane):
    """ Label for the action error column. """
    return get_action_col(f"ERR{plane.upper():s}")


# Plotting #####################################################################


def get_paired_lables(action_plane, tune_plane):
    """ Labels for the action/tune plots. """
    return (fr'$2J_{action_plane.lower():s} \quad [\mu m]$',
            fr'$\Delta Q_{tune_plane.lower():s}$')


