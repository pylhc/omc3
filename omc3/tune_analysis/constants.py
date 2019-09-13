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
    return 'lhc.bofsu:eigen_freq_{:d}_b{:d}'.format({"X": 1, "Y": 2}[plane], beam)


# Kickac Headers ###############################################################


def get_tstart_head():
    """ Label for fill start time from header. """
    return "START_TIME"


def get_tend_head():
    """ Label for fill end time from header. """
    return "END_TIME"


def get_odr_header_offset(j_plane, q_plane):
    """ Header key for odr offset (i.e. beta[0]) """
    return "ODR_J{:s}Q{:s}_OFFSET".format(j_plane, q_plane)


def get_odr_header_slope(j_plane, q_plane):
    """ Header key for odr slope (i.e. beta[1]) """
    return "ODR_J{:s}Q{:s}_SLOPE".format(j_plane, q_plane)


def get_odr_header_slope_std(j_plane, q_plane):
    """ Header key for odr slope standard deviation (i.e. sd_beta[1]) """
    return "ODR_J{:s}Q{:s}_SLOPE_STD".format(j_plane, q_plane)


def get_odr_header_offset_corr(j_plane, q_plane):
    """ Header key for corrected odr offset (i.e. beta[0]) """
    return "ODR_J{:s}Q{:s}_OFFSET_CORR".format(j_plane, q_plane)


def get_odr_header_slope_corr(j_plane, q_plane):
    """ Header key for corrected odr slope (i.e. beta[1]) """
    return "ODR_J{:s}Q{:s}_SLOPE_CORR".format(j_plane, q_plane)


def get_odr_header_slope_std_corr(j_plane, q_plane):
    """ Header key for corrected odr slope standard deviation (i.e. sd_beta[1]) """
    return "ODR_J{:s}Q{:s}_SLOPE_STD_CORR".format(j_plane, q_plane)


# Kickac Columns ###############################################################


def get_time_col():
    """ Label for the TIME column."""
    return "TIME"


def get_bbq_col(plane):
    """ Label for the BBQ column """
    return 'BBQ{:s}'.format(plane.upper())


def get_mav_col(plane):
    """ Label for the moving average BBQ column. """
    return "{:s}MAV".format(get_bbq_col(plane))


def get_used_in_mav_col(plane):
    """ Label for the column showing if BBQ value was used in moving average. """
    return "{:s}INMAV".format(get_bbq_col(plane))


def get_mav_std_col(plane):
    """ Label for the standard deviation of the moving average data. """
    return "{:s}MAVSTD".format(get_bbq_col(plane))


def get_total_natq_std_col(plane):
    """ Return the total standard deviation for the natural tune. """
    return "{:s}TOTSTD".format(get_natq_col(plane))


def get_natq_col(plane):
    """ Label for the natural tune column. """
    return 'NATQ{:s}'.format(plane.upper())


def get_natq_corr_col(plane):
    """ Label for the corrected natural tune column. """
    return "{:s}CORR".format(get_natq_col(plane))


def get_natq_err_col(plane):
    """ Label for the natural tune error column. """
    return "{:s}RMS".format(get_natq_col(plane))


def get_action_col(plane):
    """ Label for the action column. """
    return "2J{:s}RES".format(plane.upper())


def get_action_err_col(plane):
    """ Label for the action error column. """
    return get_action_col("{:s}STD".format(plane))


# Plotting #####################################################################


def get_paired_lables(action_plane, tune_plane):
    """ Labels for the action/tune plots. """
    return (r'$2J_{:s} \quad [\mu m]$'.format(action_plane.lower()),
            r'$\Delta Q_{:s}$'.format(tune_plane.lower()))


