"""
Module tune_analysis.kickac_modifiers
--------------------------------------

Functions to add data to or extract data from kick_ac files.
"""
import numpy as np

from tune_analysis import constants as const, bbq_tools
from utils import logging_tools


LOG = logging_tools.get_logger(__name__)

# Column Names
COL_ACTION = const.get_action_col
COL_ACTION_ERR = const.get_action_err_col

COL_MAV = const.get_mav_col
COL_MAV_STD = const.get_mav_std_col
COL_IN_MAV = const.get_used_in_mav_col

COL_NATQ = const.get_natq_col
COL_NATQ_STD = const.get_natq_err_col
COL_NATQ_CORR = const.get_natq_corr_col
COL_NATQ_TOTSTD = const.get_total_natq_std_col

COL_TIME = const.get_time_col
COL_BBQ = const.get_bbq_col

HEADER_CORR_OFFSET = const.get_odr_header_offset_corr
HEADER_CORR_SLOPE = const.get_odr_header_slope_corr
HEADER_CORR_SLOPE_STD = const.get_odr_header_slope_std_corr

HEADER_OFFSET = const.get_odr_header_offset
HEADER_SLOPE = const.get_odr_header_slope
HEADER_SLOPE_STD = const.get_odr_header_slope_std

PLANES = const.get_planes()


def _get_odr_headers(corrected):
    """ Return Headers needed for ODR. """
    if corrected:
        return HEADER_CORR_SLOPE, HEADER_CORR_SLOPE_STD, HEADER_CORR_OFFSET
    return HEADER_SLOPE, HEADER_SLOPE_STD, HEADER_OFFSET


def _get_ampdet_columns(corrected):
    """ Get columns needed for amplitude detuning """
    if corrected:
        return COL_NATQ_CORR, COL_NATQ_TOTSTD
    return COL_NATQ, COL_NATQ_STD


def _um_to_m(f):
    """ Convert from 1/um to 1/m and return result as rounded integer. """
    return int(round(f*1e6))


def _get_slope_label(slope, std):
    """ Returns a well formatted slope label. """
    str_slope = '{:> 7d}'.format(_um_to_m(slope))
    str_std = '{:>6d}'.format(_um_to_m(std))
    return f'{str_slope} $\pm$ {str_std} m$^{{-1}}$'


# Data Addition ################################################################


def add_bbq_data(kickac_df, bbq_series, column):
    """ Add bbq values from series to kickac dataframe into column.

    Args:
        kickac_df: kickac dataframe
                  (needs to contain column "TIME_COL" or has time as index)
        bbq_series: series of bbq data with time as index
        column: column name to add the data into

    Returns: modified kickac dataframe

    """
    time_indx = kickac_df.index
    if COL_TIME in kickac_df:
        time_indx = kickac_df[COL_TIME]

    values = []
    for time in time_indx:
        values.append(bbq_series.iloc[bbq_series.index.get_loc(time, method="nearest")])
    kickac_df[column] = values
    return kickac_df


def add_moving_average(kickac_df, bbq_df, **kwargs):
    """ Adds the moving average of the bbq data to kickac_df and bbq_df. """
    LOG.debug("Calculating moving average.")
    for plane in PLANES:
        tune = f"tune_{plane.lower()}"
        bbq_mav, bbq_std, mask = bbq_tools.get_moving_average(bbq_df[COL_BBQ(plane)],
                                                              length=kwargs["window_length"],
                                                              min_val=kwargs[f"{tune}_min"],
                                                              max_val=kwargs[f"{tune}_max"],
                                                              fine_length=kwargs["fine_window"],
                                                              fine_cut=kwargs["fine_cut"],
                                                              )
        bbq_df[COL_MAV(plane)] = bbq_mav
        bbq_df[COL_MAV_STD(plane)] = bbq_std
        bbq_df[COL_IN_MAV(plane)] = ~mask
        kickac_df = add_bbq_data(kickac_df, bbq_mav, COL_MAV(plane))
        kickac_df = add_bbq_data(kickac_df, bbq_std, COL_MAV_STD(plane))
    return kickac_df, bbq_df


def add_corrected_natural_tunes(kickac_df):
    """ Adds the corrected natural tunes to kickac

    Args:
        kickac_df: Dataframe containing the data

    Returns:
        Modified kick_ac
    """
    for plane in PLANES:
        kickac_df[COL_NATQ_CORR(plane)] = \
            kickac_df[COL_NATQ(plane)] - kickac_df[COL_MAV(plane)]
    return kickac_df


def add_odr(kickac_df, odr_fit, action_plane, tune_plane, corrected=False):
    """ Adds the odr fit of the (un)corrected data to the header of the kickac.

    Args:
        kickac_df: Dataframe containing the data
        odr_fit: odr-fit data (definitions see ``detuning_tools.py``)
        action_plane: Plane of the action
        tune_plane: Plane of the tune

    Returns:
        Modified kick_ac
    """
    header_slope, header_slope_std, header_offset = _get_odr_headers(corrected)

    kickac_df.headers[header_offset(action_plane, tune_plane)] = odr_fit.beta[0]
    kickac_df.headers[header_slope(action_plane, tune_plane)] = odr_fit.beta[1]
    kickac_df.headers[header_slope_std(action_plane, tune_plane)] = odr_fit.sd_beta[1]
    return kickac_df


def add_total_natq_std(kickac_df):
    """ Add the total standard deviation of the natural tune to the kickac.
    The total standard deviation is here defined as the standard deviation of the measurement
    plus the standard deviation of the moving average.

    Args:
        kickac_df: Dataframe containing the data

    Returns:
        Modified kick_ac
    """
    for plane in PLANES:
        kickac_df[COL_NATQ_TOTSTD(plane)] = np.sqrt(
            np.power(kickac_df[COL_NATQ_STD(plane)], 2) +
            np.power(kickac_df[COL_MAV_STD(plane)], 2)
        )
    return kickac_df


# Data Extraction ##############################################################


def get_odr_data(kickac_df, action_plane, tune_plane, corrected=False):
    """ Extract the data from odr.

    Args:
        kickac_df: Dataframe containing the data
        action_plane: Plane of the action
        tune_plane: Plane of the tune

    Returns:
        Dictionary containing x,y, ylower, yupper, offset and label

    """
    header_slope, header_slope_std, header_offset = _get_odr_headers(corrected)

    slope = kickac_df.headers[header_slope(action_plane, tune_plane)]
    slope_std = kickac_df.headers[header_slope_std(action_plane, tune_plane)]
    x = [0, max(kickac_df[COL_ACTION(action_plane)])*1.05]
    y = [0, x[1] * slope]
    y_low = [0, x[1] * (slope - slope_std)]
    y_upp = [0, x[1] * (slope + slope_std)]
    return {
        "x": np.array(x),
        "y": np.array(y),
        "label": _get_slope_label(slope, slope_std),
        "offset": kickac_df.headers[header_offset(action_plane, tune_plane)],
        "ylower": np.array(y_low),
        "yupper": np.array(y_upp),
    }


def get_ampdet_data(kickac_df, action_plane, tune_plane, corrected=False):
    """ Extract the data needed for plotting the (un)corrected amplitude detuning
    from the kickac dataframe.

    Args:
        kickac_df: Dataframe containing the data
        action_plane: Plane of the action
        tune_plane: Plane of the tune


    Returns:
        Dictionary containing x,y, x_err and y_err

    """
    col_natq, col_natq_std = _get_ampdet_columns(corrected)

    columns = {"x": COL_ACTION(action_plane),
               "xerr": COL_ACTION_ERR(action_plane),
               "y": col_natq(tune_plane),
               "yerr": col_natq_std(tune_plane),
               }

    data = kickac_df.loc[:, [columns[key] for key in columns.keys()]]
    data.columns = columns.keys()

    if data.isna().any().any():
        LOG.warn(
            f"Amplitude Detuning data for Q{tune_plane} and J{action_plane} contains NaNs"
        )
        data = data.dropna(axis=0)
    return data.to_dict('series')
