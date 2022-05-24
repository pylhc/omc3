"""
BBQ Tools
---------

Tools to handle BBQ data.
"""
from typing import Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from omc3.utils import logging_tools
from omc3.utils.outliers import get_filter_mask


LOG = logging_tools.get_logger(__name__)


def get_moving_average(data_series, length=20,
                       min_val=None, max_val=None, fine_length=None, fine_cut=None):
    """
    Get a moving average of the ``data_series`` over ``length`` entries. The data can be filtered
    beforehand. The values are shifted, so that the averaged value takes ceil((length-1)/2)
    values previous and floor((length-1)/2) following values into account.

    Args:
        data_series: `Series` of data.
        length: length of the averaging window.
        min_val: minimum value (for filtering).
        max_val: maximum value (for filtering).
        fine_length: length of the averaging window for fine cleaning.
        fine_cut: allowed deviation for fine cleaning.

    Returns:
        A filtered and averaged `Series` and the mask used for filtering data.
    """
    LOG.debug(f"Cutting data and calculating moving average of length {length:d}.")

    if bool(fine_length) != bool(fine_cut):
        raise NotImplementedError("To activate fine cleaning, both "
                                  "'fine_window' and 'fine_cut' are needed.")

    if min_val is not None:
        min_mask = data_series <= min_val
    else:
        min_mask = np.zeros(data_series.size, dtype=bool)

    if max_val is not None:
        max_mask = data_series >= max_val
    else:
        max_mask = np.zeros(data_series.size, dtype=bool)

    cut_mask = min_mask | max_mask | data_series.isna()
    _is_almost_empty_mask(~cut_mask, length)
    data_mav, std_mav = _get_interpolated_moving_average(data_series, cut_mask, length)

    if fine_length is not None:
        min_mask = data_series <= (data_mav - fine_cut)
        max_mask = data_series >= (data_mav + fine_cut)
        cut_mask = min_mask | max_mask | data_series.isna()
        _is_almost_empty_mask(~cut_mask, fine_length)
        data_mav, std_mav = _get_interpolated_moving_average(data_series, cut_mask, fine_length)

    return data_mav, std_mav, ~cut_mask


def clean_outliers_moving_average(data_series: pd.Series, length: int, limit: float) -> Tuple[pd.Series, pd.Series, ArrayLike]:
    """
    Get a moving average of the ``data_series`` over ``length`` entries, by means of
    :func:`outlier filter <omc3.utils.outliers.get_filter_mask>`.
    The values are shifted, so that the averaged value takes ceil((length-1)/2) values previous
    and floor((length-1)/2) following values into account.

    Args:
        data_series: `Series` of data.
        length: length of the averaging window.
        limit: points beyond that limit are always filtered.
    """
    LOG.debug(f"Filtering and calculating moving average of length {length:d}.")
    init_mask = ~data_series.isna()
    mask = init_mask.copy()
    for i in range(len(data_series)-length):
        mask[i:i+length] &= get_filter_mask(data_series[i:i+length], limit=limit, mask=init_mask[i:i+length])

    _is_almost_empty_mask(mask, length)
    data_mav, std_mav = _get_interpolated_moving_average(data_series, ~mask, length)
    return data_mav, std_mav, mask


# Private methods ############################################################


def _get_interpolated_moving_average(data_series: pd.Series, clean_mask: Union[pd.Series, ArrayLike], length: int) -> Tuple[pd.Series, pd.Series]:
    """
    Returns the moving average of data series with a window of length and interpolated ``NaNs``.
    """
    data = data_series.copy()
    is_datetime_index = isinstance(data.index[0], datetime)

    if is_datetime_index:
        # in case data_series has datetime or similar as index
        # interpolation works in some pandas/numpy combinations, in some not
        try:
            # if clean_mask is a Series, bring into the right order and make array ...
            clean_mask = clean_mask[data.index].to_numpy()
        except TypeError:
            pass

        # ... as we change the index of data now
        data.index = pd.Index([i.timestamp() for i in data.index])

    data[clean_mask] = np.NaN

    try:
        # 'interpolate' fills nan based on index/values of neighbours
        data = data.interpolate("index").fillna(method="bfill").fillna(method="ffill")
    except TypeError as e:
        raise TypeError("Interpolation failed. "
                        "Usually due to a dtype format that is not properly recognized.") from e

    shift = -int((length-1)/2)  # Shift average to middle value

    # calculate mean and std, fill NaNs at the ends
    data_mav = data.rolling(length).mean().shift(shift).fillna(
        method="bfill").fillna(method="ffill")
    std_mav = data.rolling(length).std().shift(shift).fillna(
        method="bfill").fillna(method="ffill")

    if is_datetime_index:
        # restore index from input series
        data_mav.index = data_series.index
        std_mav.index = data_series.index

    return data_mav, std_mav


def _is_almost_empty_mask(mask, av_length):
    """Checks if masked data could be used to calculate moving average."""
    if sum(mask) <= av_length:
        raise ValueError("Too many points have been filtered. Maybe wrong filtering parameters?")
