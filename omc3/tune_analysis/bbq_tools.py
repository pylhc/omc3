"""
BBQ Tools
---------

Tools to handle BBQ data.
"""
from typing import Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from omc3.utils import logging_tools
from omc3.utils.outliers import get_filter_mask
from dataclasses import dataclass


LOG = logging_tools.get_logger(__name__)


@dataclass
class OutlierFilterOpt:
    """Options for moving average with outlier filter.

    Args:
        window: length of the averaging window.
        limit: cleaning limit, i.e. points with deviating less than this limit from the average will not be cleaned
     """
    window: int
    limit: float = 0.


@dataclass
class MinMaxFilterOpt:
    """Options for moving average with min/max filter.

    Args:
        window: length of the averaging window.
        min: minimum value (for filtering).
        max: maximum value (for filtering).
        fine_window: length of the averaging window for fine cleaning.
        fine_cut: allowed deviation for fine cleaning.
    """
    window: int
    min: float = None
    max: float = None
    fine_window: int = None
    fine_cut: float = None


FilterOpts = Union[OutlierFilterOpt, Tuple[MinMaxFilterOpt, MinMaxFilterOpt]]


def get_moving_average(data_series: pd.Series, filter_opt: MinMaxFilterOpt) -> Tuple[pd.Series, pd.Series, ArrayLike]:
    """
    Get a moving average of the ``data_series`` over ``length`` entries. The data can be filtered
    beforehand. The values are shifted, so that the averaged value takes ceil((length-1)/2)
    values previous and floor((length-1)/2) following values into account.

    Args:
        data_series: `Series` of data.
        filter_opt: Options for the filtering, see `:class:`omc3.tune_analysis.bbq_tools.MinMaxFilterOpt`.

    Returns:
        A filtered and averaged `Series`, another `Series` with the error and the mask used for filtering data.
    """
    LOG.debug(f"Cutting data and calculating moving average of length {filter_opt.window:d}.")

    if bool(filter_opt.fine_window) != bool(filter_opt.fine_cut):
        raise NotImplementedError("To activate fine cleaning, both "
                                  "'fine_window' and 'fine_cut' are needed.")

    if filter_opt.min is not None:
        min_mask = data_series <= filter_opt.min
    else:
        min_mask = np.zeros(data_series.size, dtype=bool)

    if filter_opt.max is not None:
        max_mask = data_series >= filter_opt.max
    else:
        max_mask = np.zeros(data_series.size, dtype=bool)

    cut_mask = min_mask | max_mask | data_series.isna()
    _is_almost_empty_mask(~cut_mask, filter_opt.window)
    data_mav, err_mav = _get_interpolated_moving_average(data_series, cut_mask, filter_opt.window)

    if filter_opt.fine_window is not None:
        min_mask = data_series <= (data_mav - filter_opt.fine_cut)
        max_mask = data_series >= (data_mav + filter_opt.fine_cut)
        cut_mask = min_mask | max_mask | data_series.isna()
        _is_almost_empty_mask(~cut_mask, filter_opt.fine_window)
        data_mav, err_mav = _get_interpolated_moving_average(data_series, cut_mask, filter_opt.fine_window)

    return data_mav, err_mav, ~cut_mask


def clean_outliers_moving_average(data_series: pd.Series, filter_opt: OutlierFilterOpt) -> Tuple[pd.Series, pd.Series, NDArray[bool]]:
    """
    Get a moving average of the ``data_series`` over ``length`` entries, by means of
    :func:`outlier filter <omc3.utils.outliers.get_filter_mask>`.
    The values are shifted, so that the averaged value takes ceil((length-1)/2) values previous
    and floor((length-1)/2) following values into account.

    Args:
        data_series: `Series` of data.
        filter_opt: Options for the filtering, see :class:`omc3.tune_analysis.bbq_tools.OutlierFilterOpt`.
    """
    LOG.debug(f"Filtering and calculating moving average of length {filter_opt.window:d}.")
    window, limit = filter_opt.window, filter_opt.limit
    init_mask = ~data_series.isna()
    mask = init_mask.copy()
    for i in range(len(data_series)-filter_opt.window):
        mask[i:i+window] &= get_filter_mask(data_series[i:i+window], limit=limit, mask=init_mask[i:i+window])

    _is_almost_empty_mask(mask, window)
    data_mav, err_mav = _get_interpolated_moving_average(data_series, ~mask, window)
    return data_mav, err_mav, mask


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

    data[clean_mask] = np.nan

    try:
        # 'interpolate' fills nan based on index/values of neighbours
        data = data.interpolate("index").bfill().ffill()
    except TypeError as e:
        raise TypeError("Interpolation failed. "
                        "Usually due to a dtype format that is not properly recognized.") from e

    shift = -int((length-1)/2)  # Shift average to middle value

    # calculate mean and fill NaNs at the ends
    data_mav = data.rolling(length).mean().shift(shift).bfill().ffill()

    # calculate deviation to the moving average and fill NaNs at the ends
    std_mav = np.sqrt(((data-data_mav)**2).rolling(length).mean().shift(shift).bfill().ffill())
    err_mav = std_mav / np.sqrt(length)

    if is_datetime_index:
        # restore index from input series
        data_mav.index = data_series.index
        std_mav.index = data_series.index
        err_mav.index = data_series.index

    return data_mav, err_mav


def _is_almost_empty_mask(mask, av_length):
    """Checks if masked data could be used to calculate moving average."""
    if sum(mask) <= av_length:
        raise ValueError("Too many points have been filtered. Maybe wrong filtering parameters?")
