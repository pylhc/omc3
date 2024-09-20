"""
Outliers
--------

Helper functions for outlier detection.
"""
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import t

from omc3.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def get_filter_mask(data: ArrayLike, x_data: ArrayLike = None, limit: float = 0.0,
                    niter: int = 20, nsig: int = None, mask: ArrayLike = None) -> ArrayLike:
    r"""
    Filters the array of values which are meant to be constant or a linear function of the x-data
    array if that is provided, by checking how many sigmas they are deviating from the average.

    The outlier filtering function is utilized at multiple stages of the data analysis.
    It removes data points in the tails of the measured distribution,
    which are too populated due to the finite sample size,
    assuming a normal distribution specified by measured mean and standard
    deviation of the given data.
    A data point, outside of a user-specified limit,
    is removed if there is less than a 50% chance
    that it is from the specified normal distribution.

    In particular:
    The function gets an array :math:`y` of data of length :math:`n_y`,
    which can be any scalar data and is currently used for e.g. tune data
    measured per BPM or a time series from the BBQ.
    In addition, a cleaning ``limit`` can be given,
    inside which the data points are always kept.
    Further an optional array :math:`x` (via the parameter ``x_data``)
    can be given, in which case a linear fit :math:`y = ax + c` is attempted
    to remove any slope on the data, before calculating momenta on the data.
    An upper boundary in :math:`n_\sigma` is then determined from the
    percent point function (ppf) of a Student's t-distribution,
    given :math:`n_x` degrees of freedom at :math:`100\% - 50\% / n_y`.
    Iteratively, mean :math:`\left< y \right>` and standard deviation :math:`\sigma`
    of the data is calculated and only data within :math:`n_\sigma \cdot \sigma`,
    or the given cleaning limit, around :math:`\left< y \right>`
    is kept for the next iteration. The loop stops either after 20 iterations,
    when there is no more data outside the boundaries or
    when there are less than three data points remaining.
    
    Returns a filter mask for the original array (meaning ``True`` for elements that should be kept).

    Args:
         data (ArrayLike): Data to filter.
         x_data (ArrayLike): Additional x-data. If given a linear function is fit on the data,
                             and the outliers filtered on the difference to this function.
                             Otherwise, the distribution of data is used directly for filtering.
         limit (float): Only data deviating more than this limit from the average is filtered.
         niter (int): Maximum number of filter iterations to do.
                      Iterations are interrupted if the number of datapoints did not shrink
                      in the last iteration or if there is only two or less data-points left.
         nsig (int): number of sigmas within the data points are considered okay, outside they are
                     considered outliers. If not given, nsigmas is got from the
                     1-0.5/length(data) percentile of a student-t distribution.
         mask (ArrayLike): Boolean mask of data-points to consider.
                           If ``None``, all data is considered.

    Returns:
        ArrayLike: Boolean array containing ``True`` entries for data-points that are good,
                   and ``False`` for the ones that should be filtered.

    """
    LOGGER.debug("Creating Outlier-Filter mask.")

    if x_data is not None:
        if not len(data) == len(x_data):
            raise ValueError("Datasets are not equally long.")
    
    if mask is not None:
        if not len(data) == len(mask):
            raise ValueError("Mask is not equally long as dataset.")
    else:
        mask = np.ones_like(data, dtype=bool)

    if nsig is None:
        nsig = _get_significance_cut_from_length(np.sum(mask))

    # Set number of remaining points to check decrease in loop
    n_previous = np.sum(mask) + 1  # initialization for the first check

    # Cleaning iteration
    for _ in range(niter):
        # check that number of points decreases and some are left
        n_current = np.sum(mask)
        if not ((n_current < n_previous) and (n_current > 2)):
            break
        n_previous = n_current

        # get the (remaining) data-points used to get moments from,
        # if x-data is given, fit and remove slope.
        if x_data is None:
            y, y_orig = data[mask], data
        else:
            y, y_orig = _get_data_without_slope(mask, x_data, data)

        # keep only points within nsig sigma of the remaining data.
        avg, std = np.mean(y), np.std(y)
        mask = np.logical_and(mask, np.abs(y_orig - avg) < np.max([limit, nsig * std]))
    else:
        LOGGER.debug("Outlier Filter loop exceeds maximum number of iterations."
                     " Current filter-mask will be used.")
    return mask


def _get_data_without_slope(mask, x, y):
    """ Remove the slope on the data by performing a linear fit. """
    import warnings
    try:
        from numpy.exceptions import RankWarning as IgnoredWarning
    except ImportError:  # it does not exist on numpy versions supporting 3.9
        IgnoredWarning = Warning

    # We filter out the "Polyfit may be poorly conditioned" warning that can happen
    # during the polyfit. We relay it as a logged message, which allows us to avoid
    # polluting the stderr and allows the user to not see it depending on log level
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("ignore", category=IgnoredWarning)
        m, b = np.polyfit(x[mask], y[mask], 1)

        # We log any captured warning at warning level
        for warning in records:
            LOGGER.warning(f"Polyfit warning: {warning.message}")

        return y[mask] - b - m * x[mask], y - b - m * x


def _get_significance_cut_from_length(length):
    """ Set the sigma cut, that expects one value to be cut
    if it is sample of normal distribution."""
    return t.ppf(1 - 0.5 / length, length)
