r"""
Outliers
--------

Helper functions for outlier detection.
"""
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import t


def get_filter_mask(data: ArrayLike, x_data: ArrayLike = None, limit: float = 0.0,
                    niter: int = 20, nsig: int = None, mask: ArrayLike = None) -> ArrayLike:
    """
    Filters the array of values which are meant to be constant or a linear function of the x-data
    array if that is provided, by checking how many sigmas they are deviating from the average.
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

    # To fulfill the condition for the first iteration:
    prevlen = np.sum(mask) + 1

    # Cleaning iteration
    for _ in range(niter):
        # check that number of points decreases and some are left
        if not ((np.sum(mask) < prevlen) and
                (np.sum(mask) > 2)):
            break
        prevlen = np.sum(mask)

        # get the (remaining) data-points used to get moments from,
        # if x-data is given, fit and remove slope.
        if x_data is None:
            y, y_orig = _get_data(mask, data)
        else:
            y, y_orig = _get_data_without_slope(mask, x_data, data)

        # keep only points within nsig sigma of the remaining data.
        avg, std = np.mean(y), np.std(y)
        mask = np.logical_and(mask, np.abs(y_orig - avg) < np.max([limit, nsig * std]))
    return mask


def _get_data_without_slope(mask, x, y):
    m, b = np.polyfit(x[mask], y[mask], 1)
    return y[mask] - b - m * x[mask], y - b - m * x


def _get_data(mask, data):
    return data[mask], data


# Set the sigma cut, that expects 1 value to be cut
# if it is sample of normal distribution
def _get_significance_cut_from_length(length):
    return t.ppf(1 - 0.5 / length, length)
