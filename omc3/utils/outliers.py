r"""
Module utils.outliers
----------------------

Created on 08/05/17

:author: Lukas Malina
"""
import numpy as np
from scipy.stats import t


# nsig: Limit for not being cleaned
def get_filter_mask(data, x_data=None, limit=0.0, niter=20, nsig=None, mask=None):
    """
    It filters the array of values which are meant to be constant
    or a linear function of the other array if that is provided
    Returns a filter mask for the original array
    """
    if x_data is not None:
        if not len(data) == len(x_data):
            raise ValueError("Datasets are not equally long.")
    
    # To fulfill the condition for the first iteration:
    if mask is not None:
        if not len(data) == len(mask):
            raise ValueError("Mask is not equally long as dataset.")
    else:
        mask = np.ones_like(data, dtype=bool)
    if nsig is None:
        nsig = _get_significance_cut_from_length(np.sum(mask))

    prevlen = np.sum(mask) + 1
    for _ in range(niter):
        if not ((np.sum(mask) < prevlen) and
                (np.sum(mask) > 2)):
            break
        prevlen = np.sum(mask)
        if x_data is None:
            y, y_orig = _get_data(mask, data)
        else:
            y, y_orig = _get_data_without_slope(mask, x_data, data)
        avg, std = _get_moments(y)
        mask = np.logical_and(mask,np.abs(y_orig - avg) < np.max([limit, nsig * std]))
    return mask


def _get_moments(data):
    return np.mean(data), np.std(data)


def _get_data_without_slope(mask, x, y):
    m, b = np.polyfit(x[mask], y[mask], 1)
    return y[mask] - b - m * x[mask], y - b - m * x


def _get_data(mask, data):
    return data[mask], data


# Set the sigma cut, that expects 1 value to be cut
# if it is sample of normal distribution
def _get_significance_cut_from_length(length):
    return t.ppf([1 - 0.5 / float(length)], length)
