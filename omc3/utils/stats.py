"""
Module utils.stats
-------------------

Created on 03.07.18

:author: Lukas Malina

Provides statistical methods to compute:
    various weighted averages along specified axis and their errors
    unbiased error estimator of infinite normal distribution from finite-sized sample

TODO use weighted average and its error in circular calculations
TODO write tests
TODO LOGGER or Raising error and warnings?
TODO if zeros or nans occur in errors, fallback to uniform weights only in affected cases

"""
import numpy as np
from scipy.special import erf
from scipy.stats import t

PI2 = 2 * np.pi
PI2I = PI2 * 1j
CONFIDENCE_LEVEL = (1 + erf(1 / np.sqrt(2))) / 2


def circular_mean(data, period=PI2, errors=None, axis=None):
    """
    Computes weighted circular average along the specified axis.

    Parameters:
        data: array-like
            Contains the data to be averaged
        period: scalar, optional, default (2 * np.pi)
            Periodicity period of data
        errors: array-like, optional
            Contains errors associated with the values in data, it is used to calculated weights
        axis: int or tuple of ints, optional
            Axis or axes along which to average data

    Returns:
        Returns the weighted circular average along the specified axis.
    """

    phases = data * PI2I / period
    weights = weights_from_errors(errors, period=period)

    return np.angle(np.average(np.exp(phases), axis=axis, weights=weights)) * period / PI2


def circular_error(data, period=PI2, errors=None, axis=None, t_value_corr=True):
    """
    Computes error of weighted circular average along the specified axis.

    Parameters:
        data: array-like
            Contains the data to be averaged
        period: scalar, optional
            Periodicity period of data, default is (2 * np.pi)
        errors: array-like, optional
            Contains errors associated with the values in data, it is used to calculated weights
        axis: int or tuple of ints, optional
            Axis or axes along which to average data
        t_value_corr: bool, optional
            Species if the error is corrected for small sample size, default True

    Returns:
        Returns the error of weighted circular average along the specified axis.
        """
    phases = data * PI2I / period
    weights = weights_from_errors(errors, period=period)
    complex_phases = np.exp(phases)
    complex_average = np.average(complex_phases, axis=axis, weights=weights)

    (sample_variance, sum_of_weights) = np.average(
            np.square(np.abs(complex_phases - complex_average.reshape(_get_shape(
                    complex_phases.shape, axis)))), weights=weights, axis=axis, returned=True)
    if weights is not None:
        sample_variance = sample_variance + 1. / sum_of_weights
    error_of_complex_average = np.sqrt(sample_variance * unbias_variance(data, weights, axis=axis))
    phase_error = np.nan_to_num(error_of_complex_average / np.abs(complex_average))
    if t_value_corr:
        phase_error = phase_error * t_value_correction(effective_sample_size(data, weights, axis=axis))
    return np.where(phase_error > 0.25 * PI2, 0.3 * period, phase_error * period / PI2)


def weighted_mean(data, errors=None, axis=None):
    """
    Computes weighted average along the specified axis.

    Parameters:
        data: array-like
            Contains the data to be averaged
        errors: array-like, optional
            Contains errors associated with the values in data, it is used to calculated weights
        axis: int or tuple of ints, optional
            Axis or axes along which to average data

    Returns:
        Returns the weighted average along the specified axis.
    """
    weights = weights_from_errors(errors)
    return np.average(data, axis=axis, weights=weights)


def _get_shape(orig_shape, axis):
    new_shape = np.array(orig_shape)
    if axis is None:
        new_shape[:] = 1
    else:
        new_shape[np.array(axis)] = 1
    return tuple(new_shape)


def weighted_error(data, errors=None, axis=None, t_value_corr=True):
    """
    Computes error of weighted average along the specified axis.

    Parameters:
        data: array-like
            Contains the data to be averaged
        errors: array-like, optional
            Contains errors associated with the values in data, it is used to calculated weights
        axis: int or tuple of ints, optional
            Axis or axes along which to average data
        t_value_corr: bool, optional
            Species if the error is corrected for small sample size, default True

    Returns:
        Returns the error of weighted average along the specified axis.
    """
    weights = weights_from_errors(errors)
    weighted_average = np.average(data, axis=axis, weights=weights)
    (sample_variance, sum_of_weights) = np.average(np.square(np.abs(data - weighted_average.reshape(
            _get_shape(data.shape, axis)))), weights=weights, axis=axis, returned=True)
    if weights is not None:
        sample_variance = sample_variance + 1 / sum_of_weights
    error = np.nan_to_num(np.sqrt(sample_variance * unbias_variance(data, weights, axis=axis)))
    if t_value_corr:
        error = error * t_value_correction(effective_sample_size(data, weights, axis=axis))
    return error


def weighted_rms(data, errors=None, axis=None):
    """
    Computes weighted root mean square along the specified axis.

    Parameters:
        data: array-like
            Contains the data to be averaged
        errors: array-like, optional
            Contains errors associated with the values in data, it is used to calculated weights
        axis: int or tuple of ints, optional
            Axis or axes along which to average data

    Returns:
        Returns weighted root mean square along the specified axis.
    """
    weights = weights_from_errors(errors)
    return np.sqrt(np.average(np.square(data), weights=weights, axis=axis))


def weights_from_errors(errors, period=PI2):
    """
    Computes weights from measurement errors, weights are not output if errors contain zeros or nans

    Parameters:
        errors: array-like
            Contains errors which are used to calculated weights
        period: scalar, optional
            Periodicity period of data, default is (2 * np.pi)

    Returns:
        Returns the error of weighted circular average along the specified axis.
    """
    if errors is None:
        return None
    if np.any(np.isnan(errors)):
        # LOGGER.warning("Nans found, weights are not used.")
        return None
    if np.any(np.logical_not(errors)):
        # LOGGER.warning("Zeros found, weights are not used.")
        return None
    return 1 / np.square(errors * PI2 / period)


def effective_sample_size(data, weights, axis=None):
    """
    Computes effective sample size of weighted data along specifies axis,
    the minimum value returned is 2 to avoid non-reasonable error blow-up

    Parameters:
        data: array-like
        weights: array-like
            Contains weights associated with the values in data
        axis: int or tuple of ints, optional
            Axis or axes along which the effective sample size is computed

    Returns:
        Returns the error of weighted circular average along the specified axis.
    """
    if weights is None:
        sample_size = np.sum(np.ones(data.shape), axis=axis)
    else:
        sample_size = np.square(np.sum(weights, axis=axis)) / np.sum(np.square(weights), axis=axis)
    return np.where(sample_size > 2, sample_size, 2)


def unbias_variance(data, weights, axis=None):
    """
    Computes a correction factor to unbias variance of weighted average of data along specified axis

    Parameters:
        data: array-like
        weights: array-like
            Contains weights associated with the values in data
        axis: int or tuple of ints, optional
            Axis or axes along which the effective sample size is computed

    Returns:
        Returns the error of weighted circular average along the specified axis.
    """
    sample_size = effective_sample_size(data, weights, axis=axis)
    try:
        return sample_size / (sample_size - 1)
    except ZeroDivisionError or RuntimeWarning:
        return np.zeros(sample_size.shape)


def t_value_correction(sample_size):
    """
    Calculates the multiplicative correction factor to determine standard deviation of normally
    distributed quantity from standard deviation of its finite-sized sample
    the minimum allowed sample size is 2 to avoid non-reasonable error blow-up
    for smaller sample sizes 2 is used instead

    Args:
        sample_size: array-like

    Returns:
        multiplicative correction factor(s) of same shape as sample_size
            can contain nans
    """
    return t.ppf(CONFIDENCE_LEVEL, np.where(sample_size > 2, sample_size, 2) - 1)
