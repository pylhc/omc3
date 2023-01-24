"""
Stats
-----

Helper module providing statistical methods to compute various weighted averages along specified
axis and their errors as well as unbiased error estimator of infinite normal distribution from
finite-sized sample.

- TODO: use weighted average and its error in circular calculations
- TODO: write tests
- TODO: LOGGER or Raising error and warnings?
- TODO: if zeros or nans occur in errors, fallback to uniform weights only in affected cases
"""
import numpy as np
from scipy.special import erf
from scipy.stats import t

from omc3.definitions.constants import PI2, PI2I
from omc3.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)

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


def circular_nanmean(data, period=PI2, errors=None, axis=None):
    """"Wrapper around circular_mean with added nan handling"""
    return circular_mean(data=np.ma.array(data, mask=np.isnan(data)),
                         period=period,
                         errors= None if errors is None else np.ma.array(errors, mask=np.isnan(data)),
                         axis=axis)


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


def circular_nanerror(data, period=PI2, errors=None, axis=None, t_value_corr=True):
    """"Wrapper around circular_error with added nan handling"""
    return circular_error(data=np.ma.array(data, mask=np.isnan(data)),
                          period=period,
                          errors=None if errors is None else np.ma.array(errors, mask=np.isnan(data)),
                          axis=axis,
                          t_value_corr=t_value_corr)


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


def weighted_nanmean(data, errors=None, axis=None):
    """"Wrapper around weighted_mean with added nan handling"""
    return weighted_mean(data=np.ma.array(data, mask=np.isnan(data)),
                         errors=None if errors is None else np.ma.array(errors, mask=np.isnan(data)),
                         axis=axis)


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
    This is similar to calculating the standard deviation on the data,
    but with both, the average to which the deviation is calculated,
    as well as then the averaging over the deviations weighted by
    weights based on the errors.

    In addition, the weighted variance is unbiased by an unbias-factor
    n / (n-1), where n is the :meth:`omc3.utils.stats.effective_sample_size` .
    Additionally, a (student) t-value correction can be performed (done by default)
    which corrects the estimate for small data sets.

    Parameters:
        data: array-like
            Contains the data on which the weighted error on the average is calculated.
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
    weighted_average = weighted_average.reshape(_get_shape(data.shape, axis))
    (sample_variance, sum_of_weights) = np.ma.average(
        np.square(np.abs(data - weighted_average)),
        weights=weights, axis=axis, returned=True
    )
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


def weighted_nanrms(data, errors=None, axis=None):
    """"Wrapper around weigthed_rms with added nan handling"""
    return weighted_rms(data=np.ma.array(data, mask=np.isnan(data)),
                        errors=None if errors is None else np.ma.array(errors, mask=np.isnan(data)),
                        axis=axis)


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
        LOGGER.warning("NaNs found, weights are not used.")
        return None
    if np.any(np.logical_not(errors)):
        LOGGER.warning("Zeros found, weights are not used.")
        return None
    return 1 / np.square(errors * PI2 / period)


def effective_sample_size(data, weights, axis=None):
    r"""
    Computes effective sample size of weighted data along specified axis,
    the minimum value returned is 2 to avoid non-reasonable error blow-up.
    
    It is calculated via Kish's approximate formula 
    from the (not necessarily normalized) weights :math:`w_i` (see wikipedia):
    
    .. math::

        n_\mathrm{eff} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}

    What it represents:
    "In most instances, weighting causes a decrease in the statistical significance of results. 
    The effective sample size is a measure of the precision of the survey 
    (e.g., even if you have a sample of 1,000 people, an effective sample size of 100 would indicate 
    that the weighted sample is no more robust than a well-executed un-weighted 
    simple random sample of 100 people)." -
    https://wiki.q-researchsoftware.com/wiki/Weights,_Effective_Sample_Size_and_Design_Effects

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
    r"""
    Computes a correction factor to unbias variance of weighted average of data along specified axis,
    e.g. transform the standard deviation 1

    .. math::

        \sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - x_\mathrm{mean})^2

    into an un-biased estimator

    .. math::

        \sigma^2 = \frac{1}{N-1} \sum_{i=1}^N (x_i - x_\mathrm{mean})^2

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
    Calculates the multiplicative correction factor to determine standard deviation of
    a normally distributed quantity from standard deviation of its finite-sized sample.
    The minimum allowed sample size is 2 to avoid non-reasonable error blow-up
    for smaller sample sizes 2 is used instead.

    Note (jdilly): In other words, this transforms the area of 1 sigma under
    the given student t distribution to the 1 sigma area of a normal distribution
    (this transformation is where the ``CONFIDENCE LEVEL`` comes in).
    I hope that makes the intention more clear.

    Args:
        sample_size: array-like

    Returns:
        multiplicative correction factor(s) of same shape as sample_size.
        Can contain nans.
    """
    return t.ppf(CONFIDENCE_LEVEL, np.where(sample_size > 2, sample_size, 2) - 1)
