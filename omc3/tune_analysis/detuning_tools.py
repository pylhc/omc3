"""
Module tune_analysis.detuning_tools
-------------------------------------

Some tools for amplitude detuning, mainly plotting.

Important Convention:
    The beta-parameter in the ODR models go upwards with order, i.e.
    |  beta[0] = y-Axis offset
    |  beta[1] = slope
    |  beta[2] = quadratic term
    |  etc.

"""
import numpy as np
from scipy.odr import RealData, Model, ODR

from utils import logging_tools

LOG = logging_tools.get_logger(__name__)


# Linear ODR ###################################################################


def linear_model(beta, x):
    """ Return a linear model ``beta[0] + beta[1] * x``.

    Args:
        beta: beta[0] = y-offset
              beta[1] = slope
        x: x-value
    """
    return beta[0] + beta[1] * x


def do_linear_odr(x, y, xerr, yerr):
    """ Returns linear odr fit.

    Args:
        x: Series of x data
        y: Series of y data
        xerr: Series of x data errors
        yerr: Series of y data errors

    Returns: Linear odr fit. Betas see ``linear_model()``.
    """
    lin_model = Model(linear_model)
    data = RealData(x, y, sx=xerr, sy=yerr)
    odr_fit = ODR(data, lin_model, beta0=[0., 1.]).run()
    logging_tools.odr_pprint(LOG.debug, odr_fit)
    return odr_fit


