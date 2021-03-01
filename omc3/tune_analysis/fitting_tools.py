"""
Fitting Tools
-------------

This module contains fitting functionality for ``tune_analysis``.
It provides tools for fitting functions, mainly via odr.

Important Convention:
    The beta-parameter in the ODR models go upwards with order, i.e.
    |  beta[0] = y-Axis offset
    |  beta[1] = slope
    |  beta[2] = quadratic term
    |  etc.
"""
import numpy as np
from scipy.odr import RealData, Model, ODR

from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


# ODR ###################################################################


def get_poly_fun(order):
    """Returns the function of polynomial order. (is this pythonic enough?)."""
    def poly_func(beta, x):
        return sum(beta[i] * np.power(x, i) for i in range(order+1))
    return poly_func


def do_odr(x, y, xerr, yerr, order):
    """
    Returns the odr fit.

    Args:
        x: `Series` of x data.
        y: `Series` of y data.
        xerr: `Series` of x data errors.
        yerr: `Series` of y data errors.
        order: fit order, ``1`` for linear, ``2`` for quadratic.

    Returns: Odr fit. Betas order is index = coefficient of same order.
    """
    LOG.debug("Starting ODR fit.")
    fit_np = np.polyfit(x, y, order)[::-1]  # using polyfit as starting parameters
    LOG.debug(f"ODR fit input (from polyfit): {fit_np}")
    odr = ODR(data=RealData(x=x, y=y, sx=xerr, sy=yerr),
              model=Model(get_poly_fun(order)),
              beta0=fit_np)
    odr_fit = odr.run()
    logging_tools.odr_pprint(LOG.info, odr_fit)
    return odr_fit
