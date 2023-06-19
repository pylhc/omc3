"""
Fitting Tools
-------------

This module contains fitting functionality for ``tune_analysis``.
It provides tools for fitting functions, mainly via odr.

"""
from collections import namedtuple
from typing import Sequence, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike
from scipy.odr import RealData, Model, ODR
from scipy.optimize import curve_fit

from omc3.utils import logging_tools
from omc3.tune_analysis.constants import FakeOdrOutput, AmpDetData

LOG = logging_tools.get_logger(__name__)


# ODR ###################################################################


def get_poly_fun(order: int):
    """Returns the function of polynomial order. (is this pythonic enough?)."""
    def poly_func(beta, x):
        return sum(beta[i] * np.power(x, i) for i in range(order+1))
    return poly_func


def do_odr(x: pd.Series, y: pd.Series, xerr: pd.Series, yerr: pd.Series, order: int):
    """
    Returns the odr fit.

    Important Convention:
    The beta-parameter in the ODR models go upwards with order, i.e.

    |  beta[0] = y-Axis offset
    |  beta[1] = slope
    |  beta[2] = quadratic term
    |  etc.

    Args:
        x: `Series` of x data.
        y: `Series` of y data.
        xerr: `Series` of x data errors.
        yerr: `Series` of y data errors.
        order: fit order, ``1`` for linear, ``2`` for quadratic.

    Returns: Odr fit. Betas order is index = coefficient of same order.
    """
    LOG.debug("Starting ODR fit.")

    # Poly-Fit for starting point ---
    fit_np = Polynomial.fit(x, y, deg=order).convert()
    LOG.debug(f"ODR fit input (from polynomial fit): {fit_np}")

    # Actual ODR ---
    xerr, yerr = _check_exact_zero_errors(xerr, yerr)
    odr = ODR(data=RealData(x=x, y=y, sx=xerr, sy=yerr),
              model=Model(get_poly_fun(order)),
              beta0=fit_np.coef)
    odr_fit = odr.run()
    logging_tools.odr_pprint(LOG.info, odr_fit)
    return odr_fit


# 2D-Kick ODR ##################################################################

INPUT_ORDER = "qx0", "qy0", "dqx/dex", "dqy/dey",  "dq(x,y)/de(y,x)"


def first_order_detuning_2d(beta: Sequence, x: ArrayLike) -> ArrayLike:
    """ Calculates the 2D tune array (Qx, Qy)
        Qx = qx0 + dqx/dex * ex + dqx/dey * ey
        Qy = qy0 + dqy/dex * ex + dqy/dey * ey

    Args:
        beta: length 5 tune coefficients in order `INPUT_ORDER`
              0: qx0, 1: qy0, 2: xx, 3: yy, 4: xy/yx
        x: array size 2xN, [[ex1, ex2, ...],[ey1, ey2,...]]

    Returns:
        np.array: 2xN [[Qx1, Qx2, ...],[Qy1, Qy2, ...]]
    """
    return np.array([beta[0] + beta[2] * x[0] + beta[4] * x[1],
                     beta[1] + beta[4] * x[0] + beta[3] * x[1]])


def first_order_detuning_2d_jacobian(beta: Sequence, x: ArrayLike) -> ArrayLike:
    """ Jacobian of the 2D tune array:
        [[dqx/dex, dqx/dey ],
         [dqy/dex, dqy/dey ]]

    Args:
        beta: length 5 tune coefficients in order `INPUT_ORDER`
              0: qx0, 1: qy0, 2: xx, 3: yy, 4: xy/yx
        x: length 2 Sequence, (ex, ey)

    Returns:
        np.array: size 2x2xlen(x) containing the Jacobian of the 2d-fit function.
    """
    return np.dstack([np.array([[beta[2], beta[4]],
                     [beta[4], beta[3]]])] * len(x[0]))


def map_odr_fit_to_planes(odr_fit) -> Dict[str, Dict[str, FakeOdrOutput]]:
    """ Maps the calculated odr fit to fake odr fits with `beta` and `sd_beta`
    attributes. These would be the results of first-order amplitude detuning
    odr-fits when done independently by tune and kick plane.

    Returns: Dict[str, Dict[str: odr_fit]] of ODR fits, where the inner
             string gives the kick-plane, the outer the tune-plane..

    """
    def get_fit(a: int, b: int):
        return FakeOdrOutput(
            beta=[odr_fit.beta[a], odr_fit.beta[b]],
            sd_beta=[odr_fit.sd_beta[a], odr_fit.sd_beta[b]]
        )

    return {
        "X": {
            "X": get_fit(0, 2),
            "Y": get_fit(0, 4),
        },
        "Y": {
            "X": get_fit(1, 4),
            "Y": get_fit(1, 3),
        },
    }


def do_2d_kicks_odr(x: ArrayLike, y: ArrayLike, xerr: ArrayLike, yerr: ArrayLike):
    """
    Returns the odr fit.

    Args:
        x: `Array` of x data (2xN).
        y: `Array` of y data (2xN).
        xerr: `Array` of x data errors (2xN).
        yerr: `Array` of y data errors (2xN).

    Returns: Dict[str, Dict[str: odr_fit]] of Odr fits, where the inner
             string gives the kick-plane, the outer the tune-plane..
    """
    LOG.debug("Starting ODR fit.")

    x, y, xerr, yerr = _filter_nans(x, y, xerr, yerr)

    # Curve-Fit for starting point ---
    curve_fit_fun = lambda v, *args: first_order_detuning_2d(args, v).ravel()
    beta, beta_cov = curve_fit(f=curve_fit_fun, xdata=x, ydata=y.ravel(), p0=[0]*5)

    res_str = ",\n".join([f"{n:>16} = {b:9.3g}" for n, b in zip(INPUT_ORDER, beta)])
    LOG.info(f"\nDetuning estimate without errors (curve fit):\n{res_str}\n")

    # Actual ODR ---
    xerr, yerr = _check_exact_zero_errors(xerr, yerr)
    odr = ODR(data=RealData(x=x, y=y, sx=xerr, sy=yerr),
              # model=Model(first_order_detuning_2d),
              model=Model(first_order_detuning_2d, fjacd=first_order_detuning_2d_jacobian),
              beta0=beta)
    odr_fit = odr.run()
    logging_tools.odr_pprint(LOG.debug, odr_fit)

    res_str = ",\n".join([f"{n:>16} = {b:9.3g} +- {e:8.3g}" for n, b, e in zip(INPUT_ORDER, odr_fit.beta, odr_fit.sd_beta)])
    LOG.info(f"\nDetuning estimate with errors (odr):\n{res_str}\n")
    return map_odr_fit_to_planes(odr_fit)


def _filter_nans(*args: ArrayLike) -> List[ArrayLike]:
    """Remove all data points containing a NaN.
    Assumes input arrays are all of shape 2xN.
    TODO:
    As this is not done in plotting, points might be plotted,
    that have not been used for fitting.
    """
    a = np.array(args)
    a = a[:, :, ~np.isnan(a).any(axis=0).any(axis=0)]
    return list(a)


def _check_exact_zero_errors(xerr: ArrayLike, yerr: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Check if errors are exact zero and replace with minimum error, if so.
    Done because ODR crashes on exact zero error-bars.

    Beware that the output will always be np.arrays, even if the input is pd.Series.
    """
    def check_exact_zero_per_plane(err, plane):
        if (err != 0).all():  # no problem
            return err

        # best way to work with array and series?
        minval = np.where(err == 0, np.inf, err).min()  # assumes all values >=0

        if np.isinf(minval):
            raise ValueError(f"All errors are exactly zero in the {plane} plane."
                             f" ODR cannot be performed.")
        LOG.warning(f"Exact zeros in {plane} errors found."
                    f" Replaced by {minval} (the minimum value) to be able to perform ODR.")
        return np.where(err == 0, minval, err)

    xerr = check_exact_zero_per_plane(xerr, "horizontal")
    yerr = check_exact_zero_per_plane(yerr, "vertical")
    return xerr, yerr



