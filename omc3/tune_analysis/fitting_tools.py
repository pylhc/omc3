"""
Fitting Tools
-------------

This module contains fitting functionality for ``tune_analysis``.
It provides tools for fitting functions, mainly via odr.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import odrpack
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit

from omc3.tune_analysis.constants import FakeOdrOutput
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
    from logging import Logger

    import pandas as pd
    from numpy.typing import ArrayLike
    from odrpack.result import OdrResult

LOG: Logger = logging_tools.get_logger(__name__)


# ODR ###################################################################


def get_polynomial_function(order: int):
    """Returns the function of polynomial order."""

    def poly_func(x, beta):
        return sum(beta[i] * np.power(x, i) for i in range(order + 1))

    return poly_func


def do_odr(x: pd.Series, y: pd.Series, xerr: pd.Series, yerr: pd.Series, order: int) -> OdrResult:
    """
    Returns the odr fit from the provided data and associated error bars.

    Important Convention:
        The beta-parameter in the ODR models go upwards with order, i.e.

        |  beta[0] = y-Axis offset
        |  beta[1] = slope
        |  beta[2] = quadratic term
        |  etc.

    Args:
        x: `pandas.Series` of x data.
        y: `pandas.Series` of y data.
        xerr: `pandas.Series` of x data errors.
        yerr: `pandas.Series` of y data errors.
        order: fit order, ``1`` for linear, ``2`` for quadratic, etc.

    Returns:
        An `~odrpack.OdrResult` with the fitted coefficients in ``beta``,
        ordered by ascending polynomial degree.
    """
    LOG.debug("Starting ODR fit.")

    # Poly-Fit for starting point ---
    fit_np: Polynomial = Polynomial.fit(x, y, deg=order).convert()
    LOG.debug(f"ODR fit input (from polynomial fit): {fit_np}")

    # Actual ODR ---
    xerr, yerr = _check_exact_zero_errors(xerr, yerr)
    odr_fit: OdrResult = odrpack.odr_fit(
        f=get_polynomial_function(order),
        xdata=np.asarray(x, dtype=float),
        ydata=np.asarray(y, dtype=float),
        beta0=fit_np.coef,
        weight_x=1.0 / np.asarray(xerr, dtype=float) ** 2,
        weight_y=1.0 / np.asarray(yerr, dtype=float) ** 2,
    )
    logging_tools.odr_pprint(LOG.info, odr_fit)
    return odr_fit


# 2D-Kick ODR ##################################################################

INPUT_ORDER = "qx0", "qy0", "dqx/dex", "dqy/dey", "dq(x,y)/de(y,x)"


def first_order_detuning_2d(x: ArrayLike, beta: Sequence) -> ArrayLike:
    """
    Calculates the 2D tune array (Qx, Qy)
        Qx = qx0 + dqx/dex * ex + dqx/dey * ey
        Qy = qy0 + dqy/dex * ex + dqy/dey * ey

    Args:
        beta: length 5 tune coefficients in order `INPUT_ORDER`
              0: qx0, 1: qy0, 2: xx, 3: yy, 4: xy/yx
        x: array size 2xN, [[ex1, ex2, ...],[ey1, ey2,...]]

    Returns:
        np.array: 2xN [[Qx1, Qx2, ...],[Qy1, Qy2, ...]]
    """
    return np.array(
        [beta[0] + beta[2] * x[0] + beta[4] * x[1], beta[1] + beta[4] * x[0] + beta[3] * x[1]]
    )


def first_order_detuning_2d_jac_x(x: ArrayLike, beta: Sequence) -> np.ndarray:
    """Jacobian of `first_order_detuning_2d` w.r.t. x. Shape (q=2, m=2, n).

    Args:
        x: array size 2xN.
        beta: length 5 tune coefficients in order `INPUT_ORDER`.

    Returns:
        np.array of shape (2, 2, n).
    """
    return np.dstack([np.array([[beta[2], beta[4]], [beta[4], beta[3]]])] * len(x[0]))


def first_order_detuning_2d_jac_beta(x: ArrayLike, beta: Sequence) -> np.ndarray:
    """Jacobian of `first_order_detuning_2d` w.r.t. beta. Shape (q=2, npar=5, n).

    Args:
        x: array size 2xN.
        beta: length 5 tune coefficients in order `INPUT_ORDER`.

    Returns:
        np.array of shape (2, 5, n).
    """
    n = len(x[0])
    ones = np.ones(n)
    zeros = np.zeros(n)
    return np.array([
        [ones,  zeros, x[0],  zeros, x[1]],   # dQx/d(beta_i)
        [zeros, ones,  zeros, x[1],  x[0]],   # dQy/d(beta_i)
    ])


def map_odr_fit_to_planes(odr_fit) -> dict[str, dict[str, FakeOdrOutput]]:
    """
    Maps the calculated odr fit to fake odr fits with `beta` and `sd_beta`
    attributes. These would be the results of first-order amplitude detuning
    odr-fits when done independently by tune and kick plane.

    Returns:
        Dict[str, Dict[str: odr_fit]] of ODR fits, where the inner
        string gives the kick-plane, the outer the tune-plane.
    """

    def get_fit(a: int, b: int):
        return FakeOdrOutput(
            beta=[odr_fit.beta[a], odr_fit.beta[b]],
            sd_beta=[odr_fit.sd_beta[a], odr_fit.sd_beta[b]],
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
    def curve_fit_fun(v, *args):
        return first_order_detuning_2d(v, args).ravel()

    beta, _beta_cov = curve_fit(f=curve_fit_fun, xdata=x, ydata=y.ravel(), p0=[0] * 5)

    res_str = ",\n".join([f"{n:>16} = {b:9.3g}" for n, b in zip(INPUT_ORDER, beta)])
    LOG.info(f"\nDetuning estimate without errors (curve fit):\n{res_str}\n")

    # Actual ODR ---
    xerr, yerr = _check_exact_zero_errors(xerr, yerr)
    odr_fit: OdrResult = odrpack.odr_fit(
        f=first_order_detuning_2d,
        xdata=np.asarray(x, dtype=float),
        ydata=np.asarray(y, dtype=float),
        beta0=beta,
        weight_x=1.0 / np.asarray(xerr, dtype=float) ** 2,
        weight_y=1.0 / np.asarray(yerr, dtype=float) ** 2,
        jac_beta=first_order_detuning_2d_jac_beta,
        jac_x=first_order_detuning_2d_jac_x,
    )
    logging_tools.odr_pprint(LOG.debug, odr_fit)

    res_str = ",\n".join(
        [
            f"{n:>16} = {b:9.3g} +- {e:8.3g}"
            for n, b, e in zip(INPUT_ORDER, odr_fit.beta, odr_fit.sd_beta)
        ]
    )
    LOG.info(f"\nDetuning estimate with errors (odr):\n{res_str}\n")
    return map_odr_fit_to_planes(odr_fit)


def _filter_nans(*args: ArrayLike) -> list[ArrayLike]:
    """Remove all data points containing a NaN.
    Assumes input arrays are all of shape 2xN.
    TODO:
    As this is not done in plotting, points might be plotted,
    that have not been used for fitting.
    """
    a = np.array(args)
    a = a[:, :, ~np.isnan(a).any(axis=0).any(axis=0)]
    return list(a)


def _check_exact_zero_errors(
    xerr: pd.Series | np.ndarray, yerr: pd.Series | np.ndarray
) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]:
    """
    Check if errors are exact zero and replace with minimum error, if so.
    Done because ODR crashes on exact zero error-bars.
    """

    def check_exact_zero_per_plane(err: pd.Series | np.ndarray, plane: str) -> pd.Series | np.ndarray:
        if (err != 0).all():  # no problem
            return err

        # best way to work with array and series?
        minval = np.where(err == 0, np.inf, err).min()  # assumes all values >=0

        if np.isinf(minval):
            raise ValueError(
                f"All errors are exactly zero in the {plane} plane. ODR cannot be performed."
            )
        LOG.warning(
            f"Exact zeros in {plane} errors found."
            f" Replaced by {minval} (the minimum value) to be able to perform ODR."
        )
        return np.where(err == 0, minval, err)

    xerr = check_exact_zero_per_plane(xerr, "horizontal")
    yerr = check_exact_zero_per_plane(yerr, "vertical")
    return xerr, yerr
