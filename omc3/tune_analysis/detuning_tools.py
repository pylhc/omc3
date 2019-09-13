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
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import RealData, Model, ODR

from tune_analysis import constants as const
from utils import logging_tools
from plotshop import plot_style as ps

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
    print_odr_result(LOG.debug, odr_fit)
    return odr_fit


def print_odr_result(printer, odr_out):
        """ Logs the odr output results.

        Adapted from odr_output pretty print.
        """
        printer('Beta: {}'.format(odr_out.beta).replace("\n", ""))
        printer('Beta Std Error: {}'.format(odr_out.sd_beta).replace("\n", ""))
        printer('Beta Covariance: {}'.format(odr_out.cov_beta).replace("\n", ""))
        if hasattr(odr_out, 'info'):
            printer('Residual Variance: {}'.format(odr_out.res_var).replace("\n", ""))
            printer('Inverse Condition #: {}'.format(odr_out.inv_condnum).replace("\n", ""))
            printer('Reason(s) for Halting:')
            for r in odr_out.stopreason:
                printer('  {}'.format(r).replace("\n", ""))


def plot_linear_odr(ax, odr_fit, lim):
    """ Adds a linear odr fit to axes.
    """
    x_fit = np.linspace(lim[0], lim[1], 2)
    line_fit = odr_fit.beta[1] * x_fit
    ax.plot(x_fit, line_fit, marker="", linestyle='--', color='k',
            label='${:.4f}\, \pm\, {:.4f}$'.format(odr_fit.beta[1], odr_fit.sd_beta[1]))


# General Plotting #############################################################


def plot_detuning(x, y, xerr, yerr, labels, xmin=None, xmax=None, ymin=None, ymax=None,
                  odr_fit=None, odr_plot=plot_linear_odr, output=None, show=True):
    """ Plot amplitude detuning.

    Args:
        x: Action data.
        y: Tune data.
        xerr: Action error.
        yerr: Tune error.
        xmin: Lower action range to plot.
        xmax: Upper action range to plot.
        ymin: Lower tune range to plot.
        ymax: Upper tune range to plot.
        odr_fit: results of the odr-fit (e.g. see do_linear_odr)
        odr_plot: function to plot odr_fit (e.g. see plot_linear_odr)
        labels: Dict of labels to use for the data ("line"), the x-axis ("x") and the y-axis ("y")
        output: Output file of the plot.
        show: Show the plot in window.

    Returns:
        Plotted Figure
    """
    ps.set_style("standard",
                 {u"lines.marker": u"o",
                  u"lines.linestyle": u"",
                  u'figure.figsize': [9.5, 4],
                  }
                 )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    xmin = 0 if xmin is None else xmin
    xmax = max(x + xerr) * 1.05 if xmax is None else xmax

    offset = 0
    if odr_fit:
        odr_plot(ax, odr_fit, lim=[xmin, xmax])
        offset = odr_fit.beta[0]

    ax.errorbar(x, y - offset, xerr=xerr, yerr=yerr, label=labels.get("line", None))

    # labels
    default_labels = const.get_paired_lables("", "")
    ax.set_xlabel(labels.get("x", default_labels[0]))
    ax.set_ylabel(labels.get("y", default_labels[1]))

    # limits
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)

    # lagends
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=2,)
    ax.ticklabel_format(style="sci", useMathText=True, scilimits=(-3, 3))
    fig.tight_layout()
    fig.tight_layout()  # needs two calls for some reason to look great

    if output:
        fig.savefig(output)
        ps.set_name(os.path.basename(output))

    if show:
        plt.draw()

    return fig

