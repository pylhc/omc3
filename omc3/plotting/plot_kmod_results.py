import tfs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING


from omc3.kmod.constants import BETASTAR, WAIST, BEAM, MDL, ERR

if TYPE_CHECKING:
    from generic_parser import DotDict


PARAM_BETA = "beta"
PARAM_WAIST = "waist"

def plot_kmod_results(opt: DotDict) -> None:
    """
    Function to plot the resulting beta-beating and waist.

    """

    plot_parameter(opt.results, PARAM_BETA, opt.ip)
    plot_parameter(opt.results, PARAM_WAIST, opt.ip)


def plot_parameter(results: tfs.TfsDataFrame, parameter: str, ip: str | None = None) -> plt.Figure:
    """
    Function to plot the resulting parameter, beta function or waist.

    Args:
        results (tfs.TfsDataFrame):
            The results to plot.
        parameter (str):
            Parameter to plot. Either 'beta' or 'waist'.
        ip (str|None):
            The specific IP to plot.

    Returns:
        The created figure.

    """
    if parameter not in [PARAM_BETA, PARAM_WAIST]:
        msg = f"Parameter must be either '{PARAM_BETA}' or '{PARAM_WAIST}', instead got '{parameter}'"
        raise ValueError(msg)

    fig, ax = _get_square_axes()
    for beam in (1, 2):
        if parameter == PARAM_BETA:
            x, xerr = _get_beat_and_err(results, beam, 'X')
            y, yerr = _get_beat_and_err(results, beam, 'Y')
        elif parameter == PARAM_WAIST:
            x, xerr = _get_waist_and_err(results, beam, 'X')
            y, yerr = _get_waist_and_err(results, beam, 'Y')

        ax.errorbar(
                    x, y, xerr=xerr, yerr=yerr,
                    color=f'C{beam-1}',
                    label=f'B{beam} IP{ip}' if ip is not None else f'B{beam}'
        )

    if parameter == PARAM_BETA:
        label = r'$\Delta\beta_{plane}/\beta_{plane}$ %'
    if parameter == PARAM_WAIST:
        label = r'Waist {plane} [m]'
    
    ax.set_xlabel(label.format(plane='x'))
    ax.set_ylabel(label.format(plane='y'))
    _set_square_limits(ax)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    
    ax.legend()
    return fig


def _get_square_axes(axes_loc=[0.17, 0.15, 0.8, 0.7]):
    fig, ax  = plt.subplots(1, 1)
    ax.set_aspect('equal')
    return fig, ax


def _set_square_limits(ax: plt.Axes):
    max_lim = np.max(np.abs(list(ax.get_xlim()) + list(ax.get_ylim())))
    ax.set_xlim(-max_lim, max_lim)
    ax.set_ylim(-max_lim, max_lim)


def _get_waist_and_err(results: tfs.TfsDataFrame, beam: int, plane: str) -> tuple[float, float]:
    waist = results.loc[beam, f'{WAIST}{plane}']
    err = results.loc[beam, f'{ERR}{WAIST}{plane}']
    return waist, err


def _get_beat_and_err(results: tfs.TfsDataFrame, beam: int, plane: str) -> tuple[float, float]:
    model = results.loc[beam, f'{BETASTAR}{MDL}']
    beat = (results.loc[beam, f'{BETASTAR}{plane}'] - model) / model * 100
    err = results.loc[beam, f'{ERR}{BETASTAR}{plane}'] / model * 100
    return beat, err
