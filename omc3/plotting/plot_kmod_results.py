""" 
Plot K-Modulation Results
-------------------------

Create Plots for the K-Modulation data.

**Arguments:**

*--Required--*

- **data** *(PathOrStrOrDataFrame)*:

    Path to the K-Mod BetaStar (i.e. `results.tfs`) DataFrame, 
    e.g. from `omc3.kmod_averages`, or the DataFrame itself.


*--Optional--*

- **ip** *(int)*:

    IP this result is from (for plot label and filename only).

- **betastar** *(float)*:

    Model beta-star values (x, y) for reference.


- **waist** *(float)*:

    Model waist values (x, y) for reference.


- **manual_style** *(DictAsString)*:

    Additional style rcParameters which update the set of predefined ones.

    default: ``{}``


- **output_dir** *(PathOrStr)*:

    Path to save the plots into. If not given, no plots will be saved.


- **plot_styles** *(str)*:

    Which plotting styles to use, either from plotting.styles.*.mplstyles
    or default mpl.

    default: ``['standard', 'kmod_results']``


- **show**:

    Show the plots.

    action: ``store_true``

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import tfs
from generic_parser import EntryPointParameters, entrypoint
from generic_parser.entry_datatypes import DictAsString

from omc3.optics_measurements.constants import BEAM, BETASTAR, ERR, MDL, WAIST
from omc3.plotting.utils import style as pstyle
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, PathOrStrOrDataFrame, save_config

if TYPE_CHECKING:
    from generic_parser import DotDict

LOG = logging_tools.get_logger(__name__)

PARAM_BETA: str = "beta"
PARAM_BETABEAT: str = "betabeat"
PARAM_WAIST: str = "waist"

AXIS_LABELS = {
    PARAM_BETA: r'$\beta_{plane}$ [m]',
    PARAM_BETABEAT: r'$\Delta\beta_{plane}/\beta_{plane}$ %',
    PARAM_WAIST: r'Waist {plane} [m]',
}


def _get_params() -> EntryPointParameters:
    params = EntryPointParameters()
    # Data Related ---
    params.add_parameter(
        name="data",
        required=True,
        type=PathOrStrOrDataFrame,
        help="Path to the K-Mod DataFrame (i.e. `results.tfs`), "
             "e.g. from `omc3.kmod_averages`, or the DataFrame itself.",
    )
    params.add_parameter(
        name="ip",
        type=int,
        help="IP this result is from (for plot label and filename only)."
    )
    params.add_parameter(
        name="betastar",
        type=float,
        nargs="+",
        help="Model beta-star values (x, y) for reference.",
    )
    params.add_parameter(
        name="waist",
        type=float,
        nargs="+",
        help="Model waist values (x, y) for reference.",
    )
    params.add_parameter(
        name="output_dir",
        type=PathOrStr,
        help="Path to save the plots into. If not given, no plots will be saved.",
        )

    # Plotting Related ---
    params.add_parameter(
        name="plot_styles",
        type=str,
        nargs="+",
        default=['standard', 'kmod_results'],
        help='Which plotting styles to use, either from plotting.styles.*.mplstyles or default mpl.'
    )
    params.add_parameter(
        name="manual_style",
        type=DictAsString,
        default={},
        help='Additional style rcParameters which update the set of predefined ones.'
    )
    params.add_parameter(
        name="show",
        action="store_true",
        help="Show the plots."
    )
    return params


@entrypoint(_get_params(), strict=True)
def plot_kmod_results(opt: DotDict) -> dict[str, plt.Figure]:
    """
    Function to plot the beta-beating and waist from K-Modulation data.
    """
    LOG.info("Plotting K-Mod results.")
    
    # Loading ---
    df_kmod = opt.data
    if isinstance(df_kmod, (Path, str)):
        df_kmod = tfs.read(df_kmod, index=BEAM)

    # Plotting ---
    pstyle.set_style(opt.plot_styles, opt.manual_style)

    figs: dict[str, plt.Figure] = {}
    for parameter, reference in ((PARAM_BETA, opt.betastar), (PARAM_BETABEAT, opt.betastar), (PARAM_WAIST, opt.waist)):
        if parameter == PARAM_BETABEAT and reference is None:
            continue
        
        figs[parameter]  = plot_parameter(df_kmod, parameter, reference=reference, ip=opt.ip)
    
    # Output ---
    if opt.output_dir is not None:
        output_dir = Path(opt.output_dir)
        output_dir.mkdir(exist_ok=True)

        if isinstance(opt.data, (Path, str)):  # don't save if called with DataFrames
            save_config(output_dir, opt, __file__)

        ip_str = f"ip{opt.ip}_" if opt.ip is not None else ""
        for parameter, fig in figs.items():
            out_file = output_dir / f"{ip_str}{parameter}.{plt.rcParams['savefig.format']}"
            LOG.debug(f"Writing {parameter} plot to {out_file}.")
            fig.savefig(out_file)

    if opt.show:
        plt.show()

    return figs


def plot_parameter(
    df_kmod: tfs.TfsDataFrame,
    parameter: str,
    reference: list[float],
    ip: str | None = None,
) -> plt.Figure:
    """
    Function to plot the resulting parameter, beta function or waist.

    Args:
        results (tfs.TfsDataFrame):
            The results to plot.
        parameter (str):
            Parameter to plot. Either 'beta' or 'waist'.
        reference (list[float]):
            Reference values to plot.
        ip (str|None):
            The specific IP to plot. (only used as label)

    Returns:
        The created figure.

    """
    if parameter not in [PARAM_BETA, PARAM_WAIST, PARAM_BETABEAT]:
        msg = (
            f"Parameter must be either '{PARAM_BETA}', '{PARAM_WAIST}' "
            f"or '{PARAM_BETABEAT}', instead got '{parameter}'"
        )
        raise ValueError(msg)

    LOG.debug(f"Plotting parameter {parameter}.")
    fig, ax = _get_square_axes()
    
    # Plot reference ---
    if reference is not None:
        if len(reference) == 1:
            reference = [reference, reference]

        if parameter != PARAM_BETABEAT:
            ax.plot(*reference, color="black", label="Model", ls='none')

    # Plot data ---
    for beam in (1, 2):
        if beam not in df_kmod.index:
            LOG.info(f"Beam {beam} not found in DataFrame. Skipping.")
            continue

        if parameter == PARAM_BETA:
            x, xerr = _get_beta_and_err(df_kmod, beam, 'X')
            y, yerr = _get_beta_and_err(df_kmod, beam, 'Y')

        if parameter == PARAM_BETABEAT:
            if reference is None:
                msg = "To plot betabeating, please give a betastar for reference!"
                raise ValueError(msg)

            x, xerr = _get_beat_and_err(df_kmod, beam, 'X', reference[0])
            y, yerr = _get_beat_and_err(df_kmod, beam, 'Y', reference[1])

        if parameter == PARAM_WAIST:
            x, xerr = _get_waist_and_err(df_kmod, beam, 'X')
            y, yerr = _get_waist_and_err(df_kmod, beam, 'Y')

        ax.errorbar(
                    x, y, xerr=xerr, yerr=yerr,
                    color=f'C{beam-1}',
                    label=f'B{beam} IP{ip}' if ip is not None else f'B{beam}'
        )

    # Decorate axes ---
    label = AXIS_LABELS[parameter]
    ax.set_xlabel(label.format(plane='x'))
    ax.set_ylabel(label.format(plane='y'))
    _set_square_limits(ax)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, marker='none')
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, marker='none')
    
    ax.legend()
    return fig


def _get_square_axes() -> tuple[plt.Figure, plt.Axes]:
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


def _get_beta_and_err(results: tfs.TfsDataFrame, beam: int, plane: str) -> tuple[float, float]:
    beta = results.loc[beam, f'{BETASTAR}{plane}'] 
    err = results.loc[beam, f'{ERR}{BETASTAR}{plane}']
    return beta, err


def _get_beat_and_err(results: tfs.TfsDataFrame, beam: int, plane: str, model: float) -> tuple[float, float]:
    beat = (results.loc[beam, f'{BETASTAR}{plane}'] - model) / model * 100
    err = results.loc[beam, f'{ERR}{BETASTAR}{plane}'] / model * 100
    return beat, err
