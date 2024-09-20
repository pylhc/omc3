"""
Plot Amplitude Detuning Results
-------------------------------

Provides the plotting function for amplitude detuning analysis

**Arguments:**

*--Required--*

- **kicks**:

    Kick files as data frames or tfs files.


- **labels** *(str)*:

    Labels for the data. Needs to be same length as kicks.


- **plane** *(str)*:

    Plane of the kicks.

    choices: ``['X', 'Y', 'XY', '3D']``


*--Optional--*

- **action_plot_unit** *(str)*:

    Unit the action should be plotted in.

    choices: ``['km', 'm', 'mm', 'um', 'nm', 'pm', 'fm', 'am']``

    default: ``um``


- **action_unit** *(str)*:

    Unit the action is given in.

    choices: ``['km', 'm', 'mm', 'um', 'nm', 'pm', 'fm', 'am']``

    default: ``m``


- **bbq_corrected** *(bool)*:

    Plot the data with BBQ correction (``True``) or without (``False``).
    ``None`` plots both in separate plots. Default: ``None``.


- **correct_acd**:

    Correct for AC-Dipole kicks.

    action: ``store_true``


- **detuning_order** *(int)*:

    Order of the detuning as int. Basically just the order of the applied
    fit.

    default: ``1``


- **manual_style** *(DictAsString)*:

    Additional style rcParameters which update the set of predefined ones.

    default: ``{}``


- **output** *(str)*:

    Save the amplitude detuning plot here. Give filename with extension.
    An id for the 4 different plots will be added before the suffix.


- **plot_styles** *(UnionPathStr)*:

    Plotting styles.

    default: ``['standard', 'amplitude_detuning']``


- **show**:

    Show the amplitude detuning plot.

    action: ``store_true``


- **tune_scale** *(int)*:

    Plotting exponent of the tune.

    default: ``-3``


- **x_lim** *(float)*:

    Action limits in um (x-axis).


- **y_lim** *(float)*:

    Tune limits in units of tune scale (y-axis).


"""
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from generic_parser import DotDict, EntryPointParameters, entrypoint
from generic_parser.entry_datatypes import DictAsString
from matplotlib import MatplotlibDeprecationWarning
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike
from scipy import odr
from tfs.tools import significant_digits

from omc3.definitions.constants import PLANES, UNIT_IN_METERS
from omc3.plotting.utils import annotations as pannot
from omc3.plotting.utils import colors as pcolors
from omc3.plotting.utils import style as pstyle
from omc3.tune_analysis import constants as const
from omc3.tune_analysis import fitting_tools
from omc3.tune_analysis import kick_file_modifiers as kick_mod
from omc3.tune_analysis.kick_file_modifiers import AmpDetData
from omc3.utils import logging_tools
from omc3.utils.contexts import suppress_warnings
from omc3.utils.iotools import PathOrStr, UnionPathStr, save_config

LOG = logging_tools.get_logger(__name__)

NFIT = 100  # Points for the fitting function
X, Y = PLANES


def get_params():
    return EntryPointParameters(
        kicks=dict(
            nargs="+",
            help="Kick files as data frames or tfs files.",
            required=True,
        ),
        labels=dict(
            help="Labels for the data. Needs to be same length as kicks.",
            nargs='+',
            required=True,
            type=str,
        ),
        plane=dict(
            help="Plane of the kicks.",
            required=True,
            choices=list(PLANES) + [''.join(PLANES), '3D'],
            type=str,
        ),
        detuning_order=dict(
            help="Order of the detuning as int. Basically just the order of the applied fit.",
            type=int,
            default=1,
        ),
        correct_acd=dict(
            help="Correct for AC-Dipole kicks.",
            action="store_true",
        ),
        output=dict(
            help=("Save the amplitude detuning plot here. "
                  "Give filename with extension. An id for the 4 different "
                  "plots will be added before the suffix."),
            type=PathOrStr,
        ),
        show=dict(
            help="Show the amplitude detuning plot.",
            action="store_true",
        ),
        y_lim=dict(
            help="Tune limits in units of tune scale (y-axis).",
            type=float,
            nargs=2,
        ),
        x_lim=dict(
            help="Action limits in um (x-axis).",
            type=float,
            nargs=2,
        ),
        action_unit=dict(
            help="Unit the action is given in.",
            default="m",
            choices=list(UNIT_IN_METERS.keys()),
            type=str,
        ),
        action_plot_unit=dict(
            help="Unit the action should be plotted in.",
            default="um",
            choices=list(UNIT_IN_METERS.keys()),
            type=str,
        ),
        manual_style=dict(
            type=DictAsString,
            default={},
            help='Additional style rcParameters which update the set of predefined ones.'
        ),
        plot_styles=dict(
            help="Plotting styles.",
            type=UnionPathStr,
            nargs="+",
            default=['standard', 'amplitude_detuning'],
        ),
        tune_scale=dict(
            help="Plotting exponent of the tune.",
            default=-3,
            type=int,
        ),
        bbq_corrected=dict(
            help="Plot the data with BBQ correction (``True``) or without (``False``)."
                 " ``None`` plots both in separate plots. Default: ``None``.",
            type=bool,
        )
    )


@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.info("Plotting Amplitude Detuning Results.")
    _save_options(opt)
    _check_opt(opt)

    figs = {}

    pstyle.set_style(opt.plot_styles, opt.manual_style)

    for tune_plane in PLANES:
        if opt.plane == "3D":
            figs.update(_plot_3d(tune_plane, opt))
        else:
            figs.update(_plot_2d(tune_plane, opt))

    if opt.show:
        plt.show()

    return figs


# Plotting --------------------------------------------------------------

# 2D Plots ----------------------------

def _plot_2d(tune_plane: str, opt: DotDict) -> Dict[str, Figure]:
    """ 2D Plots per kick-plane and with/without BBQ correction. """
    figs = {}
    limits = opt.get_subdict(['x_lim', 'y_lim'])
    tune_scale = 10 ** opt.tune_scale

    for action_plane in opt.plane:
        for corrected in opt.bbq_corrected:  # with / without BBQ correction
            corr_label = "_corrected" if corrected else ""

            fig = plt.figure()
            ax = fig.add_subplot(111)
            isempty = True

            for idx, (kick_file, label) in enumerate(zip(opt.kicks, opt.labels)):
                kick_df = kick_mod.read_timed_dataframe(kick_file) if isinstance(kick_file, PathOrStr) else kick_file
                try:
                    data = kick_mod.get_ampdet_data(kick_df,
                                                    tune_plane=tune_plane,
                                                    action_plane=action_plane,
                                                    corrected=corrected
                                                    )
                except KeyError as e:
                    LOG.debug(f"Entries not found in dataframe: {str(e)}")
                    continue  # should only happen when there is no 'corrected' columns

                # Read data from kick_df headers ---
                odr_fit = kick_mod.get_odr_data(kick_df,
                                                tune_plane=tune_plane,
                                                action_plane=action_plane,
                                                order=opt.detuning_order,
                                                corrected=corrected
                                                )

                # Get label for ODR fit ---
                odr_label = _get_odr_label(odr_fit,
                                           tune_plane=tune_plane,
                                           action_plane=action_plane,
                                           action_unit=opt.action_unit,
                                           do_acd_correction=opt.correct_acd
                                           )

                # Scale and Plot ---
                data, odr_fit = _scale_data(
                    data=data, odr_fit=odr_fit,
                    action_unit=opt.action_unit, action_plot_unit=opt.action_plot_unit,
                    tune_scale=tune_scale
                )
                _plot_detuning(
                    ax, data=data, label=label, limits=limits,
                    odr_fit=odr_fit, odr_label=odr_label,
                    color=pcolors.get_mpl_color(idx),
                )
                isempty = False

            if isempty:
                plt.close(fig)  # don't show empty figs
                continue  # don't save/return empty figures

            if opt.correct_acd and tune_plane == action_plane:
                ax.text(0.0, 1.02, "* corrected for AC-Dipole.",
                        fontsize="x-small",
                        ha="left",
                        va='bottom',
                        transform=ax.transAxes,
                        )

            id_str = f"dQ{tune_plane.upper():s}d2J{action_plane.upper():s}{corr_label:s}"
            pannot.set_name(id_str, fig)

            ax_labels = (
                const.get_action_label(action_plane, opt.action_plot_unit),
                const.get_tune_label(tune_plane, opt.tune_scale)
            )
            _format_axes(ax, labels=ax_labels, limits=limits)

            if opt.output:
                output = Path(opt.output)
                fig.savefig(f'{output.with_suffix("")}_{id_str}{output.suffix}')

            figs[id_str] = fig
    return figs


def plot_odr(ax: Axes, odr_fit: odr.Output, xmax: float, label: str = '', color=None):
    """Adds a quadratic odr fit to axes."""

    color = 'k' if color is None else color
    odr_fit.beta[0] = 0   # no need to remove offset, as it is removed from data

    # get fits
    order = len(odr_fit.beta) - 1
    fit_fun = fitting_tools.get_poly_fun(order)
    f = partial(fit_fun, odr_fit.beta)
    f_low = partial(fit_fun, np.array(odr_fit.beta)-np.array(odr_fit.sd_beta))
    f_upp = partial(fit_fun, np.array(odr_fit.beta)+np.array(odr_fit.sd_beta))

    if order == 1:
        x = np.array([0, xmax])
    else:
        x = np.linspace(0, xmax, NFIT)

    line = ax.plot(x, f(x), marker="", linestyle='--', color=color, label=label)
    if color is None:
        color = line[0].get_color()
    ax.fill_between(x, f_low(x), f_upp(x), facecolor=mcolors.to_rgba(color, .3), zorder=-10)
    return color


def _plot_detuning(ax: Axes, data: AmpDetData, label: str, color=None,
                   limits: Dict[str, float] = None, odr_fit: odr.Output=None, odr_label: str =""):
    """Plot the detuning and the ODR into axes."""
    x_lim = _get_default(limits, 'x_lim', [0, max(data.action+data.action_err)])
    offset = 0

    # Plot Fit
    if odr_fit is not None:
        offset = odr_fit.beta[0]
        color = plot_odr(ax, odr_fit, xmax=x_lim[1], color=color, label=odr_label)

    # Plot Data
    data.tune -= offset
    ax.errorbar(x=data.action, xerr=data.action_err,
                y=data.tune, yerr=data.tune_err,
                label=label, color=color)


def _format_axes(ax: Axes, limits: Dict[str, Sequence[float]], labels: Sequence[str]):
    # labels
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    # limits
    if limits['x_lim'] is not None:
        ax.set_xlim(limits['x_lim'])

    if limits['y_lim'] is not None:
        ax.set_ylim(limits['y_lim'])

    pannot.make_top_legend(ax, ncol=2)


# 3D Plots ----------------------------

def _plot_3d(tune_plane: str, opt: DotDict):
    """ 3D Plots for both kick-planes, one plot each with/without BBQ correction. """
    figs = {}
    limits = opt.get_subdict(['x_lim', 'y_lim'])
    tune_scale = 10 ** opt.tune_scale

    for corrected in opt.bbq_corrected:  # with / without BBQ correction
        corr_label = "_corrected" if corrected else ""

        # hack to draw spines properly, as their style is
        # determined by lines
        rc_save = rcParams.copy()
        rcParams['lines.linestyle'] = "-"
        rcParams['lines.marker'] = ""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        all_odr_labels = [None] * len(opt.kicks)

        # restore rcParams from before the hack
        with suppress_warnings(MatplotlibDeprecationWarning):
            rcParams.update(rc_save)
        isempty = True

        for idx, (kick_file, label) in enumerate(zip(opt.kicks, opt.labels)):
            kick_df = kick_mod.read_timed_dataframe(kick_file) if isinstance(kick_file, PathOrStr) else kick_file

            datas, odr_fits, odr_labels = {}, {}, {}
            for action_plane in PLANES:
                do_acd_correction = opt.correct_acd and (action_plane == tune_plane)
                try:
                    data = kick_mod.get_ampdet_data(kick_df,
                                                    action_plane=action_plane,
                                                    tune_plane=tune_plane,
                                                    corrected=corrected)
                except KeyError as e:
                    LOG.debug(str(e))
                    continue  # should only happen when there is no 'corrected' columns

                # Read data from kick_df headers ---
                odr_fit = kick_mod.get_odr_data(kick_df, action_plane, tune_plane,
                                                order=opt.detuning_order,
                                                corrected=corrected)

                # Get label for ODR fit ---
                odr_label = _get_odr_label(odr_fit,
                                           tune_plane=tune_plane,
                                           action_plane=action_plane,
                                           action_unit=opt.action_unit,
                                           do_acd_correction=do_acd_correction)

                # Scale ---
                data, odr_fit = _scale_data(
                    data, odr_fit,
                    opt.action_unit, opt.action_plot_unit,
                    tune_scale
                )

                # Save ---
                isempty = False
                odr_fits[action_plane] = odr_fit
                odr_labels[action_plane] = odr_label
                datas[action_plane] = data

            if len(datas) < 2:
                continue

            _plot_detuning_3d(
                ax, data=datas, label=label, limits=limits,
                odr_fits=odr_fits,
                color=pcolors.get_mpl_color(idx),
            )
            all_odr_labels[idx] = odr_labels

        if isempty:
            plt.close(fig)  # don't show empty figs
            continue  # don't save/return empty figures

        id_str = f"dQ{tune_plane.upper():s}d2JXY_3D{corr_label:s}"
        pannot.set_name(id_str, fig)

        ax_labels = (
            const.get_action_label(PLANES[0], opt.action_plot_unit),
            const.get_action_label(PLANES[1], opt.action_plot_unit),
            const.get_tune_label(tune_plane, opt.tune_scale)
        )
        ax.view_init(azim=45, elev=18)
        _format_axes_3d(ax, ax_labels=ax_labels, limits=limits,
                        acd=opt.correct_acd, odr_labels=all_odr_labels)

        if opt.output:
            output = Path(opt.output)
            fig.savefig(f'{output.with_suffix("")}_{id_str}{output.suffix}')
        figs[id_str] = fig
    return figs


def _plot_detuning_3d(ax: Axes, data: Dict[str, AmpDetData], label: str, color=None, limits=None, odr_fits=None):
    """ Plot the detuning and the ODR into axes."""
    offset = 0
    jx, jx_err = data[X].action, data[X].action_err
    jy, jy_err = data[Y].action, data[Y].action_err
    tune, tune_err = data[X].tune, data[X].tune_err
    x_lim = _get_default(limits, 'x_lim', [0, max(jx+jx_err)])
    y_lim = _get_default(limits, 'x_lim', [0, max(jy+jy_err)])

    # Plot Fit
    if odr_fits is not None:
        offset = odr_fits[X].beta[0]
        color = plot_odr_3d(ax, odr_fits, xymax=[x_lim[1], y_lim[1]], color=color)

    # Plot Data
    tune -= offset

    # there is no errorbar3D in mpl it seems
    # so plotting is done manually.
    # plot points
    line = ax.plot(jx, jy, tune, label=label, color=color)
    if color is None:
        color = line[0].get_color()

    # plot errorbars
    for idx in np.arange(0, len(tune)):
        x, dx = jx.iloc[idx], jx_err.iloc[idx]
        y, dy = jy.iloc[idx], jy_err.iloc[idx]
        z, dz = tune.iloc[idx], tune_err.iloc[idx]

        ax.plot([x+dx, x-dx], [y, y], [z, z], ls="-", marker="_", color=color)
        ax.plot([x, x], [y+dy, y-dy], [z, z], ls="-", marker="_", color=color)
        ax.plot([x, x], [y, y], [z+dz, z-dz], ls="-", marker="_", color=color)


@dataclass
class FitFuncs:
    fit: callable
    upper: callable
    lower: callable


def fit_fun_odr_1d(x, y, q0, qdx, qdy):
    return q0 + qdx * x + qdy * y


def plot_odr_3d(ax: Axes, odr_fits: Dict[str, odr.Output], xymax: Sequence[float], color=None):
    """Plot the odr fit in 3D."""

    color = 'k' if color is None else color
    odr_fits[X].beta[0] = 0   # no need to remove offset, as it is removed from data
    odr_fits[Y].beta[0] = 0   # no need to remove offset, as it is removed from data

    # get fits
    order = len(odr_fits[X].beta) - 1
    if order > 1:
        raise NotImplementedError("ODR fit plots for order > 1 are not implemented.")
    f = partial(fit_fun_odr_1d,
                q0=odr_fits[X].beta[0],
                qdx=odr_fits[X].beta[1],
                qdy=odr_fits[Y].beta[1])
    f_low = partial(fit_fun_odr_1d,
                    q0=odr_fits[X].beta[0]-odr_fits[X].sd_beta[0],
                    qdx=odr_fits[X].beta[1]-odr_fits[X].sd_beta[1],
                    qdy=odr_fits[Y].beta[1]-odr_fits[Y].sd_beta[1])
    f_upp = partial(fit_fun_odr_1d,
                    q0=odr_fits[X].beta[0]+odr_fits[X].sd_beta[0],
                    qdx=odr_fits[X].beta[1]+odr_fits[X].sd_beta[1],
                    qdy=odr_fits[Y].beta[1]+odr_fits[Y].sd_beta[1])

    x = np.linspace(0, xymax[0], 6)
    y = np.linspace(0, xymax[1], 6)
    xm, ym = np.meshgrid(x, y)

    # line = ax.plot_surface(xm, ym, f(xm, ym), color=color, label=label, alpha=0.3)
    frame = ax.plot_wireframe(xm, ym, f(xm, ym), color=color, alpha=0.5)
    if color is None:
        color = frame[0].get_color()

    # Plot only the lower and upper boundaries
    # ax.plot_surface(xm, ym, f_low(xm, ym), color=color, alpha=0.3)
    # ax.plot_surface(xm, ym, f_upp(xm, ym), color=color, alpha=0.3)

    # Plot the full surface
    x = np.linspace(0, xymax[0], 2)
    y = np.linspace(0, xymax[1], 2)
    plot_cube(ax, x, y, f_low, f_upp, color=color, alpha=0.2, rcount=2, ccount=2)
    return color


def plot_cube(ax: Axes, x: ArrayLike, y: ArrayLike, f_low: callable, f_upp: callable, **kwargs):
    """Plots a cube-like plot with f_low and f_upp as surfaces. """
    # lower
    xm, ym = np.meshgrid(x, y)
    zm = f_low(xm, ym)
    ax.plot_surface(xm, ym, zm, **kwargs)

    # upper
    zm = f_upp(xm, ym)
    ax.plot_surface(xm, ym, zm, **kwargs)

    # north
    xm, ym = np.array([x, x]), y[0]*np.ones([2, len(y)])
    zm = np.array([f_low(x, y[0]), f_upp(x, y[0])])
    ax.plot_surface(xm, ym, zm, **kwargs)

    # south
    xm, ym = np.array([x, x]), y[-1]*np.ones([2, len(y)])
    zm = np.array([f_low(x, y[-1]), f_upp(x, y[-1])])
    ax.plot_surface(xm, ym, zm, **kwargs)

    # east
    xm, ym = x[0]*np.ones([2, len(x)]), np.array([y, y])
    zm = np.array([f_low(x[0], y), f_upp(x[0], y)])
    ax.plot_surface(xm, ym, zm, **kwargs)

    # west
    xm, ym = x[-1]*np.ones([2, len(x)]), np.array([y, y])
    zm = np.array([f_low(x[-1], y), f_upp(x[-1], y)])
    ax.plot_surface(xm, ym, zm, **kwargs)


def _format_axes_3d(
        ax: Axes, limits: Dict[str, float], ax_labels: Sequence[str],
        acd: bool, odr_labels: Sequence[Dict[str, str]] = None):
    # labels
    ax.set_xlabel(ax_labels[0], labelpad=15)
    ax.set_ylabel(ax_labels[1], labelpad=15)
    ax.set_zlabel(ax_labels[2], labelpad=10)

    # limits
    if limits['x_lim'] is not None:
        limj = limits['x_lim']
    else:
        limj = [0, max([ax.get_xlim()[1], ax.get_ylim()[1]])]
    ax.set_xlim(limj)
    ax.set_ylim(limj)

    if limits['y_lim'] is not None:
        ax.set_zlim(limits['y_lim'])

    # pannot.make_top_legend(ax, ncol=2)
    if acd:
        ax.text2D(
            1.4, 0.02, "* corrected for AC-Dipole.",
            fontsize="x-small",
            ha="left",
            va='bottom',
            transform=ax.transAxes,
        )
    handles, labels = get_labels_with_odr_labels(ax, odr_labels)
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.2, 0.98), prop={'size': 'small'})
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)  # move tune axis to the other side

    # tight layout so that the legend fits in figure
    # We catch and ignore 'UserWarning: The figure layout has changed to tight'
    # because it is something we did on purpose, let's not pollute the output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ax.figure.tight_layout()
        ax.figure.tight_layout()


# Labels -----------------------------------------------------------------------

def _get_odr_label(odr_fit: odr.Output, tune_plane: str, action_plane: str,
                   action_unit: str, do_acd_correction: bool):
    """ Returns the label for the ODR fit, nicely formatted and scaled. """
    order = len(odr_fit.beta) - 1
    str_list = [None] * order
    for o in range(order):
        acd_correction = 1
        if do_acd_correction:
            if order == 1 and tune_plane == action_plane:
                acd_correction = 0.5
            if order == 2 and tune_plane == action_plane:
                acd_correction = 1/3

        label = _get_scaled_odr_label(
            odr_fit,
            order=o+1,
            action_unit=action_unit,
            acd_correction=acd_correction,
            magnitude_exponent=const.get_detuning_exponent_for_order(o+1)
        )
        order_str = f"^{o+1}" if o else ""
        str_list[o] = f"$Q_{{{tune_plane},{action_plane}{order_str}}}$: {label}"
    return ", ".join(str_list)


def _get_scaled_odr_label(odr_fit, order, action_unit, acd_correction, magnitude_exponent=3):
    """ Returns the label for the ODR fit after scaling to readable units and accounting for AC-Dipole correction."""
    str_acd_scale = "$^*$" if acd_correction != 1  else ""

    scale = acd_correction * (10 ** -magnitude_exponent) / (UNIT_IN_METERS[action_unit] ** order)
    str_val, str_std = _get_scaled_labels(odr_fit.beta[order], odr_fit.sd_beta[order], scale)
    str_mag = ''
    if magnitude_exponent != 0:
        str_mag = fr'$\cdot$ 10$^{{{magnitude_exponent}}}$'
    return fr'({str_val} $\pm$ {str_std}){str_acd_scale} {str_mag} m$^{{-{order}}}$'


def _get_scaled_labels(val, std, scale):
    scaled_vas, scaled_std = val*scale, std*scale
    if abs(scaled_std) > 1 or scaled_std == 0:
        return f'{int(round(scaled_vas)):d}', f'{int(round(scaled_std)):d}'
    return significant_digits(scaled_vas, scaled_std)


def get_labels_with_odr_labels(ax, odr_labels):
    handles, labels = ax.get_legend_handles_labels()
    if odr_labels is None:
        return handles, labels

    empty = Line2D([0], [0], ls='none', marker='', label='')
    h_new, l_new = [], []
    for handle, label, odr_label in zip(handles, labels, odr_labels):
        h_new.append(handle)
        l_new.append(label)
        if odr_label is not None:
            h_new.append(empty)
            l_new.append("\n".join(odr_label.values()))
    return h_new, l_new


# Helper -----------------------------------------------------------------------

def _save_options(opt):
    if opt.output:
        out_path = Path(opt.output).parent
        save_config(out_path, opt, __file__)


def _check_opt(opt):
    if opt.bbq_corrected is None:
        opt.bbq_corrected = (False, True)  # loop both
    else:
        opt.bbq_corrected = (opt.bbq_corrected,)  # make iterable

    if len(opt.labels) != len(opt.kicks):
        raise ValueError("'kicks' and 'labels' need to be of same size!")


def _get_default(ddict, key, default):
    """Returns ``default`` if either the dict itself or the entry is ``None``."""
    if ddict is None or key not in ddict or ddict[key] is None:
        return default
    return ddict[key]


def _scale_data(data: AmpDetData, odr_fit: odr.Output,
                action_unit: str, action_plot_unit: str, tune_scale: float):
    """Scale data to plot-units (y=tune_scale, x=um)."""
    x_scale = UNIT_IN_METERS[action_unit] / UNIT_IN_METERS[action_plot_unit]
    data.action *= x_scale
    data.action_err *= x_scale

    # correct for tune scaling
    y_scale = 1. / tune_scale
    data.tune *= y_scale
    data.tune_err *= y_scale

    # same for odr_fit:
    for idx in range(len(odr_fit.beta)):
        full_scale = y_scale / (x_scale**idx)
        odr_fit.beta[idx] *= full_scale
        odr_fit.sd_beta[idx] *= full_scale
    return data, odr_fit


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    main()

