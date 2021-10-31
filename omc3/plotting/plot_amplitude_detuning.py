"""
Plot Amplitude Detuning Results
-------------------------------

Provides the plotting function for amplitude detuning analysis

**Arguments:**

*--Required--*

- **kicks**: Kick files as data frames or tfs files.

- **labels** *(str)*: Labels for the data. Needs to be same length as kicks.

- **plane** *(str)*: Plane of the kicks.

  Choices: ``('X', 'Y')``

*--Optional--*

- **action_plot_unit** *(str)*: Unit the action should be plotted in.

  Choices: ``['km', 'm', 'mm', 'um', 'nm', 'pm', 'fm', 'am']``
  Default: ``um``
- **action_unit** *(str)*: Unit the action is given in.

  Choices: ``['km', 'm', 'mm', 'um', 'nm', 'pm', 'fm', 'am']``
  Default: ``m``
- **correct_acd**: Correct for AC-Dipole kicks.

  Action: ``store_true``
- **detuning_order** *(int)*: Order of the detuning as int. Basically just the order of the applied fit.

  Default: ``1``
- **manual_style** *(DictAsString)*: Additional plotting style.

  Default: ``{}``
- **output** *(str)*: Save the amplitude detuning plot here.

- **show**: Show the amplitude detuning plot.

  Action: ``store_true``
- **tune_scale** *(int)*: Plotting exponent of the tune.

  Default: ``-3``
- **x_lim** *(float)*: Action limits in um (x-axis).

- **y_lim** *(float)*: Tune limits in units of tune scale (y-axis).
"""
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
from generic_parser import entrypoint, EntryPointParameters
from generic_parser.entry_datatypes import DictAsString
from generic_parser.entrypoint_parser import save_options_to_config
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from tfs.tools import significant_digits

from omc3.definitions import formats
from omc3.definitions.constants import UNIT_IN_METERS, PLANES
from omc3.plotting.utils import colors as pcolors, annotations as pannot, style as pstyle
from omc3.tune_analysis import constants as const, kick_file_modifiers as kick_mod, fitting_tools
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)

NFIT = 100  # Points for the fitting function

MANUAL_STYLE_DEFAULT = {
    u'figure.figsize': [9.5, 4],
    u"lines.marker": u"o",
    u"lines.linestyle": u"",
}


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
            choices=PLANES,
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
            type=str,
        ),
        show=dict(
            help="Show the amplitude detuning plot.",
            action="store_true",
        ),
        y_lim=dict(
            help="Tune limits in units of tune scale (y-axis).",
            type=float,
        ),
        x_lim=dict(
            help="Action limits in um (x-axis).",
            type=float,
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
            help="Additional plotting style.",
            type=DictAsString,
            default={}
        ),
        tune_scale=dict(
            help="Plotting exponent of the tune.",
            default=-3,
            type=int,
        )
    )


@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.info("Plotting Amplitude Detuning Results.")
    _save_options(opt)
    _check_opt(opt)

    kick_plane = opt.plane
    figs = {}

    _set_plotstyle(opt.manual_style)
    limits = opt.get_subdict(['x_lim', 'y_lim'])

    for tune_plane in PLANES:
        for corrected in [False, True]:
            corr_label = "_corrected" if corrected else ""
            acd_corr = 1
            if opt.correct_acd and (kick_plane == tune_plane):
                acd_corr = 0.5

            fig = plt.figure()
            ax = fig.add_subplot(111)

            for idx, (kick, label) in enumerate(zip(opt.kicks, opt.labels)):
                kick_df = kick_mod.read_timed_dataframe(kick) if isinstance(kick, str) else kick
                try:
                    data_df = kick_mod.get_ampdet_data(kick_df, kick_plane, tune_plane, corrected=corrected)
                except KeyError:
                    continue  # should only happen when there is no 'corrected' columns

                odr_fit = kick_mod.get_odr_data(kick_df, kick_plane, tune_plane,
                                                order=opt.detuning_order, corrected=corrected)
                data_df, odr_fit = _correct_and_scale(data_df, odr_fit,
                                                   opt.action_unit, opt.action_plot_unit,
                                                   10**opt.tune_scale, acd_corr)

                _plot_detuning(ax, data=data_df, label=label, limits=limits,
                               odr_fit=odr_fit,
                               color=pcolors.get_mpl_color(idx),
                               action_unit=opt.action_plot_unit, tune_scale=10**opt.tune_scale)

            ax_labels = const.get_paired_lables(tune_plane, kick_plane, opt.tune_scale)
            id_str = f"dQ{tune_plane.upper():s}d2J{kick_plane.upper():s}{corr_label:s}"
            pannot.set_name(id_str, fig)
            _format_axes(ax, labels=ax_labels, limits=limits)

            if opt.output:
                output = Path(opt.output)
                fig.savefig(f'{output.with_suffix("")}_{id_str}{output.suffix}')

            figs[id_str] = fig

    if opt.show:
        plt.show()

    return figs


# Plotting --------------------------------------------------------------


def _get_scaled_odr_label(odr_fit, order, action_unit, tune_scale, magnitude_exponent=3):
    scale = (tune_scale * (10 ** -magnitude_exponent)) / (UNIT_IN_METERS[action_unit] ** order)
    str_val, str_std = _get_scaled_labels(odr_fit.beta[order], odr_fit.sd_beta[order], scale)
    str_mag = ''
    if magnitude_exponent != 0:
        str_mag = f'$\cdot$ 10$^{{{magnitude_exponent}}}$'
    return f'({str_val} $\pm$ {str_std}) {str_mag} m$^{{-{order}}}$'


def _get_odr_label(odr_fit, action_unit, tune_scale):
    order = len(odr_fit.beta) - 1

    str_list = [None] * order
    for o in range(order):
        str_list[o] = _get_scaled_odr_label(odr_fit, o+1, action_unit, tune_scale,
                                            magnitude_exponent=const.get_detuning_exponent_for_order(o+1))
    return ", ".join(str_list)


def plot_odr(ax, odr_fit, xmax, action_unit, tune_scale, color=None):
    """Adds a quadratic odr fit to axes."""

    color = 'k' if color is None else color
    odr_fit.beta[0] = 0   # no need to remove offset, as it is removed from data
    label = _get_odr_label(odr_fit, action_unit, tune_scale)

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

    ax.fill_between(x, f_low(x), f_upp(x), facecolor=mcolors.to_rgba(color, .3))
    ax.plot(x, f(x), marker="", linestyle='--', color=color, label=label)


# Main Plot ---


def _plot_detuning(ax, data, label, action_unit, tune_scale, color=None, limits=None, odr_fit=None):
    """Plot the detuning and the ODR into axes."""
    x_lim = _get_default(limits, 'x_lim', [0, max(data['action']+data['action_err'])])

    # Plot Fit
    offset = odr_fit.beta[0]
    plot_odr(ax, odr_fit, xmax=x_lim[1], action_unit=action_unit, tune_scale=tune_scale, color=color)

    # Plot Data
    data['tune'] -= offset
    ax.errorbar(x=data['action'], xerr=data['action_err'],
                y=data['tune'], yerr=data['tune_err'],
                label=label, color=color)


def _set_plotstyle(manual_style):
    mstyle = MANUAL_STYLE_DEFAULT
    mstyle.update(manual_style)
    pstyle.set_style("standard", mstyle)


def _format_axes(ax, limits, labels):
    # labels
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    # limits
    if limits['x_lim'] is not None:
        ax.set_xlim(limits['x_lim'])

    if limits['y_lim'] is not None:
        ax.set_ylim(limits['y_lim'])

    pannot.make_top_legend(ax, ncol=2)

    ax.figure.tight_layout()
    ax.figure.tight_layout()  # needs two calls for some reason to look great


def _get_scaled_labels(val, std, scale):
    scaled_vas, scaled_std = val*scale, std*scale
    if abs(scaled_std) > 1 or scaled_std == 0:
        return f'{int(round(scaled_vas)):d}', f'{int(round(scaled_std)):d}'
    return significant_digits(scaled_vas, scaled_std)


# Helper -----------------------------------------------------------------------


def _save_options(opt):
    if opt.output:
        out_path = Path(opt.output).parent
        out_path.mkdir(exist_ok=True, parents=True)
        save_options_to_config(str(out_path / formats.get_config_filename(__file__)),
                               OrderedDict(sorted(opt.items()))
                               )


def _check_opt(opt):
    if len(opt.labels) != len(opt.kicks):
        raise ValueError("'kicks' and 'labels' need to be of same size!")


def _get_default(ddict, key, default):
    """Returns ``default`` if either the dict itself or the entry is ``None``."""
    if ddict is None or key not in ddict or ddict[key] is None:
        return default
    return ddict[key]


def _correct_and_scale(data, odr_fit, action_unit, action_plot_unit, tune_scale, acd_corr):
    """Corrects data for AC-Dipole and scales to plot-units (y=tune_scale, x=um)."""
    # scale action units
    x_scale = UNIT_IN_METERS[action_unit] / UNIT_IN_METERS[action_plot_unit]
    data['action'] *= x_scale
    data['action_err'] *= x_scale

    # correct for ac-diple and tune scaling
    y_scale = acd_corr / tune_scale
    data['tune'] *= y_scale
    data['tune_err'] *= y_scale

    # same for odr_fit:
    for idx in range(len(odr_fit.beta)):
        full_scale = y_scale / (x_scale**idx)
        odr_fit.beta[idx] *= full_scale
        odr_fit.sd_beta[idx] *= full_scale
    return data, odr_fit


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    main()

