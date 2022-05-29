"""
Plot BBQ
--------

Provides the plotting function for the extracted and cleaned BBQ data from timber.

**Arguments:**

*--Required--*

- **input**:

    BBQ data as data frame or tfs file.


*--Optional--*

- **interval** *(float)*:

    x_axis interval that was used in calculations.


- **kick**:

    Kick file as data frame or tfs file.


- **manual_style** *(DictAsString)*:

    Additional style rcParameters which update the set of predefined ones.

    default: ``{}``


- **output** *(str)*:

    Save figure to this location.


- **plot_styles** *(UnionPathStr)*:

    Which plotting styles to use, either from plotting.styles.*.mplstyles
    or default mpl.

    default: ``['standard', 'bbq']``


- **show**:

    Show plot.

    action: ``store_true``


- **two_plots**:

    Plot two axis into the figure.

    action: ``store_true``


- **x_lim** *(float)*:

    X-Axis limits. (yyyy-mm-dd HH:mm:ss.mmm)


- **y_lim** *(float)*:

    Y-Axis limits.
    

"""
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
from generic_parser import entrypoint, EntryPointParameters
from generic_parser.entry_datatypes import DictAsString
from generic_parser.entrypoint_parser import save_options_to_config
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pandas.plotting import register_matplotlib_converters

from omc3 import amplitude_detuning_analysis as ad_ana
from omc3.definitions import formats
from omc3.definitions.constants import PLANES
from omc3.plotting.utils import colors as pcolors, style as pstyle
from omc3.tune_analysis import kick_file_modifiers as kick_mod
from omc3.tune_analysis.constants import (get_mav_window_header, get_used_in_mav_col,
                                          get_bbq_col, get_mav_col)
from omc3.utils import logging_tools
from omc3.utils.iotools import UnionPathStr, PathOrStr, PathOrStrOrDataFrame

LOG = logging_tools.get_logger(__name__)

# Registering converters for datetime plotting as pandas won't do it for us automatically anymore
register_matplotlib_converters()


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="input",
        help="BBQ data as data frame or tfs file.",
        required=True,
        type=PathOrStrOrDataFrame
    )
    params.add_parameter(
        name="kick",
        help="Kick file as data frame or tfs file.",
        type=PathOrStrOrDataFrame
    )
    params.add_parameter(
        name="output",
        help="Save figure to this location.",
        type=PathOrStr,
    )
    params.add_parameter(
        name="show",
        help="Show plot.",
        action="store_true"
    )
    params.add_parameter(
        name="x_lim",
        help="X-Axis limits. (yyyy-mm-dd HH:mm:ss.mmm)",
        type=float,
        nargs=2,
    )
    params.add_parameter(
        name="y_lim",
        help="Y-Axis limits.",
        type=float,
        nargs=2,
    )
    params.add_parameter(
        name="interval",
        help="x_axis interval that was used in calculations.",
        type=float,
        nargs=2,
    )
    params.add_parameter(
        name="two_plots",
        help="Plot two axis into the figure.",
        action="store_true",
    )
    params.add_parameter(
        name="plot_styles",
        type=UnionPathStr,
        nargs="+",
        default=['standard', 'bbq'],
        help='Which plotting styles to use, either from plotting.styles.*.mplstyles or default mpl.'
    )
    params.add_parameter(
        name="manual_style",
        type=DictAsString,
        default={},
        help='Additional style rcParameters which update the set of predefined ones.'
    )
    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    """Plot BBQ wrapper."""
    LOG.info("Plotting BBQ.")
    _save_options(opt)
    pstyle.set_style(opt.pop("plot_styles"), opt.pop("manual_style"))

    bbq_df = kick_mod.read_timed_dataframe(opt.input) if isinstance(opt.input, (Path, str)) else opt.input
    opt.pop("input")

    if opt.kick is not None:
        if opt.interval is not None:
            raise ValueError("interval and kick-file given. Both are used for the same purpose. Please only use one.")

        window = 0  # not too important, bars will then indicate first and last kick directly
        with suppress(KeyError):
            window = bbq_df.headers[get_mav_window_header()]

        kick_df = kick_mod.read_timed_dataframe(opt.kick) if isinstance(opt.kick, (Path, str)) else opt.kick
        opt.interval = ad_ana.get_approx_bbq_interval(bbq_df, kick_df.index, window)
        bbq_df = bbq_df.loc[opt.interval[0]:opt.interval[1]]
    opt.pop("kick")

    show = opt.pop("show")
    out = opt.pop("output")

    fig = _plot_bbq_data(bbq_df, **opt)

    if show:
        plt.show()
    if out:
        fig.savefig(out)

    return fig


def _plot_bbq_data(bbq_df, interval=None, x_lim=None, y_lim=None, two_plots=False):
    """
    Plot BBQ data.

    Args:
        bbq_df: BBQ Dataframe with moving average columns.
        interval: start and end time of used interval, will be marked with red bars.
        x_lim: x limits (time).
        y_lim: y limits (tune).
        output: Path to the output file.
        show: Shows plot if ``True``.
        two_plots: Plots each tune in it's own axes if ``True``.

    Returns:
        Plotted figure.
    """
    LOG.debug("Plotting BBQ data.")
    fig, axs = plt.subplots(1+two_plots, 1)

    if not two_plots:
        axs = [axs, axs]

    handles = [None] * (3 * len(PLANES))
    for idx, plane in enumerate(PLANES):
        color = pcolors.get_mpl_color(idx)
        mask = np.array(bbq_df[get_used_in_mav_col(plane)], dtype=bool)

        # plot and save handles for nicer legend
        handles[idx] = axs[idx].plot([i.datetime for i in bbq_df.index],
                                    bbq_df[get_bbq_col(plane)],
                                    color=pcolors.change_color_brightness(color, .4),
                                    marker="o", markerfacecolor="None",
                                    label="$Q_{:s}$".format(plane.lower(),)
                                    )[0]
        filtered_data = bbq_df.loc[mask, get_bbq_col(plane)].dropna()
        handles[len(PLANES)+idx] = axs[idx].plot(filtered_data.index, filtered_data.to_numpy(),
                                                color=pcolors.change_color_brightness(color, .7),
                                                marker=".",
                                                label="filtered".format(plane.lower())
                                                )[0]
        handles[2*len(PLANES)+idx] = axs[idx].plot(bbq_df.index, bbq_df[get_mav_col(plane)],
                                                  color=color,
                                                  linestyle="-",
                                                  label="moving av.".format(plane.lower())
                                                  )[0]

        if (y_lim is None or y_lim[0] is None) and two_plots:
            axs[idx].set_ylim(bottom=min(bbq_df.loc[mask, get_bbq_col(plane)]))

        if (y_lim is None or y_lim[1] is None) and two_plots:
            axs[idx].set_ylim(top=max(bbq_df.loc[mask, get_bbq_col(plane)]))

    # things to add/do only once if there is only one plot
    for idx in range(1+two_plots):
        if interval:
            axs[idx].axvline(x=interval[0], color="red")
            axs[idx].axvline(x=interval[1], color="red")

        if two_plots:
            axs[idx].set_ylabel("$Q_{:s}$".format(PLANES[idx]))
        else:
            axs[idx].set_ylabel('Tune')

        if y_lim is not None:
            axs[idx].set_ylim(y_lim)
        axs[idx].yaxis.set_major_formatter(FormatStrFormatter('%.5f'))

        if x_lim is not None:
            axs[idx].set_xlim(x_lim)
        axs[idx].set_xlabel('Time')
        axs[idx].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        if idx:
            # don't show labels on upper plot (if two plots)
            # use the visibility to allow cursor x-position to be shown
            axs[idx].tick_params(labelbottom=False)
            axs[idx].xaxis.get_label().set_visible(False)

        if not two_plots or idx:
            # reorder legend
            axs[idx].legend(handles, [h.get_label() for h in handles],
                           loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=3,)
    return fig


def _save_options(opt):
    if opt.output:
        out_path = Path(opt.output).parent
        out_path.mkdir(exist_ok=True, parents=True)
        save_options_to_config(str(out_path / formats.get_config_filename(__file__)),
                               OrderedDict(sorted(opt.items()))
                               )

# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    main()
