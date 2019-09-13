"""
Module tune_analysis.bbq_tools
----------------------------------

Tools to handle BBQ data.

This package contains a collection of tools to handle and modify BBQ data:
 - Calculating moving average
 - Plotting
"""

import datetime
import os

import matplotlib.dates as mdates
import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib import colors
from tune_analysis import constants as const
from utils import logging_tools
from plotshop import plot_style as ps

TIMEZONE = const.get_experiment_timezone()

PLANES = const.get_planes()

COL_MAV = const.get_mav_col
COL_IN_MAV = const.get_used_in_mav_col
COL_BBQ = const.get_bbq_col

LOG = logging_tools.get_logger(__name__)


def get_moving_average(data_series, length=20,
                       min_val=None, max_val=None, fine_length=None, fine_cut=None):
    """ Get a moving average of the ``data_series`` over ``length`` entries.
    The data can be filtered beforehand.
    The values are shifted, so that the averaged value takes ceil((length-1)/2) values previous
    and floor((length-1)/2) following values into account.

    Args:
        data_series: Series of data
        length: length of the averaging window
        min_val: minimum value (for filtering)
        max_val: maximum value (for filtering)
        fine_length: length of the averaging window for fine cleaning
        fine_cut: allowed deviation for fine cleaning

    Returns: filtered and averaged Series and the mask used for filtering data.
    """
    LOG.debug("Calculating BBQ moving average of length {:d}.".format(length))

    if bool(fine_length) != bool(fine_cut):
        raise NotImplementedError("To activate fine cleaning, both "
                                  "'fine_window' and 'fine_cut' are needed.")

    if min_val is not None:
        min_mask = data_series <= min_val
    else:
        min_mask = np.zeros(data_series.size, dtype=bool)

    if max_val is not None:
        max_mask = data_series >= max_val
    else:
        max_mask = np.zeros(data_series.size, dtype=bool)

    cut_mask = min_mask | max_mask
    _is_almost_empty_mask(~cut_mask, length)
    data_mav, std_mav = _get_interpolated_moving_average(data_series, cut_mask, length)

    if fine_length is not None:
        min_mask = data_series <= (data_mav - fine_cut)
        max_mask = data_series >= (data_mav + fine_cut)
        cut_mask = min_mask | max_mask
        _is_almost_empty_mask(~cut_mask, fine_length)
        data_mav, std_mav = _get_interpolated_moving_average(data_series, cut_mask, fine_length)

    return data_mav, std_mav, cut_mask


def plot_bbq_data(bbq_df,
                  interval=None, xmin=None, xmax=None, ymin=None, ymax=None,
                  output=None, show=True, two_plots=False):
    """ Plot BBQ data.

    Args:
        bbq_df: BBQ Dataframe with moving average columns
        interval: start and end time of used interval, will be marked with red bars
        xmin: Lower x limit (time)
        xmax: Upper x limit (time)
        ymin: Lower y limit (tune)
        ymax: Upper y limit (tune)
        output: Path to the output file
        show: Shows plot if `True`
        two_plots: Plots each tune in it's own axes if `True`

    Returns:
        Plotted figure

    """
    LOG.debug("Plotting BBQ data.")

    ps.set_style("standard", {
        u'figure.figsize': [12.24, 7.68],
        u"lines.marker": u"",
        u"lines.linestyle": u""}
        )

    fig = plt.figure()

    if two_plots:
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        ax = [fig.add_subplot(gs[1]), fig.add_subplot(gs[0])]
    else:
        gs = gridspec.GridSpec(1, 1, height_ratios=[1])
        ax = fig.add_subplot(gs[0])
        ax = [ax, ax]

    bbq_df.index = [datetime.datetime.fromtimestamp(time, tz=TIMEZONE) for time in bbq_df.index]

    handles = [None] * (3 * len(PLANES))
    for idx, plane in enumerate(PLANES):
        color = ps.get_mpl_color(idx)
        mask = bbq_df[COL_IN_MAV(plane)]

        # plot and save handles for nicer legend
        handles[idx] = ax[idx].plot(bbq_df.index, bbq_df[COL_BBQ(plane)],
                                    color=ps.change_color_brightness(color, .4),
                                    marker="o", markerfacecolor="None",
                                    label="$Q_{:s}$".format(plane.lower(),)
                                    )[0]
        filtered_data = bbq_df.loc[mask, COL_BBQ(plane)].dropna()
        handles[len(PLANES)+idx] = ax[idx].plot(filtered_data.index, filtered_data.values,
                                                color=ps.change_color_brightness(color, .7),
                                                marker=".",
                                                label="filtered".format(plane.lower())
                                                )[0]
        handles[2*len(PLANES)+idx] = ax[idx].plot(bbq_df.index, bbq_df[COL_MAV(plane)],
                                                  color=color,
                                                  linestyle="-",
                                                  label="moving av.".format(plane.lower())
                                                  )[0]

        if ymin is None and two_plots:
            ax[idx].set_ylim(bottom=min(bbq_df.loc[mask, COL_BBQ(plane)]))

        if ymax is None and two_plots:
            ax[idx].set_ylim(top=max(bbq_df.loc[mask, COL_BBQ(plane)]))

    # things to add/do only once if there is only one plot
    for idx in range(1+two_plots):
        if interval:
            ax[idx].axvline(x=interval[0], color="red")
            ax[idx].axvline(x=interval[1], color="red")

        if two_plots:
            ax[idx].set_ylabel("$Q_{:s}$".format(PLANES[idx]))
        else:
            ax[idx].set_ylabel('Tune')

        ax[idx].set_ylim(bottom=ymin, top=ymax)
        ax[idx].yaxis.set_major_formatter(FormatStrFormatter('%.5f'))

        ax[idx].set_xlim(left=xmin, right=xmax)
        ax[idx].set_xlabel('Time')
        ax[idx].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        if idx:
            # don't show labels on upper plot (if two plots)
            # use the visibility to allow cursor x-position to be shown
            ax[idx].tick_params(labelbottom=False)
            ax[idx].xaxis.get_label().set_visible(False)

        if not two_plots or idx:
            # reorder legend
            ax[idx].legend(handles, [h.get_label() for h in handles],
                           loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=3,)

    fig.tight_layout()
    fig.tight_layout()

    if output:
        fig.savefig(output)
        ps.set_name(os.path.basename(output))

    if show:
        plt.draw()

    return fig


# Private methods ############################################################


def _get_interpolated_moving_average(data_series, clean_mask, length):
    """ Returns the moving average of data series with a window of length and interpolated NaNs"""
    data = data_series.copy()
    data[clean_mask] = np.NaN

    # 'interpolate' fills nan based on index/values of neighbours
    data = data.interpolate("index").fillna(method="bfill").fillna(method="ffill")

    shift = -int((length-1)/2)  # Shift average to middle value

    # calculate mean and std, fill NaNs at the ends
    data_mav = data.rolling(length).mean().shift(shift).fillna(
        method="bfill").fillna(method="ffill")
    std_mav = data.rolling(length).std().shift(shift).fillna(
        method="bfill").fillna(method="ffill")
    return data_mav, std_mav


def _is_almost_empty_mask(mask, av_length):
    """ Checks if masked data could be used to calculate moving average. """
    if sum(mask) <= av_length:
        raise ValueError("Too many points have been filtered. Maybe wrong tune, cutoff?")
