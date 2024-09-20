"""
Plot Spectrum - Waterfall Plotter
---------------------------------

Waterfall plotting functionality for spectrum plotter.
"""
import matplotlib
import numpy as np
from generic_parser import DotDict
from matplotlib import pyplot as plt, colors

from omc3.plotting.spectrum.utils import (plot_lines, PLANES, LABEL_Y_WATERFALL,
                                          LABEL_X, AMPS, FREQS, output_plot)
from omc3.plotting.utils.annotations import get_fontsize_as_float
from omc3.utils import logging_tools

LOG = logging_tools.getLogger(__name__)


def create_waterfall_plots(figures: dict, opt: DotDict) -> None:
    """ Main loop for waterfall plot creation. """
    LOG.debug("  ...creating Waterfall Plot")

    for fig_id, fig_cont in figures.items():
        LOG.debug(f'   Plotting Figure: {fig_id}.')
        fig_cont.fig.canvas.manager.set_window_title(fig_id)

        _plot_waterfall(fig_cont, opt.line_width, opt.cmap, opt.common_plane_colors)
        plot_lines(fig_cont, opt.lines)
        _format_axes(fig_cont, opt.limits, opt.ncol_legend)
        output_plot(fig_cont)

    if opt.show:
        plt.show()


def _plot_waterfall(fig_cont, line_width, cmap, common_plane_colors):
    """ Create the waterfall plot for this figure container. """
    for idx_plane, plane in enumerate(PLANES):
        ax = fig_cont.axes[idx_plane]
        norm = _get_color_norm(fig_cont.minmax, plane, common_plane_colors)
        for idx_data, (label, data) in enumerate(fig_cont.data.items()):
            if data[plane] is None:
                continue

            freqs = data[plane][FREQS]
            amps = data[plane][AMPS]

            if line_width == "auto":
                _plot_color_mesh(ax, freqs, amps, idx_data, cmap, norm)
            else:
                _plot_vlines(ax, freqs, amps, idx_data, cmap, norm, line_width)
        ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)


def _get_color_norm(minmax, plane, common):
    """ Returns the color-norm calculated from the min and max values. """
    if not common:
        return colors.LogNorm(*minmax[plane])

    return colors.LogNorm(min(mm[0] for mm in minmax.values()),
                          max(mm[1] for mm in minmax.values()))


def _plot_color_mesh(ax, freqs, amps, idx_data, cmap, norm):
    """ Plots the frequencies as a mesh, with amplitudes as colors. """
    freqs = freqs.sort_values()
    amps = amps.loc[freqs.index]

    freqs_mesh = np.tile(np.array([*freqs.to_numpy().T, .5]), [2, 1])
    y_mesh = np.tile([idx_data - 0.5, idx_data + 0.5], [len(freqs) + 1, 1]).T
    ax.pcolormesh(freqs_mesh, y_mesh, amps.to_frame().T,
                  cmap=cmap, norm=norm, zorder=-3)


def _plot_vlines(ax, freqs, amps, idx_data, cmap, norm, line_width):
    """ Plots the frequencies as vertical lines, with amplitudes as colors. """
    lines = ax.vlines(x=freqs, ymin=idx_data - .5, ymax=idx_data + .5,
                      linestyles='solid', cmap=cmap, norm=norm,
                      linewidths=line_width, zorder=-3,
                      )
    lines.set_array(amps)  # sets the colors of the segments


def _format_axes(fig_cont, limits, ncol):
    ylabels = fig_cont.data.keys()
    for idx_plane, plane in enumerate(PLANES):
        ax = fig_cont.axes[idx_plane]
        if ncol < 1:
            ax.set_yticks(ticks=[], labels=[])
        else:
            # Provide ticks and labels together or matplotlib issues a UserWarning
            # See "Discouraged" admonition at https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html 
            ax.set_yticks(
                ticks=range(len(ylabels)),
                labels=ylabels,
                fontdict={'fontsize': get_fontsize_as_float(matplotlib.rcParams[u'axes.labelsize']) * .5},
            )
        ax.set_xlabel(LABEL_X)
        ax.set_ylabel(LABEL_Y_WATERFALL.format(plane=plane.upper()))
        ax.set_xlim(limits.xlim)
        ax.set_ylim([-.5, len(ylabels) - .5])
        ax.tick_params(axis='x', which='both')
