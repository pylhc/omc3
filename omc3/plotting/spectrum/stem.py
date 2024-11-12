"""
Plot Spectrum - Stem Plotter
-----------------------------

Stem plotting functionality for spectrum plotter.
"""
import matplotlib
from generic_parser import DotDict
from matplotlib import pyplot as plt, lines as mlines

from omc3.plotting.spectrum.utils import (plot_lines, FigureContainer, get_cycled_color,
                                          get_approx_size_in_axes_coordinates,
                                          PLANES, STEM_LINES_ALPHA, LABEL_Y_SPECTRUM,
                                          LABEL_X, AMPS, FREQS, output_plot)
from omc3.plotting.utils.annotations import get_fontsize_as_float
from omc3.utils import logging_tools

LOG = logging_tools.getLogger(__name__)


def create_stem_plots(figures: dict, opt: DotDict) -> None:
    """ Main loop for stem-plot creation. """
    LOG.debug("  ...creating Stem Plots")
    for fig_id, fig_cont in figures.items():
        LOG.debug(f'   Plotting Figure: {fig_id}.')
        fig_cont.fig.canvas.manager.set_window_title(fig_id)

        _plot_stems(fig_cont)
        plot_lines(fig_cont, opt.lines)
        _format_axes(fig_cont, opt.limits)
        if opt.ncol_legend > 0:
            _create_legend(fig_cont.axes[0], fig_cont.data.keys(), opt.lines, opt.ncol_legend)
        output_plot(fig_cont)

    if opt.show:
        plt.show()


def _plot_stems(fig_cont: FigureContainer) -> None:
    """ Plot the spectrum stems for this figure container. """
    for idx_plane, plane in enumerate(PLANES):
        ax = fig_cont.axes[idx_plane]
        for idx_data, (label, data) in enumerate(fig_cont.data.items()):
            if data[plane] is None:
                continue
            # plot
            try:
                # Matplotlib < v3.8
                markers, stems, base = ax.stem(data[plane][FREQS], data[plane][AMPS], basefmt='none', label=label,
                                               use_line_collection=True)
            except TypeError:
                # Matplotlib >= v3.8
                markers, stems, base = ax.stem(data[plane][FREQS], data[plane][AMPS], basefmt='none', label=label)

            # Set appropriate colors
            color = get_cycled_color(idx_data)
            markers.set_markeredgecolor(color)
            stems.set_color(color)
            stems.set_alpha(STEM_LINES_ALPHA)
            LOG.debug(f"    {label} {plane}: color={color}, nstems={len(data[plane][FREQS])}")


def _format_axes(fig_cont: FigureContainer, limits: DotDict):
    for idx_plane, plane in enumerate(PLANES):
        ax = fig_cont.axes[idx_plane]

        ax.set_yscale('log')
        ax.set_xlim(limits.xlim)
        ax.set_ylim(limits.ylim)
        ax.set_ylabel(LABEL_Y_SPECTRUM.format(plane=plane.upper()))
        ax.set_xlabel(LABEL_X)
        ax.tick_params(axis='both', which='major')


def _create_legend(ax, labels, lines, ncol):
    lines_params = dict(
        marker=matplotlib.rcParams[u'lines.marker'],
        markersize=matplotlib.rcParams[u'font.size'] * 0.5,
        linestyle='None',
    )
    legend_params = dict(
        loc='lower right',
        ncol=ncol,
        fancybox=False, shadow=False, frameon=False,
    )
    handles = [mlines.Line2D([], [], color=get_cycled_color(idx), label=bpm, **lines_params)
               for idx, bpm in enumerate(labels)]
    leg = ax.legend(handles=handles, **legend_params)

    leg.axes.figure.canvas.draw()  # to get the legend extend

    # check if it is wider than the axes
    legend_width = leg.get_window_extent().transformed(leg.axes.transAxes.inverted()).width
    x_shift = 0
    if legend_width > 1:
        x_shift = (legend_width - 1) / 2.  # shift more into center

    # move above line-labels
    nlines = sum([line is not None for line in lines.values()]) + 0.05
    label_size = get_fontsize_as_float(matplotlib.rcParams['axes.labelsize'])
    y_shift = get_approx_size_in_axes_coordinates(leg.axes, label_size=label_size) * nlines
    leg.axes.legend(handles=handles, bbox_to_anchor=(1. + x_shift, 1. + y_shift), **legend_params)
