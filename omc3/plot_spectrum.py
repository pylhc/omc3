"""
Plot Spectrum
--------------------

Takes data from frequency analysis and creates a stem-frequency plot for every given BPM - or all in a single figure -
with the possibility to include spectral lines.
Optionally, a waterfall plot for all BPMs is created as well.
Plots are saved in a sub-directory of the given output dir with the name of the original TbT file.
Returns two dictionaries with the filename as keys, where the dictionary first
one contains the stem plot(s - subdict with bpms as keys) and the second one contains the waterfall plot.

*--Required--*

- **files**: List with basenames of Tbt files, ie. tracking.sdds

*--Optional--*

- **amp_limit** *(float)*: All amplitudes <= limit are filtered.
  This value needs to be at least 0 to filter non-found frequencies.

  Default: ``0.0``
- **bpms**: List of BPMs for which spectra will be plotted. If not given all BPMs are used.

- **filetype** *(str)*: Filetype to save plots as (i.e. extension without ".")

  Default: ``pdf``
- **hide_bpm_labels**: Hide the bpm labels in the plots.

  Action: ``store_true``
- **lines_manual** *(DictAsString)*: List of manual lines to plot. Need to contain arguments for axvline,
  and may contain the additional key "loc" which is one of ['bottom', 'top', 'line bottom', 'line top']
  and places the label as text at the given location.

  Default: ``[]``
- **lines_nattune** *(tuple)*: List of natural tune lines to plot

  Default: ``[(1, 0), (0, 1)]``
- **lines_tune** *(tuple)*: list of tune lines to plot

  Default: ``[(1, 0), (0, 1)]``
- **manual_style** *(DictAsString)*: Additional Style parameters which update the set of predefined ones.

- **output_dir** *(str)*: Directory to write results to. If no option is given, plots will not be saved.

- **rescale**: Flag to rescale plots amplitude to max-line = 1

  Action: ``store_true``
- **show_plots**: Flag to show plots

  Action: ``store_true``
- **stem_plot**: Flag to create stem plot

  Action: ``store_true``
- **bpms_single_fig**: Flag to plot given bpms into one single stem-plot

  Action: ``store_true``
- **waterfall_cmap** *(str)*: Colormap to use for waterfall plot.

  Default: ``inferno``
- **waterfall_line_width**: Line width of the waterfall frequency lines. "auto" fills them up until the next one.

  Default: ``2``
- **waterfall_plot**: Flag to create waterfall plot.

  Action: ``store_true``
- **xlim** *(float)*: Limits on the x axis (Tupel)

  Default: ``[0, 0.5]``
- **ylim** *(float)*: Limits on the y axis (Tupel)

  Default: ``[1e-09, 1.0]``

"""
import os
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tfs
from cycler import cycler
from generic_parser.entry_datatypes import DictAsString
from generic_parser.entrypoint_parser import entrypoint, EntryPointParameters, save_options_to_config, DotDict
from matplotlib import cm, colors, transforms, lines as mlines
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pandas as pd

from definitions import formats
from utils import logging_tools
from harpy.constants import FILE_AMPS_EXT, FILE_FREQS_EXT, FILE_LIN_EXT

LOG = logging_tools.getLogger(__name__)

PLANES = ('x', 'y')

STEM_LINES_ALPHA = 0.5
RESONANCE_LINES_ALPHA = 0.5
PATCHES_ALPHA = 0.2

LABEL_Y_SPECTRUM = 'Amplitude in {plane:s} [a.u]'
LABEL_Y_WATERFALL = 'Plane {plane:s}'
LABEL_X = 'Frequency [tune units]'

NCOL_LEGEND = 5  # number of columns in the legend
WATERFALL_FILENAME = "waterfall_spectrum"
SPECTRUM_FILENAME = "spectrum"
CONFIG_FILENAME = "plot_spectrum_{time:s}.ini"

AMPS = FILE_AMPS_EXT.format(plane='')
FREQS = FILE_FREQS_EXT.format(plane='')
LIN = FILE_LIN_EXT.format(plane='')

COL_NAME = 'NAME'

MANUAL_LOCATIONS = {
    'bottom': dict(y=-0.01, va='top', ha='center'),
    'top': dict(y=1.01, va='bottom', ha='center'),
    'line bottom': dict(y=0.01, va='bottom', ha='right', rotation=90),
    'line top': dict(y=0.99, va='top', ha='right', rotation=90),
}


def get_reshuffled_tab20c():
    """ Reshuffel tab20c so that the colors change between next lines.
    Needs to be up here as it is used in DEFAULTS which is loaded early."""
    tab20c = cm.get_cmap('tab20c').colors
    out = [None] * 20
    step, chunk = 4, 5
    for idx in range(step):
        start = idx * chunk
        out[start:start + chunk] = tab20c[idx::step]
    return cycler(color=out)


DEFAULTS = DotDict(
    waterfall_cmap='inferno',
    ylim=[1e-9, 1 ** .2],
    xlim=[0, .5],
    filetype='pdf',
    waterfall_line_width=2,
    manual_style={
        u'figure.figsize': [18, 9],
        u'axes.labelsize': 15,
        u'axes.prop_cycle': get_reshuffled_tab20c(),
        u'lines.linestyle': '-',
        u'lines.marker': 'o',
        u'lines.markersize': 3,
        u'markers.fillstyle': u'none',
        u'figure.subplot.hspace': 0.3,  # space between subplots
    }
)


def get_params():
    params = EntryPointParameters()
    params.add_parameter(name="files",
                         required=True,
                         nargs='+',
                         help='List with basenames of Tbt files, ie. tracking.sdds')
    params.add_parameter(name="output_dir",
                         type=str,
                         help='Directory to write results to. If no option is given, plots will not be saved.')
    params.add_parameter(name="bpms",
                         nargs='+',
                         help='List of BPMs for which spectra will be plotted. If not given all BPMs are used.')
    params.add_parameter(name="amp_limit",
                         type=float,
                         default=0.,
                         help='All amplitudes <= limit are filtered. '
                              'This value needs to be at least 0 to filter non-found frequencies.')
    params.add_parameter(name="rescale",
                         action="store_true",
                         help='Flag to rescale plots amplitude to max-line = 1')
    params.add_parameter(name="plot_type",
                         nargs="+",
                         choices=['stem', 'waterfall'],
                         default=['stem'],
                         help='Choose plot type (Multiple choices possible).')
    params.add_parameter(name="bpms_single_fig",
                         action="store_true",
                         help='Flag to plot given bpms into one single stem-plot')
    params.add_parameter(name="files_single_fig",
                         action="store_true",
                         help='Flag to plot given files into the same plots (both stem and waterfall)')
    params.add_parameter(name="waterfall_line_width",
                         default=DEFAULTS.waterfall_line_width,
                         help='Line width of the waterfall frequency lines. "auto" fills them up until the next one.')
    params.add_parameter(name="waterfall_cmap",
                         type=str,
                         default=DEFAULTS.waterfall_cmap,
                         help="Colormap to use for waterfall plot.")
    params.add_parameter(name="waterfall_common_plane_colors",
                         action="store_true",
                         help="Same colorbar scale for both planes in waterfall plots.")
    params.add_parameter(name="show_plots",
                         action="store_true",
                         help='Flag to show plots')
    params.add_parameter(name="lines_tune",
                         nargs="*",
                         type=tuple,
                         default=[(1, 0), (0, 1)],
                         help='list of tune lines to plot')
    params.add_parameter(name="lines_nattune",
                         nargs="*",
                         type=tuple,
                         default=[(1, 0), (0, 1)],
                         help='List of natural tune lines to plot')
    params.add_parameter(name="lines_manual",
                         nargs="*",
                         default=[],
                         type=DictAsString,
                         help='List of manual lines to plot. Need to contain arguments for axvline, and may contain '
                              f'the additional key "loc" which is one of {list(MANUAL_LOCATIONS.keys())} '
                              'and places the label as text at the given location.')
    params.add_parameter(name="xlim",
                         nargs=2,
                         type=float,
                         default=DEFAULTS.xlim,
                         help='Limits on the x axis (Tupel)')
    params.add_parameter(name="ylim",
                         nargs=2,
                         type=float,
                         default=DEFAULTS.ylim,
                         help='Limits on the y axis (Tupel)')
    params.add_parameter(name="ncol_legend",
                         type=int,
                         default=NCOL_LEGEND,
                         help='Number of bpm legend-columns. If < 1 no legend is shown.')
    params.add_parameter(name="filetype",
                         type=str,
                         default=DEFAULTS.filetype,
                         help='Filetype to save plots as (i.e. extension without ".")')
    params.add_parameter(name="manual_style",
                         type=DictAsString,
                         help='Additional Style parameters which update the set of predefined ones.')
    return params


# Main -------------------------------------------------------------------------


@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.info("Starting spectrum plots.")
    if opt.output_dir is not None:
        _save_options_to_config(opt)

    opt = _check_opt(opt)
    matplotlib.rcParams.update(opt.manual_style)
    stem_opt, waterfall_opt, sorting_opt = _sort_opt(opt)
    stem, waterfall = _sort_input_data(sorting_opt)

    if stem_opt.plot:
        _create_stem_plots(stem.figs, stem_opt)

    if waterfall_opt.plot:
        _create_waterfall_plot(waterfall.figs, waterfall_opt)

    return stem.fig_list, waterfall.fig_list


# Data Sorting -----------------------------------------------------------------


class FigureContainer(object):
    """ Container for attaching additional information to one figure. """
    def __init__(self, path: str) -> None:
        self.fig, self.axes = plt.subplots(nrows=len(PLANES), ncols=1)
        self.data = {}  # hint: needs to be ordered. Which is the case for python3 dicts!
        self.tunes = {p: [] for p in PLANES}
        self.nattunes = {p: [] for p in PLANES}
        self.path = path
        self.minmax = {p: (1, 0) for p in PLANES}

    def add_data(self, label: str, new_data: dict):
        self.data[label] = new_data
        for plane in PLANES:
            # Add tunes
            self.tunes[plane].append(new_data[plane][LIN].loc[f'TUNE{plane.upper()}'])
            with suppress(KeyError):
                self.nattunes[plane].append(new_data[plane][LIN].loc[f'NATTUNE{plane.upper()}'])

            # update min/max
            mmin, mmax = self.minmax[plane]
            self.minmax[plane] = (
                min(mmin, new_data[plane][AMPS].min(skipna=True)),
                max(mmax, new_data[plane][AMPS].max(skipna=True))
            )


@dataclass
class IdData:
    """ Container to keep track of the id-sorting output """
    id: str     # id for the figure-container dictionary
    label: str  # plot labels
    path: str   # figure output path


@dataclass
class FigureCollector:
    """ Class to collect figure containers and manage data adding. """
    fig_list: dict   # dictionary of matplotlib figures, for output
    figs: dict       # dictionary of FigureContainers, for this routine

    def add_data_for_id(self, id_data: IdData, data: dict):
        """ Add the data at the appropriate figure container. """
        try:
            figure_cont = self.figs[id_data.id]
        except KeyError:
            figure_cont = FigureContainer(id_data.path)
            self.figs[id_data.id] = figure_cont
            self.fig_list[id_data.id] = figure_cont.fig
        figure_cont.add_data(id_data.label, data)


def _sort_input_data(opt: DotDict) -> tuple:
    """ Load and sort input data by file and bpm and assign correct figure-containers. """
    LOG.debug("Sorting input data.")

    stem_figs = FigureCollector({}, {})
    waterfall_figs = FigureCollector({}, {})

    # Data Sorting
    for file_path, filename in _get_unique_filenames(opt.files):
        LOG.info(f"Loading data for file '{filename}'.")

        data = _load_spectrum_data(file_path, opt.bpms)
        data = _filter_amps(data, opt.amp_limit)
        bpms = _get_all_bpms(_get_bpms(data[LIN], opt.bpms, file_path))

        for collector, get_id_fun, active in ((stem_figs, _get_stem_id, opt.plot_stem),
                                              (waterfall_figs, _get_waterfall_id, opt.plot_waterfall)):
            if not active:
                continue

            for bpm in bpms:
                the_id = get_id_fun(filename, bpm,
                                    opt.output_dir, opt.files_single_fig, opt.bpms_single_fig, opt.filetype)
                collector.add_data_for_id(the_id, _get_data_for_bpm(data, bpm, opt.rescale))
    return stem_figs, waterfall_figs


def _get_data_for_bpm(data: dict, bpm: str, rescale: bool) -> dict:
    """ Loads data from files and returns a dictionary (over planes) of a dictionary over the files containing
    the bpm data as pandas series. """
    data_series = {p: {} for p in PLANES}
    for plane in PLANES:
        try:
            freqs = data[FREQS][plane].loc[:, bpm]
            amps = data[AMPS][plane].loc[:, bpm]
            lin = data[LIN][plane].loc[bpm, :]
        except KeyError:  # bpm not in this plane
            data_series[plane] = None
        else:
            idxs_data = _get_valid_indices(amps, freqs)
            data_series[plane][LIN] = lin
            data_series[plane][FREQS] = freqs.loc[idxs_data]
            data_series[plane][AMPS] = amps.loc[idxs_data]
            if rescale:
                data_series[plane][AMPS] = _rescale_amp(data_series[plane][AMPS])

            if any(data_series[plane][AMPS].isna()):
                raise Exception("NAN FOUND")
    return data_series


def _get_stem_id(filename: str, bpm: str, output_dir: str,
                 files_single_fig: bool, bpms_single_fig: bool, filetype: str) -> IdData:
    """ Returns the stem-dictionary id and the path to which the output file should be written.
    By using more or less unique identifiers, this controls the creation of figures in the dictionary."""
    fun_map = {
        (True, True): _get_id_single_fig_files_and_bpms,
        (True, False): _get_id_single_fig_files,
        (False, True): _get_id_single_fig_bpms,
        (False, False): _get_id_multi_fig,
    }
    return fun_map[(files_single_fig, bpms_single_fig)](
        output_dir, SPECTRUM_FILENAME, filename, bpm, filetype
    )


def _get_waterfall_id(filename: str, bpm: str, output_dir: str,
                      files_single_fig: bool, bpms_single_fig: bool, filetype: str) -> IdData:
    """ Returns the waterfall-dictionary id and the path to which the output file should be written.
    By using identifiers for figures and unique lables per figure,
    this controls the creation of figures in the dictionary."""
    fun_map = {
        (True, True): _get_id_single_fig_files_and_bpms,
        (True, False): _get_id_single_fig_files,
        (False, True): _get_id_single_fig_bpms,
        (False, False): _get_id_single_fig_bpms,  # single figure per file AND bpm does not make sense for waterfall
    }
    return fun_map[(files_single_fig, bpms_single_fig)](
        output_dir, WATERFALL_FILENAME, filename, bpm, filetype
    )


# IdData Mapping ---

def _get_id_single_fig_files_and_bpms(output_dir: str, default_name: str, filename: str, bpm: str, filetype: str) -> IdData:
    """ Same id for all plots. Creates single figure. The label of the lines is a combination of filename and bpm. """
    return IdData(
        id=default_name,
        label=f"{filename} {bpm}",
        path=_get_figure_path(output_dir, filename=None, figurename=f"{default_name}.{filetype}")
    )


def _get_id_single_fig_files(output_dir: str, default_name: str, filename: str, bpm: str, filetype: str) -> IdData:
    """ BPM as id for plots. Creates len(bpm) figures, with filenames as labels for lines. """
    return IdData(
        id=bpm,
        label=filename,
        path=_get_figure_path(output_dir, filename=None, figurename=f"{default_name}_{bpm}.{filetype}")
    )


def _get_id_single_fig_bpms(output_dir: str, default_name: str, filename: str, bpm: str, filetype: str) -> IdData:
    """ Filename as ID for plots. Creates len(files) figures, with bpms as lables for lines."""
    return IdData(id=filename,
                  label=bpm,
                  path=_get_figure_path(output_dir, filename=filename, figurename=f"{default_name}.{filetype}")
                  )


def _get_id_multi_fig(output_dir: str, default_name: str, filename: str, bpm: str, filetype: str) -> IdData:
    """ Combination of Filename and BPM as ID. Creates len(files)*len(bpms) plots. BPM-name is printed as label."""
    return IdData(id=f"{filename}_{bpm}",
                  label=bpm,
                  path=_get_figure_path(output_dir, filename=filename, figurename=f"{default_name}_{bpm}.{filetype}")
                  )


# Stem Plotting ----------------------------------------------------------------


def _create_stem_plots(figures: dict, opt: DotDict) -> None:
    """ Main loop for stem-plot creation. """
    LOG.debug(f"  ...creating Stem Plots")
    for fig_id, fig_cont in figures.items():
        LOG.debug(f'   Plotting Figure: {fig_id}.')
        fig_cont.fig.canvas.set_window_title(fig_id)

        _plot_stems(fig_cont)
        _plot_lines(fig_cont, opt.lines)
        _format_stem_axes(fig_cont, opt.limits)
        if opt.ncol_legend > 0:
            _create_legend(fig_cont.axes[0], fig_cont.data.keys(), opt.lines, opt.ncol_legend)
        _output_plot(fig_cont)

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
            markers, stems, base = ax.stem(data[plane][FREQS], data[plane][AMPS],
                                           use_line_collection=True, basefmt='none', label=label)

            # Set appropriate colors
            color = get_cycled_color(idx_data)
            markers.set_markeredgecolor(color)
            stems.set_color(color)
            stems.set_alpha(STEM_LINES_ALPHA)
            LOG.debug(f"    {label} {plane}: color={color}, nstems={len(data[plane][FREQS])}")


# Finalizing ---


def _format_stem_axes(fig_cont: FigureContainer, limits: DotDict):
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
    legend_width = leg.get_window_extent().inverse_transformed(leg.axes.transAxes).width
    x_shift = 0
    if legend_width > 1:
        x_shift = (legend_width - 1) / 2.  # shift more into center

    # move above line-labels
    nlines = sum([line is not None for line in lines.values()]) + 0.05
    y_shift = get_approx_size_in_axes_coordinates(leg.axes, label_size=matplotlib.rcParams['axes.labelsize']) * nlines
    leg.axes.legend(handles=handles, bbox_to_anchor=(1. + x_shift, 1. + y_shift), **legend_params)


# Waterfall Plotting -----------------------------------------------------------


def _create_waterfall_plot(figures: dict, opt: DotDict) -> None:
    LOG.debug(f"  ...creating Waterfall Plot")

    for fig_id, fig_cont in figures.items():
        LOG.debug(f'   Plotting Figure: {fig_id}.')
        fig_cont.fig.canvas.set_window_title(fig_id)

        _plot_waterfall(fig_cont, opt.line_width, opt.cmap, opt.common_plane_colors)
        _plot_lines(fig_cont, opt.lines)
        _format_waterfall_axes(fig_cont, opt.limits, opt.ncol_legend)
        _output_plot(fig_cont)

    if opt.show:
        plt.show()


def _plot_waterfall(fig_cont, line_width, cmap, common_plane_colors):
    for idx_plane, plane in enumerate(PLANES):
        ax = fig_cont.axes[idx_plane]
        norm = _get_waterfall_norm(fig_cont.minmax, plane, common_plane_colors)
        for idx_data, (label, data) in enumerate(fig_cont.data.items()):
            if data[plane] is None:
                continue
            freqs = data[plane][FREQS]
            amps = data[plane][AMPS]

            if line_width == "auto":
                freqs = freqs.sort_values()
                amps = amps.loc[freqs.index]

                f_values = freqs.to_numpy().T
                freqs_mesh = np.tile(np.array([*f_values, .5]), [2, 1])
                y_mesh = np.tile([idx_data - 0.5, idx_data + 0.5], [len(freqs) + 1, 1]).T
                ax.pcolormesh(freqs_mesh, y_mesh, amps.to_frame().T, cmap=cmap, norm=norm, zorder=-3)
            else:
                lc = ax.vlines(x=freqs, ymin=idx_data - .5, ymax=idx_data + .5,
                               linestyles='solid', cmap=cmap, norm=norm,
                               linewidths=line_width, zorder=-3,
                               )
                lc.set_array(amps)  # sets the colors of the segments
        ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)


def _get_waterfall_norm(minmax, plane, common):
    """ Get color norm either from ylimits (if given) else from min and max values. """
    if not common:
        return colors.LogNorm(*minmax[plane])

    return colors.LogNorm(min(mm[0] for mm in minmax.values()),
                          max(mm[1] for mm in minmax.values()))


# Finalize ---


def _format_waterfall_axes(fig_cont, limits, ncol):
    ylabels = fig_cont.data.keys()
    for idx_plane, plane in enumerate(PLANES):
        ax = fig_cont.axes[idx_plane]
        if ncol < 1:
            ax.set_yticklabels([])
            ax.set_yticks([])
        else:
            ax.set_yticklabels(ylabels, fontdict={'fontsize': matplotlib.rcParams[u'axes.labelsize'] * .5})
            ax.set_yticks(range(len(ylabels)))
        ax.set_xlabel(LABEL_X)
        ax.set_ylabel(LABEL_Y_WATERFALL.format(plane=plane.upper()))
        ax.set_xlim(limits.xlim)
        ax.set_ylim([-.5, len(ylabels) - .5])
        ax.tick_params(axis='x', which='both')


# Plot Lines -------------------------------------------------------------------


def _plot_lines(fig_cont, lines):
    label_size = matplotlib.rcParams['axes.labelsize'] * 0.7
    bottom_qlabel = 1.01

    for idx_plane, plane in enumerate(PLANES):
        ax = fig_cont.axes[idx_plane]
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        bottom_natqlabel = bottom_qlabel + 2 * get_approx_size_in_axes_coordinates(ax, label_size)

        # Tune Lines ---
        for line_params in (("", fig_cont.tunes, lines.tune, "--", bottom_qlabel),
                            ("NAT", fig_cont.nattunes, lines.nattune, ":", bottom_natqlabel)):
            _plot_tune_lines(ax, trans, label_size, *line_params)

        # Manual Lines ---
        for mline in lines.manual:
            _plot_manual_line(ax, mline, trans, label_size)


def _plot_tune_lines(ax, transform, label_size, q_string, tunes, resonances, linestyle, label_y):
    if len(resonances) == 0:
        return

    pref = q_string[0] if len(q_string) else ""
    q_mean = np.array([np.mean(tunes[p]) for p in PLANES])
    q_min = np.array([np.min(tunes[p]) for p in PLANES])
    q_max = np.array([np.max(tunes[p]) for p in PLANES])
    freqs_mean = _get_resonance_frequencies(resonances, q_mean)
    freqs_min = _get_resonance_frequencies(resonances, q_min)
    freqs_max = _get_resonance_frequencies(resonances, q_max)
    for res, f_mean, f_min, f_max in zip(resonances, freqs_mean, freqs_min, freqs_max):
        if not np.isnan(f_mean):
            label, order = f'{pref}({res[0]}, {res[1]})', sum(np.abs(res)) + 1
            color = get_cycled_color(order-2)
            ax.axvline(x=f_mean, label=label,
                       linestyle=linestyle, color=color, marker='None',
                       zorder=-1, alpha=RESONANCE_LINES_ALPHA)
            ax.text(x=f_mean, y=label_y, s=label, transform=transform,
                    color=color,
                    va='bottom', ha='center',
                    fontdict={'size': label_size})
            ax.add_patch(Rectangle(xy=(f_min, 0), width=f_max-f_min, height=1,
                                   transform=transform, color=color, alpha=PATCHES_ALPHA, zorder=-2,))


def _plot_manual_line(ax, mline, transform, label_size):
    mline.setdefault('alpha', RESONANCE_LINES_ALPHA)
    mline.setdefault('marker', 'None')
    mline.setdefault('zorder', -1)
    if 'linestyle' not in mline and 'ls' not in mline:
        mline['linestyle'] = '--'

    label = mline.get('label', None)
    loc = mline.pop('loc', None)  # needs to be removed in axvline

    line = ax.axvline(**mline)
    mline['loc'] = loc  # reset it for later axes/plots

    if label is not None and loc is not None:
        if loc not in MANUAL_LOCATIONS:
            raise ValueError(f"Unknown value '{loc}' for label location.")

        ax.text(x=mline['x'], s=label, transform=transform,
                color=line.get_color(), fontdict={'size': label_size}, **MANUAL_LOCATIONS[loc])


def _get_resonance_frequencies(resonances, q):
    """ Calculates the frequencies for the resonance lines, but also filters lines in case the tune was not found. """
    resonances = np.array(resonances)

    # find zero-tune filter: if tune in plane is not used (i.e. coefficient is zero) we can still plot the line
    use_idx = np.ones(resonances.shape[0], dtype=bool)
    for idx, tune in enumerate(q):
        if tune == 0:
            use_idx &= resonances[:, idx] == 0

    if sum(use_idx) == 0:
        LOG.warning("No usable tunes found to calculate resonance frequencies. "
                    " Maybe you gave natural lines for a free kick?")

    freqs = np.mod(resonances @ q, 1)
    freqs = np.where(freqs > .5, 1 - freqs, freqs)

    freqs.dtype = np.float64  # in case of all zeros, this is int and causes crash with float-nan
    freqs[~use_idx] = np.nan
    return freqs


# Plot Helpers -----------------------------------------------------------------


def get_cycled_color(idx):
    """ Get the color at (wrapped) idx in the color cycle. The CN-Method only works until 'C9'."""
    cycle = matplotlib.rcParams[u"axes.prop_cycle"].by_key()['color']
    return cycle[idx % len(cycle)]


def get_approx_size_in_axes_coordinates(ax, label_size):
    transform = ax.transAxes.inverted().transform
    _, label_size_ax = transform((0, label_size)) - transform((0, 0))
    return label_size_ax


# Input ------------------------------------------------------------------------


def _check_opt(opt):
    if (opt.waterfall_line_width is not None and opt.waterfall_line_width != DEFAULTS.waterfall_line_width
            and 'waterfall' not in opt.plot_type):
        LOG.warning("Setting 'waterfall_line_width' option has no effect, when waterfall plots are deactivated!")

    if (opt.waterfall_cmap is not None and opt.waterfall_cmap != DEFAULTS.waterfall_cmap
            and 'waterfall' not in opt.plot_type):
        LOG.warning("Setting 'waterfall_cmap' option has no effect, when waterfall plots are deactivated!")

    if opt.amp_limit < 0:
        raise ValueError("The amplitude limit needs to be at least '0' to filter for non-found frequencies.")

    style_dict = DEFAULTS['manual_style']
    if opt.manual_style is not None:
        style_dict.update(opt.manual_style)
    opt.manual_style = style_dict

    return opt


def _sort_opt(opt):
    # lines structure
    lines = opt.get_subdict(('lines_tune', 'lines_nattune', 'lines_manual'))
    lines = _rename_dict_keys(lines, to_remove="lines_")
    for key, val in lines.items():
        if val is None:
            lines[key] = []

    # limits structure
    limits = opt.get_subdict(("xlim", "ylim"))

    # stem-plot options
    stem = opt.get_subdict(('ncol_legend',))
    stem['plot'] = 'stem' in opt.plot_type

    # waterfall-plot options
    waterfall = opt.get_subdict(('waterfall_line_width', 'waterfall_cmap', 'waterfall_common_plane_colors',
                                 'ncol_legend'))
    waterfall = _rename_dict_keys(waterfall, to_remove="waterfall_")
    waterfall['plot'] = 'waterfall' in opt.plot_type

    # needed in both
    for d in (stem, waterfall):
        d['show'] = opt['show_plots']
        d['limits'] = limits
        d['lines'] = lines

    # sorting options
    sort = opt.get_subdict(('files_single_fig', 'bpms_single_fig',
                           'filetype', 'files', 'bpms', 'output_dir', 'amp_limit', 'rescale'))
    sort['plot_stem'] = stem.plot
    sort['plot_waterfall'] = waterfall.plot

    return stem, waterfall, sort


def _get_unique_filenames(files):
    """ Way too complicated method to assure unique dictionary names."""
    def _get_filename(path, nparts):
        return "_".join(os.path.split(path)[nparts:])

    paths = [None] * len(files)
    names = [None] * len(files)
    parts = -1
    for idx, fpath in enumerate(files):
        fname = _get_filename(fpath, parts)
        while fname in names:
            parts -= 1
            for idx_old in range(idx):
                names[idx_old] = _get_filename(paths[idx_old], parts)
            fname = _get_filename(fpath, parts)
        names[idx] = fname
        paths[idx] = fpath
    return zip(paths, names)


def _load_spectrum_data(file_path, bpms):
    LOG.info("Loading HARPY data.")
    with suppress(FileNotFoundError):
        return _get_harpy_data(file_path)

    LOG.info("Some files not present. Loading SUSSIX data format")
    with suppress(FileNotFoundError):
        return _get_sussix_data(file_path, bpms)

    raise FileNotFoundError(f"Neither harpy nor sussix files found in '{os.path.dirname(file_path)}' "
                            f"matching the name '{os.path.basename(file_path)}'.")

# Harpy Loader ---


def _get_harpy_data(file_path):
    return {
        AMPS: _get_amplitude_files(file_path),
        FREQS: _get_frequency_files(file_path),
        LIN: _get_lin_files(file_path),
    }


def _get_amplitude_files(file_path):
    return _get_planed_files(file_path, ext=FILE_AMPS_EXT)


def _get_frequency_files(file_path):
    return _get_planed_files(file_path, ext=FILE_FREQS_EXT)


def _get_lin_files(file_path):
    return _get_planed_files(file_path, ext=FILE_LIN_EXT, index=COL_NAME)


def _get_planed_files(file_path, ext, index=None):
    directory, filename = _get_dir_and_name(file_path)
    return {
        plane: tfs.read(os.path.join(directory, f'{filename}{ext.format(plane=plane.lower())}'), index=index)
        for plane in PLANES
    }


# Sussix loader ---


def _get_sussix_data(file_path, bpms):
    directory, filename = _get_dir_and_name(file_path)
    bpm_dir = os.path.join(directory, 'BPM')
    files = {LIN: {}, AMPS: {}, FREQS: {}}
    for plane in PLANES:
        files[LIN][plane] = tfs.read(os.path.join(directory, f'{filename}_lin{plane}'), index=COL_NAME)
        for id_ in (FREQS, AMPS):
            files[id_][plane] = tfs.TfsDataFrame(columns=bpms)
        for bpm in bpms:
            with suppress(FileNotFoundError):
                df = tfs.read(os.path.join(bpm_dir, f'{bpm}.{plane}'))
                files[FREQS][plane][bpm] = df["FREQ"]
                files[AMPS][plane][bpm] = df["AMP"]
        for id_ in (FREQS, AMPS):
            files[id_][plane] = files[id_][plane].fillna(0)
    return files


# Other ---


def _get_dir_and_name(file_path):
    return os.path.dirname(file_path), os.path.basename(file_path)


def _get_bpms(lin_files, given_bpms, file_path):
    found_bpms = {}
    for plane in PLANES:
        found_bpms[plane] = list(lin_files[plane].index)
        if given_bpms is not None:
            found_bpms[plane] = [bpm for bpm in found_bpms[plane] if bpm in given_bpms]
            bpms_not_found = [bpm for bpm in given_bpms if bpm not in found_bpms[plane]]
            if len(bpms_not_found):
                LOG.warning(
                    f"({file_path}) The following BPMs are not present or not present in plane {plane}:"
                    f" {list2str(bpms_not_found)}"
                )
        if len(found_bpms[plane]) == 0:
            LOG.warning(f"({file_path}) No BPMs found for plane {plane} !")

    if not any([len(bpms) for bpms in found_bpms.values()]):
        raise IOError(f"({file_path}) No BPMs found in any plane!")
    return found_bpms


def _get_valid_indices(amps, freqs):
    """ Intersection of filtered AMPS and FREQS indices. """
    return index_filter(amps).intersection(index_filter(freqs))


def index_filter(data):
    """ Only non-NaN and non-Zero data allowed. (Amps should not be zero due to _filter_amps() anyway.)"""
    return data[~(data.isna() | (data == 0))].index


def _filter_amps(files, limit):
    for plane in PLANES:
        filter_idx = files[AMPS][plane] <= limit
        files[AMPS][plane][filter_idx] = np.NaN
        files[FREQS][plane][filter_idx] = np.NaN
    return files


# Output -----------------------------------------------------------------------


def _save_options_to_config(opt):
    os.makedirs(opt.output_dir, exist_ok=True)
    save_options_to_config(os.path.join(opt.output_dir, _get_ini_filename()), OrderedDict(sorted(opt.items())))


def _get_figure_path(out_dir, filename, figurename):
    path = _make_output_dir(out_dir, filename)
    if path is not None and figurename is not None:
        path = os.path.join(path, figurename)
    return path


def _make_output_dir(out_dir, filename):
    if out_dir is not None:
        if filename is not None:
            out_dir = os.path.join(out_dir, os.path.splitext(filename)[0])
        os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _get_ini_filename():
    return CONFIG_FILENAME.format(time=datetime.utcnow().strftime(formats.TIME))


def _output_plot(fig_cont):
    fig = fig_cont.fig

    fig.tight_layout()
    fig.tight_layout()  # sometimes better to do twice

    if fig_cont.path is not None:
        LOG.info(f"Saving Plot '{fig_cont.path}'")
        fig.savefig(fig_cont.path)


# Helper -----------------------------------------------------------------------


def list2str(list_):
    return str(list_)[1:-1]


def _rescale_amp(amp_data):
    # return amp_data.divide(amp_data.max(axis=0), axis=1)  # dataframe
    return amp_data.divide(amp_data.max(skipna=True))  # series


def _get_all_bpms(bpms_dict):
    """ Returns a union of all bpms for both planes """
    return set.union(*[set(v) for v in bpms_dict.values()])


def _rename_dict_keys(d, to_remove):
    for key in list(d.keys()):  # using list to copy keys
        d[key.replace(to_remove, "")] = d.pop(key)
    return d


if __name__ == "__main__":
    main()
