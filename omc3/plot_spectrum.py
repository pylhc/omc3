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
- **stem_single_fig**: Flag to plot given bpms into one single stem-plot

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
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tfs
from cycler import cycler
from generic_parser.entry_datatypes import DictAsString
from generic_parser.entrypoint_parser import entrypoint, EntryPointParameters, save_options_to_config
from matplotlib import cm, colors, transforms, lines as mlines

from definitions import formats
from utils import logging_tools
from harpy.constants import FILE_AMPS_EXT, FILE_FREQS_EXT, FILE_LIN_EXT

LOG = logging_tools.getLogger(__name__)

PLANES = ('x', 'y')

STEM_LINES_ALPHA = 0.5
RESONANCE_LINES_ALPHA = 0.5

LABEL_Y_SPECTRUM = 'Amplitude in {plane:s} [a.u]'
LABEL_Y_WATERFALL = 'Items in {plane:s}'
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


DEFAULTS = dict(
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
    params.add_parameter(name="stem_plot",
                         action="store_true",
                         help='Flag to create stem plot')
    params.add_parameter(name="stem_single_fig",
                         action="store_true",
                         help='Flag to plot given bpms into one single stem-plot')
    params.add_parameter(name="waterfall_plot",
                         action="store_true",
                         help='Flag to create waterfall plot.')
    params.add_parameter(name="waterfall_line_width",
                         default=DEFAULTS['waterfall_line_width'],
                         help='Line width of the waterfall frequency lines. "auto" fills them up until the next one.')
    params.add_parameter(name="waterfall_cmap",
                         type=str,
                         default=DEFAULTS['waterfall_cmap'],
                         help="Colormap to use for waterfall plot.")
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
                         default=DEFAULTS['xlim'],
                         help='Limits on the x axis (Tupel)')
    params.add_parameter(name="ylim",
                         nargs=2,
                         type=float,
                         default=DEFAULTS['ylim'],
                         help='Limits on the y axis (Tupel)')
    params.add_parameter(name="hide_bpm_labels",
                         action="store_true",
                         help='Hide the bpm labels in the plots.')
    params.add_parameter(name="filetype",
                         type=str,
                         default=DEFAULTS['filetype'],
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

    # Input
    opt = _check_opt(opt)

    matplotlib.rcParams.update(opt.manual_style)
    out, limits, lines, stem_opt, waterfall_opt = _sort_opt(opt)

    spectrum_figs = {}
    waterfall_figs = {}

    for file_path in opt.files:
        filename = os.path.basename(file_path)
        LOG.info(f"Creating plots for file '{filename}'.")

        # Loading Data
        if opt.output_dir is not None:
            out.dir = _make_output_dir(opt.output_dir, filename)

        files = _load_spectrum_data(file_path, opt.bpms)
        files = _filter_amps(files, opt.amp_limit)
        bpms = _get_bpms(files[LIN], opt.bpms)


        # Plotting
        if stem_opt.plot:
            spectrum_figs[filename] = _create_stem_plots(out, bpms, files, lines, limits, stem_opt)

        if waterfall_opt.plot:
            waterfall_figs[filename] = _create_waterfall_plot(out, bpms, files, lines, limits, waterfall_opt)

    return spectrum_figs, waterfall_figs


# Stem Plotting ----------------------------------------------------------------


def _create_stem_plots(out, bpms, files, lines, limits, opts):
    if opts.single_fig:
        return _singlefigure_stems(out, bpms, files, lines, limits, opts)
    return _multifigure_stems(out, bpms, files, lines, limits, opts)


def _singlefigure_stems(out, bpms, files, lines, limits, opts):
    LOG.debug(f"  ...creating single figure stem plots.")
    fig, axs = plt.subplots(nrows=2, ncols=1)
    tunes = _get_tunes(files[LIN])
    all_bpms = _get_all_bpms(bpms)
    for ax, plane in zip(axs, PLANES):
        _plot_lines(ax, tunes, lines, zorder=-1)
        for idx, bpm in enumerate(all_bpms):
            if bpm not in bpms[plane]:
                LOG.info(f"    {bpm} not found in plane {plane}")
                continue
            _plot_stems(ax, files, plane, bpm, idx, opts.rescale)
    _format_stem_axes(axs, limits)
    _create_legend(axs, all_bpms, lines, opts.hide_bpm_labels)
    _output_plot(fig, out, SPECTRUM_FILENAME)
    return fig


def _multifigure_stems(out, bpms, files, lines, limits, opts):
    LOG.debug(f"  ...creating multi figure stem plots.")
    bpm_figs = {}
    all_bpms = _get_all_bpms(bpms)
    for bpm in all_bpms:
        tunes = _get_tunes(files[LIN], bpm)
        fig, axs = plt.subplots(nrows=2, ncols=1)
        for idx, (ax, plane) in enumerate(zip(axs, PLANES)):
            if bpm not in bpms[plane]:
                LOG.info(f"    {bpm} not found in plane {plane}")
                continue
            _plot_lines(ax, tunes, lines, zorder=-1)
            _plot_stems(ax, files, plane, bpm, idx, opts.rescale)
        _format_stem_axes(axs, limits)
        _create_legend(axs, bpm, lines, opts.hide_bpm_labels)
        _output_plot(fig, out, f'{bpm}_{SPECTRUM_FILENAME}')
        bpm_figs[bpm] = fig
    return bpm_figs


def _plot_stems(ax, files, plane, bpm, idx, rescale):
    idxs = _get_valid_indices(files, plane, bpm)
    freqs = files[FREQS][plane].loc[idxs, [bpm]]
    amps = files[AMPS][plane].loc[idxs, [bpm]]
    if rescale:
        amps = _rescale_amp(amps)

    markers, stems, base = ax.stem(freqs, amps, use_line_collection=True, basefmt='none', label=bpm)

    # Set appropriate colors
    color = get_cycled_color(idx)
    markers.set_markeredgecolor(color)
    stems.set_color(color)
    stems.set_alpha(STEM_LINES_ALPHA)

    LOG.debug(f"    {bpm}: color={color}, nstems={sum(idxs)}")


def _format_stem_axes(axs, limits):
    for ax, plane in zip(axs, PLANES):
        ax.set_yscale('log')
        ax.set_xlim(limits.xlim)
        ax.set_ylim(limits.ylim)
        ax.set_ylabel(LABEL_Y_SPECTRUM.format(plane=plane.upper()))
        ax.set_xlabel(LABEL_X)
        ax.tick_params(axis='both', which='major')


def _create_legend(axs, bpms, lines, hide_labels):
    legends = []
    lines_params = dict(
        marker=matplotlib.rcParams[u'lines.marker'],
        markersize=matplotlib.rcParams[u'font.size'] * 0.5,
        linestyle='None',
    )
    legend_params = dict(
        loc='lower right',
        ncol=NCOL_LEGEND,
        fancybox=False, shadow=False, frameon=False,
    )
    if isinstance(bpms, str):
        for idx, ax in enumerate(axs):
            handles = [mlines.Line2D([], [], color=get_cycled_color(idx), label=bpms, **lines_params)]
            leg = ax.legend(handles=handles, **legend_params)
            legends.append((leg, handles))

    else:
        if not hide_labels:
            handles = [mlines.Line2D([], [], color=get_cycled_color(idx), label=bpm, **lines_params)
                       for idx, bpm in enumerate(bpms)]
            leg = axs[0].legend(handles=handles, **legend_params)
            legends.append((leg, handles))

    for leg, handles in legends:
        leg.axes.figure.canvas.draw()

        # check if it is wider than the axes
        legend_width = leg.get_window_extent().inverse_transformed(leg.axes.transAxes).width
        x_shift = 0
        if legend_width > 1:
            x_shift = (legend_width - 1) / 2.  # shift more into center

        # move above line-labels
        nlines = sum([l is not None for l in lines.values()]) + 0.05
        y_shift = get_textsize_in_axes_coordinates(leg.axes, label_size=matplotlib.rcParams['axes.labelsize']) * nlines
        leg.axes.legend(handles=handles, bbox_to_anchor=(1. + x_shift, 1. + y_shift), **legend_params)


# Waterfall Plotting -----------------------------------------------------------


def _create_waterfall_plot(out, bpms, files, lines, limits, opts):
    LOG.debug(f"  ...creating Waterfall Plot")
    tunes = _get_tunes({plane: files[LIN][plane].loc[bpms[plane], :] for plane in PLANES})
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(18, 9))
    for ax, plane in zip(axs, PLANES):
        if len(bpms[plane]) == 0:
            continue

        freqs = files[FREQS][plane].loc[:, bpms[plane]]
        amps = files[AMPS][plane].loc[:, bpms[plane]]
        if opts.rescale:
            amps = _rescale_amp(amps)
        _plot_waterfall(ax, freqs, amps, opts)
        _plot_lines(ax, tunes, lines)
        _format_waterfall_axes(ax, freqs, plane, limits, opts.hide_bpm_labels)
    _output_plot(fig, out, WATERFALL_FILENAME)
    return fig


def _plot_waterfall(ax, freqs, amps, opts):
    nbpms, nfreqs = len(freqs.columns), len(freqs.index)
    norm = colors.LogNorm(amps.min().min(), amps.max().max())
    if opts.line_width == "auto":
        for idx, bpm in enumerate(freqs.columns):
            f_bpm = freqs[bpm].to_numpy().T
            freqs_mesh = np.tile(np.array([*f_bpm, .5]), [2, 1])
            y_mesh = np.tile([idx - 0.5, idx + 0.5], [nfreqs + 1, 1]).T
            ax.pcolormesh(freqs_mesh, y_mesh, amps[[bpm]].T, cmap=opts.cmap, norm=norm)
    else:
        for idx, bpm in enumerate(freqs.columns):
            lc = ax.vlines(x=freqs[bpm], ymin=idx - .5, ymax=idx + .5,
                           linestyles='solid', cmap=opts.cmap, norm=norm,
                           linewidths=opts.line_width,
                           )
            lc.set_array(amps[bpm])  # sets the colors of the segments
    ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=opts.cmap), ax=ax)


def _format_waterfall_axes(ax, freqs, plane, limits, hide_labels):
    if hide_labels:
        ax.set_yticklabels([])
        ax.set_yticks([])
    else:
        ax.set_yticklabels(freqs.columns, fontdict={'fontsize': matplotlib.rcParams[u'axes.labelsize'] * .5})
        ax.set_yticks(range(len(freqs.columns)))
    ax.set_xlabel(LABEL_X)
    ax.set_ylabel(LABEL_Y_WATERFALL.format(plane=plane.upper()))
    ax.set_xlim(limits.xlim)
    ax.set_ylim([-.5, len(freqs.columns) - .5])
    ax.tick_params(axis='x', which='both')


# Plot Lines -------------------------------------------------------------------


def _plot_lines(ax, tunes, lines, zorder=None):
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    label_size = matplotlib.rcParams['axes.labelsize'] * 0.7
    bottom_qlabel = 1.01
    bottom_natqlabel = bottom_qlabel + (get_textsize_in_axes_coordinates(ax, label_size) * 1.1)

    # Tune Lines ---
    for line_params in (("", lines.tune, "--", bottom_qlabel), ("NAT", lines.nattune, ":", bottom_natqlabel)):
        _plot_tune_line(ax, tunes, trans, zorder, label_size, *line_params)

    # Manual Lines ---
    for mline in lines.manual:
        _plot_manual_line(ax, mline, zorder, trans, label_size)


def _plot_tune_line(ax, tunes, transform, zorder, label_size, q_string, resonances, linestyle, label_y):
    if len(resonances) == 0:
        return

    pref = q_string[0] if len(q_string) else ""
    q = np.array([tunes[f'{q_string}Q{p.upper()}'] for p in PLANES])
    freqs = _get_resonance_frequencies(resonances, q)
    for res, freq in zip(resonances, freqs):
        if not np.isnan(freq):
            label, order = f'{pref}({res[0]}, {res[1]})', sum(res) + 1
            ax.axvline(x=freq, label=label,
                       linestyle=linestyle, color=f"C{order - 2}", marker='None',
                       zorder=zorder, alpha=RESONANCE_LINES_ALPHA)
            ax.text(x=freq, y=label_y, s=label, transform=transform, color=f"C{order - 2}", va='bottom', ha='center',
                    fontdict={'size': label_size})


def _plot_manual_line(ax, mline, zorder, transform, label_size):
    mline.setdefault('alpha', RESONANCE_LINES_ALPHA)
    mline.setdefault('marker', 'None')
    mline.setdefault('zorder', zorder)
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


def get_textsize_in_axes_coordinates(ax, label_size):
    transform = ax.transAxes.inverted().transform
    _, label_size_ax = transform((0, label_size)) - transform((0, 0))
    return label_size_ax


# Input ------------------------------------------------------------------------


def _check_opt(opt):
    if not (opt.waterfall_plot or opt.stem_plot):
        raise ValueError("Either spectrum or waterfall plots need to be activated!")

    if opt.stem_single_fig and not opt.stem_plot:
        LOG.warning("'stem_single_fig' option has no effect, when stem plots are deactivated!")

    if opt.waterfall_line_width and not opt.waterfall_plot:
        LOG.warning("'waterfall_line_width' option has no effect, when waterfall plots are deactivated!")

    if opt.waterfall_cmap and not opt.waterfall_plot:
        LOG.warning("'waterfall_cmap' option has no effect, when waterfall plots are deactivated!")

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

    # out structure
    out = opt.get_subdict(('show_plots', 'filetype'))
    out.dir = None  # set later if not None

    # stem-plot options
    stem = opt.get_subdict(('stem_plot', 'stem_single_fig', 'hide_bpm_labels', 'rescale'))
    stem = _rename_dict_keys(stem, to_remove="stem_")

    # waterfall-plot options
    waterfall = opt.get_subdict(('waterfall_plot', 'waterfall_line_width', 'waterfall_cmap', 'hide_bpm_labels', 'rescale'))
    waterfall = _rename_dict_keys(waterfall, to_remove="waterfall_")

    return out, limits, lines, stem, waterfall


def _load_spectrum_data(file_path, bpms):
    try:
        return {
            AMPS: _get_amplitude_files(file_path),
            FREQS: _get_frequency_files(file_path),
            LIN: _get_lin_files(file_path),
        }
    except FileNotFoundError:
        LOG.info("Some files not present. Trying to load old data format")
        return _get_old_data(file_path, bpms)


def _get_amplitude_files(file_path):
    return _get_planed_files(file_path, id=FILE_AMPS_EXT)


def _get_frequency_files(file_path):
    return _get_planed_files(file_path, id=FILE_FREQS_EXT)


def _get_lin_files(file_path):
    return _get_planed_files(file_path, id=FILE_LIN_EXT, index=COL_NAME)


def _get_planed_files(file_path, id, index=None):
    directory, filename = _get_dir_and_name(file_path)
    return {
        plane: tfs.read(os.path.join(directory, f'{filename}.{id.format(plane=plane.lower())}'), index=index)
        for plane in PLANES
    }


def _get_old_data(file_path, bpms):
    directory, filename = _get_dir_and_name(file_path)
    bpm_dir = os.path.join(directory, 'BPM')
    files = {LIN: {}, AMPS: {}, FREQS: {}}
    for plane in PLANES:
        files[LIN][plane] = tfs.read(os.path.join(directory, f'{filename}_lin{plane}'), index=COL_NAME)
        for id in (FREQS, AMPS):
            files[id][plane] = tfs.TfsDataFrame(columns=bpms)
        for bpm in bpms:
            with suppress(FileNotFoundError):
                df = tfs.read(os.path.join(bpm_dir, f'{bpm}.{plane}'))
                files[FREQS][plane][bpm] = df["FREQ"]
                files[AMPS][plane][bpm] = df["AMP"]
        for id in (FREQS, AMPS):
            files[id][plane] = files[id][plane].fillna(0)
    return files


def _get_tunes(lin, bpm=None):
    if bpm is None:
        def get_value(x):
            return x.mean()
    else:
        def get_value(x):
            return x[bpm]

    out = {f'{nat}Q{plane.upper()}': 0 for plane in PLANES for nat in ('', 'NAT')}
    for nat in ('', 'NAT'):
        for plane in PLANES:
            with suppress(KeyError):
                out[f'{nat}Q{plane.upper()}'] = get_value(lin[plane].loc[:, f'{nat}TUNE{plane.upper()}'])
    return out


def _get_dir_and_name(file_path):
    return os.path.dirname(file_path), os.path.basename(file_path)


def _get_bpms(lin_files, given_bpms):
    found_bpms = {}
    for plane in PLANES:
        found_bpms[plane] = list(lin_files[plane].index)
        if given_bpms is not None:
            found_bpms[plane] = [bpm for bpm in found_bpms[plane] if bpm in given_bpms]
            bpms_not_found = [bpm for bpm in given_bpms if bpm not in found_bpms[plane]]
            if len(bpms_not_found):
                LOG.warning(
                    f"The following BPMs are not present or not present in plane {plane}: {list2str(bpms_not_found)}"
                )
        if len(found_bpms[plane]) == 0:
            LOG.warning(f"No BPMs found for plane {plane} !")

    if not any([len(bpms) for bpms in found_bpms.values()]):
        raise IOError("No BPMs found in any plane!")
    return found_bpms


def _get_valid_indices(files, plane, bpm):
    """ Intersection of filtered AMPS and FREQS indices. """
    return index_filter(files[AMPS][plane][bpm]).intersection(index_filter(files[FREQS][plane][bpm]))


def index_filter(data):
    """ Only non-NaN and non-Zero data allowed. (Should not be zero due to _filter_amps() anyway.)"""
    return data[~(data.isna() | data == 0)].index


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


def _make_output_dir(out_dir, filename):
    out_dir = os.path.join(out_dir, os.path.splitext(filename)[0])
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _get_ini_filename():
    return CONFIG_FILENAME.format(time=datetime.utcnow().strftime(formats.TIME))


def _output_plot(fig, out, fname):
    fig.tight_layout()
    fig.tight_layout()  # sometimes better to do twice

    if out.show_plots:
        # fig.show()  # does not work, as it does not wait for user
        plt.show()  # waits for user to close plot

    if out.dir is not None:
        out_path = os.path.join(out.dir, f'{fname}.{out.filetype}')
        LOG.info(f"Saving Plot '{out_path}'")
        fig.savefig(out_path)

    plt.close(fig)  # should free RAM


# Helper -----------------------------------------------------------------------


def list2str(l):
    return str(l)[1:-1]


def _rescale_amp(amp_data):
    return amp_data.divide(amp_data.max(axis=0), axis=1)  # dataframe


def _get_all_bpms(bpms_dict):
    return set.union(*[set(v) for v in bpms_dict.values()])


def _rename_dict_keys(d, to_remove):
    for key in list(d.keys()):  # using list to copy keys
        d[key.replace(to_remove, "")] = d.pop(key)
    return d


if __name__ == "__main__":
    main()
