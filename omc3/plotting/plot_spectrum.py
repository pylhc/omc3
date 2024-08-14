"""
Plot Spectrum
-------------

Spectrum plotter for frequency analysis output-data (supports also DRIVE output).

The spectra can be either plotted as `stem`-plots or as `waterfall`-plots.
The stem-plots can be in any combination: split by given files, split by given bpms or combined
in any way (by usage of the `combine_by` option).
Note that if both of those are false (as is default) there will anyway be only one waterfall plot
per given input file.


In case of split-by-file, plots are saved in a sub-directory of the given `output_dir` with the
name of the original TbT file.
In case of split by bpm the plots will have the bpm-name in their filename.


The `lines_tunes` and `lines_nattunes` lists accept tuples of multipliers for the respective
tunes, which define the resonance lines plotted into the spectrum as well. A dashed line will
indicate the average of all tunes given in the data of one figure, while a semi-transparent area
will indicate min- and max- values of this line.

With `lines_manual`, one can plot vertical lines at manual locations (see
parameter specs below).

The function returns two dictionaries, where the first dictionary contains the
stem plots and the second one the waterfall plots. They are identifyable
by unique id's which depend on which combination of merging the spectra into
one figure is used.


**Arguments:**

*--Required--*

- **files**:

    List of paths to the spectrum files. The files need to be given
    without their '.lin'/'.amps[xy]','.freqs[xy]' endings.  (So usually
    the path of the TbT-Data file.)


*--Optional--*

- **amp_limit** *(float)*:

    All amplitudes <= limit are filtered. This value needs to be at least
    0 to filter non-found frequencies.

    default: ``0.0``


- **bpms**:

    List of BPMs for which spectra will be plotted. If not given all BPMs
    are used.


- **combine_by**:

    Choose how to combine the data into figures.

    choices: ``['bpms', 'files']``

    default: ``[]``


- **filetype** *(str)*:

    Filetype to save plots as (i.e. extension without ".")

    default: ``pdf``


- **lines_manual** *(DictAsString)*:

    List of manual lines to plot. Need to contain arguments for axvline,
    and may contain the additional keys "text" and "loc" which is one of
    ['bottom', 'top', 'line bottom', 'line top'] and places the text at
    the given location.

    default: ``[]``


- **lines_nattunes** *(tuple)*:

    List of natural tune lines to plot

    default: ``[(1, 0), (0, 1)]``


- **lines_tunes** *(tuple)*:

    list of tune lines to plot

    default: ``[(1, 0), (0, 1)]``


- **manual_style** *(DictAsString)*:

    Additional style rcParameters which update the set of predefined ones.

    default: ``{}``


- **ncol_legend** *(int)*:

    Number of bpm legend-columns. If < 1 no legend is shown.

    default: ``5``


- **output_dir** *(str)*:

    Directory to write results to. If no option is given, plots will not
    be saved.


- **plot_styles** *(UnionPathStr)*:

    Which plotting styles to use, either from plotting.styles.*.mplstyles
    or default mpl.

    default: ``['standard', 'spectrum']``


- **plot_type**:

    Choose plot type (Multiple choices possible).

    choices: ``['stem', 'waterfall']``

    default: ``['stem']``


- **rescale**:

    Flag to rescale plots amplitude to max-line = 1

    action: ``store_true``


- **show_plots**:

    Flag to show plots

    action: ``store_true``


- **waterfall_cmap** *(str)*:

    Colormap to use for waterfall plot.

    default: ``inferno``


- **waterfall_common_plane_colors**:

    Same colorbar scale for both planes in waterfall plots.

    action: ``store_true``


- **waterfall_line_width**:

    Line width of the waterfall frequency lines. "auto" fills them up
    until the next one.

    default: ``2``


- **xlim** *(float)*:

    Limits on the x axis (Tupel)

    default: ``[0, 0.5]``


- **ylim** *(float)*:

    Limits on the y axis (Tupel)

    default: ``[1e-09, 1.0]``


"""
import os
from collections import OrderedDict
from typing import Tuple

import matplotlib
from cycler import cycler

from generic_parser.entry_datatypes import DictAsString
from generic_parser.entrypoint_parser import (entrypoint, EntryPointParameters,
                                              save_options_to_config, DotDict)
from omc3.definitions import formats
from omc3.plotting.spectrum.stem import create_stem_plots
from omc3.plotting.spectrum.utils import (NCOL_LEGEND, LIN,
                                          FigureCollector, get_unique_filenames,
                                          filter_amps, get_bpms, get_stem_id,
                                          get_waterfall_id, get_data_for_bpm,
                                          load_spectrum_data)
from omc3.plotting.spectrum.waterfall import create_waterfall_plots
from omc3.plotting.utils import style as pstyle
from omc3.plotting.utils.lines import VERTICAL_LINES_TEXT_LOCATIONS
from omc3.utils import logging_tools
from omc3.utils.iotools import UnionPathStr

LOG = logging_tools.getLogger(__name__)


def get_reshuffled_tab20c():
    """
    Reshuffle tab20c so that the colors change between next lines.
    Needs to be up here as it is used in ``DEFAULTS`` which is loaded early.
    """
    tab20c = matplotlib.colormaps['tab20c'].colors
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
        u'axes.prop_cycle': get_reshuffled_tab20c(),
    }
)


def get_params():
    params = EntryPointParameters()
    params.add_parameter(name="files",
                         required=True,
                         nargs='+',
                         help=("List of paths to the spectrum files. The files need to be given"
                               " without their '.lin'/'.amps[xy]','.freqs[xy]' endings. "
                               " (So usually the path of the TbT-Data file.)"))
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
    params.add_parameter(name="combine_by",
                         nargs="*",
                         choices=['bpms', 'files'],
                         default=[],
                         help='Choose how to combine the data into figures.')
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
    params.add_parameter(name="lines_tunes",
                         nargs="*",
                         type=tuple,
                         default=[(1, 0), (0, 1)],
                         help='list of tune lines to plot')
    params.add_parameter(name="lines_nattunes",
                         nargs="*",
                         type=tuple,
                         default=[(1, 0), (0, 1)],
                         help='List of natural tune lines to plot')
    params.add_parameter(name="lines_manual",
                         nargs="*",
                         default=[],
                         type=DictAsString,
                         help='List of manual lines to plot. Need to contain arguments for axvline, and may contain '
                              'the additional keys "text" and "loc" which is one of '
                              f'{list(VERTICAL_LINES_TEXT_LOCATIONS.keys())} and places the text at the given location.'
                         )
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
    params.add_parameter(name="plot_styles",
                         type=UnionPathStr,
                         nargs="+",
                         default=['standard', 'spectrum'],
                         help='Which plotting styles to use, either from plotting.styles.*.mplstyles or default mpl.'
                         )
    params.add_parameter(name="manual_style",
                         type=DictAsString,
                         default={},
                         help='Additional style rcParameters which update the set of predefined ones.'
                         )
 
    return params


# Main -------------------------------------------------------------------------


@entrypoint(get_params(), strict=True)
def main(opt):
    LOG.info("Starting spectrum plots.")
    if opt.output_dir is not None:
        _save_options_to_config(opt)

    opt = _check_opt(opt)
    pstyle.set_style(opt.plot_styles, opt.manual_style)
    stem_opt, waterfall_opt, sorting_opt = _sort_opt(opt)
    stem, waterfall = _sort_input_data(sorting_opt)

    if stem_opt.plot:
        create_stem_plots(stem.figs, stem_opt)

    if waterfall_opt.plot:
        create_waterfall_plots(waterfall.figs, waterfall_opt)

    return stem.fig_dict, waterfall.fig_dict


# Input ------------------------------------------------------------------------


def _check_opt(opt):
    if (opt.waterfall_line_width is not None and opt.waterfall_line_width != DEFAULTS.waterfall_line_width
            and 'waterfall' not in opt.plot_type):
        LOG.warning("Setting 'waterfall_line_width' option has no effect, "
                    "when waterfall plots are deactivated!")

    if (opt.waterfall_cmap is not None and opt.waterfall_cmap != DEFAULTS.waterfall_cmap
            and 'waterfall' not in opt.plot_type):
        LOG.warning("Setting 'waterfall_cmap' option has no effect, "
                    "when waterfall plots are deactivated!")

    if opt.amp_limit < 0:
        raise ValueError("The amplitude limit needs to be at least '0' "
                         "to filter for non-found frequencies.")

    style_dict = DEFAULTS.manual_style.copy()
    if opt.manual_style is not None:
        style_dict.update(opt.manual_style)
    opt.manual_style = style_dict

    return opt


def _sort_opt(opt):
    # lines structure
    lines = opt.get_subdict(('lines_tunes', 'lines_nattunes', 'lines_manual'))
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
    waterfall = opt.get_subdict(('waterfall_line_width', 'waterfall_cmap',
                                 'waterfall_common_plane_colors', 'ncol_legend'))
    waterfall = _rename_dict_keys(waterfall, to_remove="waterfall_")
    waterfall['plot'] = 'waterfall' in opt.plot_type

    # needed in both
    for d in (stem, waterfall):
        d['show'] = opt['show_plots']
        d['limits'] = limits
        d['lines'] = lines

    # sorting options
    sort = opt.get_subdict(('filetype', 'files',
                            'bpms', 'output_dir',
                            'amp_limit', 'rescale'))
    sort.plot_stem = stem.plot
    sort.plot_waterfall = waterfall.plot
    sort.combine_by = frozenset(opt.combine_by)

    return stem, waterfall, sort


# Output ---

def _save_options_to_config(opt):
    os.makedirs(opt.output_dir, exist_ok=True)
    save_options_to_config(os.path.join(opt.output_dir, formats.get_config_filename(__file__)),
                           OrderedDict(sorted(opt.items()))
                           )


# Load Data --------------------------------------------------------------------


def _sort_input_data(opt: DotDict) -> Tuple[FigureCollector, FigureCollector]:
    """Load and sort input data by file and bpm and assign correct figure-containers."""
    LOG.debug("Sorting input data.")

    stem_figs = FigureCollector()
    waterfall_figs = FigureCollector()

    # Data Sorting
    for file_path, filename in get_unique_filenames(opt.files):
        filename = "_".join(filename)
        LOG.info(f"Loading data for file '{filename}'.")

        data = load_spectrum_data(file_path, opt.bpms)
        data = filter_amps(data, opt.amp_limit)
        bpms = _get_all_bpms(get_bpms(data[LIN], opt.bpms, filename))

        for collector, get_id_fun, active in ((stem_figs, get_stem_id, opt.plot_stem),
                                              (waterfall_figs, get_waterfall_id, opt.plot_waterfall)):
            if not active:
                continue

            for bpm in bpms:
                the_id = get_id_fun(filename, bpm,
                                    opt.output_dir, opt.combine_by, opt.filetype)
                collector.add_data_for_id(the_id, get_data_for_bpm(data, bpm, opt.rescale))
    return stem_figs, waterfall_figs


# Helper -----------------------------------------------------------------------


def _get_all_bpms(bpms_dict):
    """Returns a union of all bpms for both planes."""
    return set.union(*[set(v) for v in bpms_dict.values()])


def _rename_dict_keys(d, to_remove):
    for key in list(d.keys()):  # using list to copy keys
        d[key.replace(to_remove, "")] = d.pop(key)
    return d


# Script Mode ------------------------------------------------------------------


if __name__ == "__main__":
    main()
