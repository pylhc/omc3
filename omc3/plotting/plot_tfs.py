"""
Plot TFS
-------------------------

Easily plot tfs-files with all kinds of additional functionality and ways to
combine plots.


"""
import os
from collections import OrderedDict
from pathlib import Path

import matplotlib
import tfs
from generic_parser import EntryPointParameters, entrypoint, DotDict
from generic_parser.entry_datatypes import DictAsString
from generic_parser.entrypoint_parser import save_options_to_config
from matplotlib import pyplot as plt, rcParams

from omc3.definitions import formats
from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import EXT
from omc3.plotting.optics_measurements.constants import DEFAULTS
from omc3.plotting.optics_measurements.utils import FigureCollector
from omc3.plotting.spectrum.utils import get_unique_filenames, output_plot
from omc3.plotting.utils import (annotations as pannot, lines as plines,
                                 style as pstyle, colors as pcolors)
from omc3.plotting.utils.lines import VERTICAL_LINES_TEXT_LOCATIONS
from omc3.utils.logging_tools import get_logger, list2str

LOG = get_logger(__name__)


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="files",
        help=("Path to files to plot. "
              f"If planes are used, omit file-ending. In this case '{EXT}' is"
              "assumed"),
        required=True,
        nargs="+",
        type=str,
    )
    params.add_parameter(
        name="y_columns",
        help="List of column names to plot (e.g. BETX, BETY or just BET if `planes` is used.)",
        required=True,
        type=str,
        nargs="+",
    )
    params.add_parameter(
        name="x_columns",
        help="List of column names to use as x-values.",
        required=True,
        type=str,
        nargs="+",
    )
    params.add_parameter(
        name="error_columns",
        help="List of parameters to get error values from.",
        type=str,
        nargs="+",
    )
    params.add_parameter(
        name="column_labels",
        help="Column-Labels for the plots, default: y_columns.",
        type=str,
        nargs="+",
    )
    params.add_parameter(
        name="file_labels",
        help="Labels for the files, default: filenames.",
        type=str,
        nargs="+",
    )
    params.add_parameter(
        name="x_labels",
        help="Labels for the x-axis, default: x_columns.",
        type=str,
        nargs="+",
    )
    params.add_parameter(
        name="y_labels",
        help=("Override labels for the y-axis, default: file_labels or column_labels "
              "(depending on same_axes). Needs to be a list of lists, where the "
              "inner list goes over the axes in one figure and the outer over "
              "the figures. If the respective length is 1, the same label will "
              "be used for all figures or axes."
              ),
        nargs='+',
    )
    params.add_parameter(
        name="planes",
        help=("Works only with filenames ending in 'x' and 'y' and "
              "columns ending in X or Y. These suffixes will be attached "
              "to the given files and y_columns."),
        type=str,
        nargs='+',
        choices=PLANES,
    )
    params.add_parameter(
        name="output",
        help="Folder to output the plots to.",
        type=str,
    )
    params.add_parameter(
        name="output_prefix",
        help="Prefix for the output filename.",
        type=str,
        default="plot_"
    )
    params.add_parameter(
        name="show",
        help="Shows plots.",
        action="store_true",
    )
    params.add_parameter(
        name="same_axes",
        help="Combine plots into single axes. Multiple choices possible.",
        type=str,
        nargs='+',
        choices=['files', 'columns', 'planes']
    )
    params.add_parameter(
        name="same_figure",
        help=("Plot two axes into the same figure "
              "(can't be the same as 'same_axes'). "
              "Has no effect if there is only one of the given thing."),
        type=str,
        choices=['files', 'columns', 'planes']
    )
    params.add_parameter(
        name="single_legend",
        help="Show only one legend instance (at the top plot).",
        action="store_true",
    )
    params.add_parameter(
        name="x_lim",
        nargs=2,
        type=float,
        help='Limits on the x axis (Tupel)'
    )
    params.add_parameter(
        name="y_lim",
        nargs=2,
        type=float,
        help='Limits on the y axis (Tupel)'
    )
    params.add_parameter(
        name="vertical_lines",
        nargs="*",
        default=[],
        type=DictAsString,
        help='List of vertical lines (e.g. IR positions) to plot. '
             'Need to contain arguments for axvline, and may contain '
             'the additional keys "text" and "loc" which is one of '
             f' {list(VERTICAL_LINES_TEXT_LOCATIONS.keys())} and places the text at the given location.')

    # Plotting Style Parameters ---
    params.add_parameter(
        name="plot_styles",
        type=str,
        nargs="+",
        default=['standard'],
        help='Which plotting styles to use, either from plotting.utils.*.mplstyles or default mpl.'
    )
    params.add_parameter(
        name="manual_style",
        type=DictAsString,
        default={},
        help='Additional style rcParameters which update the set of predefined ones.'
    )
    params.add_parameter(
        name="change_marker",
        help="Changes marker for each line in the plot.",
        action="store_true",
    )
    params.add_parameter(
        name="ncol_legend",
        type=int,
        default=DEFAULTS['ncol_legend'],
        help='Number of bpm legend-columns. If < 1 no legend is shown.'
    )
    params.add_parameter(
        name="errorbar_alpha",
        help="Alpha value for error bars",
        type=float,
        default=DEFAULTS['errorbar_alpha'],
    )
    return params


# Main invocation ############################################################


@entrypoint(get_params(), strict=True)
def plot(opt):
    LOG.info(f"Starting plotting of tfs files: {list2str(opt.files):s}")
    if opt.output is not None:
        _save_options_to_config(opt)

    # preparations
    opt = _check_opt(opt)
    pstyle.set_style(opt.plot_styles, opt.manual_style)

    # extract data
    fig_collection = sort_data(opt.get_subdict(
        ['files', 'planes', 'x_columns', 'y_columns', 'error_columns',
         'file_labels', 'column_labels', 'x_labels', 'y_labels',
         'same_axes', 'same_figure', 'output', 'output_prefix']))

    # plotting
    _create_plots(fig_collection,
                  opt.get_subdict(['x_lim', 'y_lim',
                                   'ncol_legend', 'single_legend',
                                   'change_marker', 'errorbar_alpha',
                                   'vertical_lines',
                                   'show']))

    return fig_collection.fig_dict


# Sorting ----------------------------------------------------------------------


def sort_data(opt):
    """ Load all data from files and sort into figures"""
    collector = FigureCollector()
    same_axes_set = frozenset()
    if opt.same_axes:
        same_axes_set = frozenset(opt.same_axes)
    for (file_path, filename), file_label in zip(get_unique_filenames(opt.files), opt.file_labels):
        for x_col, y_col, err_col, x_label, column_label in zip(
                opt.x_columns, opt.y_columns, opt.error_columns, opt.x_labels, opt.column_labels):

            axes_ids = _get_axes_ids(opt)

            if opt.planes is None:
                id_map = get_id(filename, y_col, file_label, column_label, same_axes_set,
                                opt.same_figure, opt.output_prefix)
                output_path = Path(opt.output) / f"{id_map['figure_id']}.{matplotlib.rcParams['savefig.format']}"

                collector.add_data_for_id(
                    figure_id=id_map['figure_id'],
                    label=id_map['legend_label'],
                    data=_read_data(file_path, x_col, y_col, err_col),
                    path=output_path,
                    x_label=x_label,
                    y_label=id_map['ylabel'],
                    axes_id=id_map['axes_id'],
                    axes_ids=axes_ids,
                )

            else:
                for plane in opt.planes:
                    id_map = get_id(filename, y_col, file_label, column_label, same_axes_set,
                                    opt.same_figure, opt.output_prefix, plane, opt.planes)
                    output_path = Path(opt.output) / f"{id_map['figure_id']}.{matplotlib.rcParams['savefig.format']}"

                    file_path_plane = file_path.with_name(f'{file_path.name}{plane.lower()}{EXT}')
                    y_col_plane = f'{y_col}{plane.upper()}'
                    err_col_plane = f'{err_col}{plane.upper()}'

                    collector.add_data_for_id(
                        figure_id=id_map['figure_id'],
                        label=id_map['legend_label'],
                        data=_read_data(file_path_plane, x_col, y_col_plane, err_col_plane),
                        path=output_path,
                        x_label=x_label,
                        y_label=id_map['ylabel'],
                        axes_id=id_map['axes_id'],
                        axes_ids=axes_ids,
                    )
    if opt.y_labels:
        _update_y_labels(collector, opt.y_labels)
    return collector


def get_id(filename, column, file_label, column_label, same_axes, same_figure, prefix, plane='', planes=[]):
    """ Get the right IDs for the current sorting way.

    This is where the actual sorting happens, by mapping the right IDs according
    to the chosen options.
    """
    file_last = filename[-1].replace(EXT, "").strip("_")
    file_output = "_".join(filename).replace(EXT, "").strip("_")
    if file_label is not None:
        file_output = f'{file_label}_{file_last}'
    else:
        file_label = file_output

    if column_label is None:
        column_label = f'{column}{plane}'
    else:
        column_label = _safe_format(column_label, plane)

    planes = "".join(planes)

    axes_id = {'files': f'{filename}',
               'columns': f'{column}',
               'planes': f'{plane}'
               }.get(same_figure, '')

    key = same_axes if same_figure is None else same_axes.union({same_figure})

    return {
        frozenset(['files', 'columns', 'planes']): dict(
            figure_id=f'{prefix}',
            axes_id=axes_id,
            legend_label=f'{file_label} {column_label}',
            ylabel=f'',
        ),
        frozenset(['files', 'columns']): dict(
            figure_id=f'{prefix}{plane}',
            axes_id=axes_id,
            legend_label=f'{file_label} {column_label}',
            ylabel=f'',
        ),
        frozenset(['planes', 'columns']): dict(
            figure_id=f'{prefix}{file_output}{planes}',
            axes_id=axes_id,
            legend_label=column_label,
            ylabel=column_label,
        ),
        frozenset(['planes', 'files']): dict(
            figure_id=f'{prefix}{column}{planes}',
            axes_id=axes_id,
            legend_label=f'{file_label} {column_label}',
            ylabel=column_label,
        ),
        frozenset(['planes']): dict(
            figure_id=f'{prefix}{file_output}_{column}{planes}',
            axes_id=axes_id,
            legend_label=column_label,
            ylabel=column_label,
        ),
        frozenset(['files']): dict(
            figure_id=f'{prefix}{column}{plane}',
            axes_id=axes_id,
            legend_label=f'{file_label}',
            ylabel=column_label,
        ),
        frozenset(['columns']): dict(
            figure_id=f'{prefix}{file_output}{plane}',
            axes_id=axes_id,
            legend_label=column_label,
            ylabel=f'{file_label}',
        ),
        frozenset([]): dict(
            figure_id=f'{prefix}{file_output}_{column}{plane}',
            axes_id=axes_id,
            legend_label=f'',
            ylabel=column_label,
        ),
    }[key]


def _read_data(file_path, x_col, y_col, err_col):
    tfs_data = tfs.read(str(file_path))
    return DotDict(
        x=tfs_data[x_col],
        y=tfs_data[y_col],
        err=tfs_data[err_col] if err_col is not None else None,
    )


def _update_y_labels(fig_collection, y_labels):
    if len(y_labels) == 1:
        y_labels = y_labels * len(fig_collection.figs)

    for idx_fig, fig_container in enumerate(fig_collection.figs.values()):
        axes_labels = y_labels[idx_fig]
        if len(axes_labels) == 1:
            axes_labels = axes_labels * len(fig_container.ylabels)

        for idx_ax, (ax_id, _) in enumerate(fig_container.ylabels.items()):
            fig_container.ylabels[ax_id] = _safe_format(axes_labels[idx_ax], ax_id)


# Plotting ---------------------------------------------------------------------


def _create_plots(fig_collection, opt):
    """ Main plotting routine """
    for fig_container in fig_collection.figs.values():
        for idx_ax, (ax, data, ylabel, xlabel) in enumerate(
                zip(fig_container.axes.values(), fig_container.data.values(),
                    fig_container.ylabels.values(), fig_container.xlabels.values())):
            _plot_vlines(ax, opt.vertical_lines)
            _plot_data(ax, data, opt.change_marker, opt.errorbar_alpha)
            _set_axes_layout(ax, opt.x_lim, opt.y_lim, ylabel, xlabel)

            if idx_ax == 0 or not opt.single_legend:
                pannot.make_top_legend(ax, opt.ncol_legend)

        output_plot(fig_container)

    if opt.show:
        plt.show()


def _plot_data(ax, data, change_marker, ebar_alpha):
    for idx, (label, values) in enumerate(data.items()):
        ebar = ax.errorbar(values.x, values.y, yerr=values.err,
                           ls=rcParams[u"lines.linestyle"],
                           fmt=_get_marker(idx, change_marker),
                           label=label)

        pcolors.change_ebar_alpha_for_line(ebar, ebar_alpha)


def _plot_vlines(ax, lines):
    if not lines:
        return

    for line in lines:
        loc = line.pop('loc', None)
        text = line.pop('text', None)
        plines.plot_vertical_line(ax, line, text=text, text_loc=loc)
        line['loc'] = loc  # put back for other axes
        line['text'] = text


def _set_axes_layout(ax, x_lim, y_lim, ylabel, xlabel):
    ax.set_xlim(x_lim)
    ax.set_xlabel(xlabel)
    ax.set_ylim(y_lim)
    ax.set_ylabel(ylabel)


# Output ---


def _save_options_to_config(opt):
    output_dir = Path(opt.output)
    os.makedirs(output_dir, exist_ok=True)
    save_options_to_config(output_dir / formats.get_config_filename(__file__),
                           OrderedDict(sorted(opt.items()))
                           )


def _export_plots(figs, output):
    """ Export all created figures to PDF """
    for param in figs:
        pdf_path = f"{output:s}_{param:s}.pdf"
        figs[param].savefig(pdf_path, bbox_inches='tight')
        LOG.debug(f"Exported tfs-contents to PDF '{pdf_path:s}'")


# Helper -----------------------------------------------------------------------


def _check_opt(opt):
    """ Sanity checks for the opt structure """
    if opt.file_labels is None:
        opt.file_labels = [None] * len(opt.files)
    elif len(opt.file_labels) != len(opt.files):
        raise AttributeError("The number of file-labels and number of files differ!")

    if len(opt.x_columns) == 1 and len(opt.y_columns) > 1:
        opt.x_columns *= len(opt.y_columns)
    elif len(opt.x_columns) != len(opt.y_columns):
        raise AttributeError("The number of x-columns and y-columns differ!")

    if len(opt.x_labels) == 1 and len(opt.x_columns) > 1:
        opt.x_labels *= len(opt.x_columns)
    elif len(opt.x_labels) != len(opt.x_columns):
        raise AttributeError("The number of x-labels and x-columns differ!")

    if opt.error_columns is None:
        opt.error_columns = [None] * len(opt.y_columns)
    elif len(opt.error_columns) != len(opt.y_columns):
        raise AttributeError("The number of error-columns and number of y-columns differ!")

    if opt.column_labels is None:
        opt.column_labels = [None] * len(opt.y_columns)
    elif len(opt.column_labels) == 1 and len(opt.y_columns) > 1:
        opt.column_labels *= len(opt.y_columns)
    elif len(opt.column_labels) != len(opt.y_columns):
        raise AttributeError("The number of column-labels and number of y columns differ!")

    if opt.same_figure is not None:
        if opt.same_axes is not None and opt.same_figure in opt.same_axes:
            raise AttributeError("Found the same option in 'same_axes' "
                                 "and 'same_figure'. This is not allowed.")

    return opt


def _get_axes_ids(opt):
    if opt.same_figure == "columns":
        axes_ids = opt.get("y_columns")
    else:
        axes_ids = opt.get(opt.same_figure)

    if axes_ids is None:  # do not put into 'get' as could be None at multiple levels
        axes_ids = ('',)
    return axes_ids


def _get_marker(idx, change):
    """ Return the marker used """
    if change:
        return plines.MarkerList.get_marker(idx)
    else:
        return rcParams['lines.marker']


def _safe_format(label, insert):
    try:
        return label.format(insert)
    except KeyError:  # can happen for latex strings
        return label


# Script Mode ------------------------------------------------------------------


if __name__ == "__main__":
    plot()
