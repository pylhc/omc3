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
        help="Path to files to plot. If planes are used, omit file-ending.",
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
        name="y_labels",
        help="Labels for the y-axis, default: file_labels or column_labels (depending on combine_by).",
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
        name="planes",
        help=("Works only with filenames ending in '_x' and '_y' and "
              "columns ending in X or Y. These suffixes will be attached "
              "to the given names. If 'XY' is chosen here, both are present in "
              "the same figure. Either as top- and bottom- axes or, if "
              "'combine_by planes' is chosen, in one single axis."),
        type=str,
        nargs='+',
        choices=["X", "Y", "XY"],
    )
    params.add_parameter(
        name="output",  # TODO: use dir only, filename should be created from file_label column_label plane !!
        help="Folder to output the plots to.",
        type=str,
    )
    params.add_parameter(
        name="show",
        help="Shows plots.",
        action="store_true",
    )
    params.add_parameter(
        name="combine_by",
        help="Combine plots into one. The option 'planes' only works if planes 'XY' is chosen.",
        type=str,
        choices=['files', 'columns', 'planes']
    )
    params.add_parameter(
        name="xlim",
        nargs=2,
        type=float,
        help='Limits on the x axis (Tupel)'
    )
    params.add_parameter(
        name="ylim",
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
             f'the additional key "loc" which is one of {list(VERTICAL_LINES_TEXT_LOCATIONS.keys())} '
             'and places the label as text at the given location.')

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
    opt, sorting_opts = _check_opt(opt)
    pstyle.set_style(opt.plot_styles, opt.manual_style)

    # extract data
    fig_collection = sort_data(opt.get_subdict(
        ['planes', 'x_columns', 'y_columns', 'error_columns',
         'file_labels', 'column_labels', 'x_labels', 'y_labels',
         'combine_by', 'output']))

    # plotting
    _create_plots(fig_collection,
                  opt.get_subdict(['xlim', 'ylim',
                                   'ncols_legend', 'change_marker', 'errorbar_alpha',
                                   'vertical_lines']))

    return fig_collection.fig_dict


# Sorting ----------------------------------------------------------------------


def sort_data(opt):
    """ Load all data from files and sort into figures"""
    collector = FigureCollector()
    combine_by_set = frozenset(opt.combine_by) - {'planes'}
    for (file_path, filename), file_label in zip(get_unique_filenames(opt.files), opt.file_labels):
        for x_col, y_col, err_col, column_label in zip(opt.x_columns, opt.y_columns, opt.error_columns, opt.column_labels):
            id_map = get_id(filename, y_col, file_label, column_label, combine_by_set)
            output_path = f"{opt.output}_{id_map['id']}.{matplotlib.rcParams['savefig.format']}"

            if opt.planes is None:

                collector.add_data_for_id(
                    id_=id_map['id'], label=id_map['label'],
                    data=_read_data(file_path, x_col, y_col, err_col),
                    path=output_path, y_label=id_map['ylabel']
                    )
            else:

                planes = opt.planes
                if 'XY' in planes:
                    planes = ['X', 'Y']

                for idx_plane, plane in enumerate(planes):
                    file_path_plane = file_path.with_name(f'{file_path.name}{plane.lower()}')
                    y_col_plane = f'{y_col}{plane.upper()}'
                    y_label_plane = id_map['ylabel'].format(plane)
                    collector.add_data_for_id(
                        id_=id_map['id'], label=id_map['label'],
                        data=_read_data(file_path_plane, x_col, y_col_plane, err_col),
                        path=output_path, y_label=y_label_plane,
                        n_axes=len(planes),
                        axis_idx=idx_plane,
                        combine_planes='planes' in opt.combine_by,
                    )
    return collector


def get_id(filename, column, file_label, column_label, combine_by):
    if filename is None:
        file_label = filename

    if column_label is None:
        column_label = column

    map = {
        frozenset(['files', 'columns']): dict(
            id=f'',
            label=f'{file_label} {column}',
            ylabel=f'{column_label}'
        ),
        frozenset(['files']): dict(
            id=f'{column}',
            label=f'{file_label}',
            ylabel=f'{column_label}'
        ),
        frozenset(['columns']): dict(
            id=f'{filename}',
            label=f'{column_label}',
            ylabel=f'{file_label}'
        ),
        frozenset([]): dict(
            id=f'{filename}_{column}',
            label=f'',
            ylabel=f'{column_label}'
        ),

    }
    return map[combine_by]


def _read_data(file_path, x_col, y_col, err_col):
    tfs_data = tfs.read(str(file_path))
    return DotDict(
        x=tfs_data[x_col],
        y=tfs_data[y_col],
        err=tfs_data[err_col] if err_col is not None else None,
    )


# Plotting ---------------------------------------------------------------------


def _create_plots(fig_collection, opt):
    for fig_container in fig_collection.figs:
        for idx_ax, (ax, data, ylabel) in enumerate(zip(fig_container.axes, fig_container.data, fig_container.ylabels)):
            _plot_vlines(ax, opt.vertical_lines)
            _plot_data(ax, data, opt.change_marker, opt.errorbar_alpha)
            _set_axes_layout(ax, opt.xlim, opt.ylim, ylabel)

            if idx_ax == 0:
                pannot.make_top_legend(ax, opt.ncols_legend)

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
    for line in lines:
        loc = line.pop('loc', None)
        plines.plot_vertical_line(ax, line, label_loc=loc)
        line['loc'] = loc  # put back for other axes


def _set_axes_layout(ax, xlim, ylim, ylabel):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)


# Output ---


def _save_options_to_config(opt):
    output_dir = Path(opt.output).parent
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
        opt.x_columns = opt.x_columns * len(opt.y_columns)
    elif len(opt.x_columns) != len(opt.y_columns):
        raise AttributeError("The number of x-columns and y-columns differ!")

    if opt.error_columns is None:
        opt.error_columns = [None] * len(opt.y_columns)
    elif len(opt.error_columns) != len(opt.y_columns):
        raise AttributeError("The number of error columns and number of y columns differ!")

    if opt.column_labels is None:
        opt.column_labels = [None] * len(opt.y_columns)
    elif len(opt.column_labels) != len(opt.y_columns):
        raise AttributeError("The number of column-labels and number of y columns differ!")

    if 'XY' in opt.planes and len(opt.planes) > 1:
        raise AttributeError("Planes 'XY' can not be chosen in combination with other planes.")

    if 'planes' in opt.combine_by and 'XY' not in opt.planes:
        raise AttributeError("Combine by 'planes' can only be used in combination with planes 'XY'.")

    return opt


def _get_marker(idx, change):
    """ Return the marker used """
    if change:
        return plines.MarkerList.get_marker(idx)
    else:
        return rcParams['lines.marker']


# TODO
    if auto_scale:
        current_y_lims = _get_auto_scale(y_val, auto_scale)
        if y_lims is None:
            y_lims = current_y_lims
        else:
            y_lims = [min(y_lims[0], current_y_lims[0]),
                      max(y_lims[1], current_y_lims[1])]
        if last_line:
            ax.set_ylim(*y_lims)

# Script Mode ------------------------------------------------------------------


if __name__ == "__main__":
    plot()
