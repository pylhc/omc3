"""
Plot TFS
-------------------------

Wrapper to easily plot tfs-files. With entrypoint functionality.


"""
import os
from collections import Iterable
from contextlib import suppress
from pathlib import Path

import tfs
from generic_parser import EntryPointParameters, entrypoint
from generic_parser.entrypoint_parser import save_options_to_config
from matplotlib import pyplot as plt, rcParams

from omc3.definitions import formats
from omc3.plotting.utils import annotations, lines, style as pstyle
from omc3.optics_measurements.constants import EXT
from pylhc.constants.plot_tfs import IR_POS_DEFAULT, MANUAL_STYLE, ERROR_ALPHA, MAX_LEGENDLENGTH, COMPLEX_NAMES

from omc3.utils.logging_tools import get_logger, list2str
from omc3.plotting.optics_measurements.utils import FigureCollector, IdData
from omc3.plotting.spectrum.utils import get_unique_filenames

LOG = get_logger(__name__)


# Constants, Style and Arguments #############################################


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
        help="Column-Lables for the plots, default: y_column.",
        type=str,
        nargs="+",
    )
    params.add_parameter(
        name="file_labels",
        help="Names for the sources for the plots, default: filenames.",
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
        name="output",
        help="Base-Name of the output files. Some ID and file-suffix will be attached.",
        type=str,
    )
    params.add_parameter(
        name="plot_suffix",
        help="Suffix for the plots.",
        type=str,
        default=".pdf"
    )
    params.add_parameter(
        name="change_marker",
        help="Changes marker for each line in the plot.",
        action="store_true",
    )
    params.add_parameter(
        name="no_legend",
        help="Deactivates the legend.",
        action="store_true",
    )
    params.add_parameter(
        name="show",
        help="Shows plots.",
        action="store_true",
    )
    params.add_parameter(
        name="combine_by",
        help="Combine plots into one. The option 'planes' only works if planes 'XY' is chosen.",
        action="store_true",
        choices=['files', 'columns', 'planes']
    )
    params.add_parameter(
        name="xlim",
        nargs=2,
        type=float,
        help='Limits on the x axis (Tupel)')
    params.add_parameter(
        name="ylim",
        nargs=2,
        type=float,
        help='Limits on the y axis (Tupel)')
    params.add_parameter(
        name="vertical_lines",
        nargs="*",
        default=[],
        type=DictAsString,
        help='List of vertical lines (e.g. IR positions) to plot. '
             'Need to contain arguments for axvline, and may contain '
             f'the additional key "loc" which is one of {list(MANUAL_LOCATIONS.keys())} '
             'and places the label as text at the given location.')
    return params


# Main invocation ############################################################


@entrypoint(get_params(), strict=True)
def plot(opt):
    LOG.info(f"Starting plotting of tfs files: {list2str(opt.files):s}")
    if opt.output is not None:
        _save_options_to_config(opt)

    # preparations
    opt, sorting_opts = _check_opt(opt)

    # extract data
    fig_collection = sort_data(opt.get_subdict(
        ['planes',
         'x_columns', 'y_columns', 'error_coumns',
         'file_labels', 'column_labels',
         'combine_by', 'output', 'plot_suffix']))

    # plotting
    figs = create_plots(twiss_data, opt.x_cols, opt.y_cols, opt.e_cols, opt.file_labels, opt.column_labels,
                        opt.y_labels, opt.xy, opt.change_marker, opt.no_legend, opt.auto_scale, opt.figure_per_file)

    # exports
    if opt.output:
        export_plots(figs, opt.output)

    if not opt.no_show:
        plt.show()

    return figs


# Private Functions ------------------------------------------------------------


def sort_data(opt):
    """ Load all data from files and sort into figures"""
    collector = FigureCollector()
    combine_by_set = frozenset(opt.combine_by) - {'planes'}
    for (file_path, filename), file_label in zip(get_unique_filenames(opt.files), opt.file_labels):
        for x_col, y_col, err_col, column_label in zip(opt.x_columns, opt.y_columns, opt.error_columns, opt.column_labels):
            id_map = get_id(filename, y_col, file_label, column_label, combine_by_set)
            if opt.planes is None:

                collector.add_data_for_id(
                    id_=id_map['id'], label=id_map['label'],
                    data=_read_data(file_path, x_col, y_col, err_col),
                    path=opt.output, y_label=id_map['ylabel']
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
                        path=opt.output, y_label=y_label_plane,
                        n_planes=len(planes),
                        plane_idx=idx_plane,
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
    return dict(
        x=tfs_data[x_col],
        y=tfs_data[y_col],
        err=tfs_data[err_col] if err_col is not None else None,
    )



def get_ncols(self):
    """ Returns the number of columns for the legend.

    Done here, as this class divides single-plot from multiplot anyway
    """
    names = self.column_labels if self.figure_per_dataframe else self.dataframe_labels
    names = [n for n in names if n is not None]
    try:
        return annotations.get_legend_ncols(names, MAX_LEGENDLENGTH)
    except ValueError:
        return 3


def create_plots(dataframes, x_cols, y_cols, e_cols, dataframe_labels, column_labels, y_labels, xy, change_marker,
                 no_legend, auto_scale, figure_per_dataframe=False):
    # create layout
    pstyle.set_style("standard", MANUAL_STYLE)
    ir_positions, x_is_position = _get_ir_positions(dataframes, x_cols)

    y_lims = None
    the_loop = _LoopGenerator(x_cols, y_cols, e_cols, dataframes,
                              dataframe_labels, column_labels, y_labels,
                              xy, figure_per_dataframe)

    for ax, idx_plot, idx, data, x_col, y_col, e_col, legend, y_label, last_line in the_loop():
        # plot data
        y_label_from_col, y_plane, y_col, e_col, chromatic = _get_names_and_columns(idx_plot, xy,
                                                                                     y_col, e_col)

        x_val, y_val, e_val = _get_column_data(data, x_col, y_col, e_col)

        ebar = ax.errorbar(x_val, y_val, yerr=e_val,
                           ls=rcParams[u"lines.linestyle"], fmt=get_marker(idx, change_marker),
                           label=legend)

        _change_ebar_alpha(ebar)

        if auto_scale:
            current_y_lims = _get_auto_scale(y_val, auto_scale)
            if y_lims is None:
                y_lims = current_y_lims
            else:
                y_lims = [min(y_lims[0], current_y_lims[0]),
                          max(y_lims[1], current_y_lims[1])]
            if last_line:
                ax.set_ylim(*y_lims)

        # things to do only once
        if last_line:
            # setting the y_label
            if y_label is None:
                _set_ylabel(ax, y_col, y_label_from_col, y_plane, chromatic)
            else:
                y_label_from_label = ""
                if y_label:
                    y_label_from_label, y_plane, _, _, chromatic = _get_names_and_columns(
                        idx_plot, xy, y_label, "")
                if xy:
                    y_label = f"{y_label:s} {y_plane:s}"
                _set_ylabel(ax, y_label, y_label_from_label, y_plane, chromatic)

            # setting x limits
            if x_is_position:
                with suppress(AttributeError, ValueError):
                    post_processing.set_xlimits(data.SEQUENCE, ax)

            # setting visibility, ir-markers and label
            if xy and idx_plot == 0:
                ax.axes.get_xaxis().set_visible(False)
                if x_is_position and ir_positions:
                    annotations.show_ir(ir_positions, ax, mode='lines')
            else:
                if x_is_position:
                    annotations.set_xaxis_label(ax)
                    if ir_positions:
                        annotations.show_ir(ir_positions, ax, mode='outside')

            if not no_legend and idx_plot == 0:
                annotations.make_top_legend(ax, the_loop.get_ncols())

    return the_loop.get_figs()



# Output ---

def _save_options_to_config(opt):
    os.makedirs(Path(opt.output).parent, exist_ok=True)
    save_options_to_config(os.path.join(opt.output_dir, formats.get_config_filename(__file__)),
                           OrderedDict(sorted(opt.items()))
                           )

def export_plots(figs, output):
    """ Export all created figures to PDF """
    for param in figs:
        pdf_path = f"{output:s}_{param:s}.pdf"
        figs[param].savefig(pdf_path, bbox_inches='tight')
        LOG.debug(f"Exported tfs-contents to PDF '{pdf_path:s}'")


# Helper #####################################################################


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


def get_marker(idx, change):
    """ Return the marker used """
    if change:
        return lines.MarkerList.get_marker(idx)
    else:
        return rcParams['lines.marker']


def _get_column_data(data, x_col, y_col, e_col):
    """ Extract column data """
    x_val = data[x_col]
    y_val = data[y_col]
    try:
        e_val = data[e_col]
    except (KeyError, ValueError):
        e_val = None
    return x_val, y_val, e_val


def _change_ebar_alpha(ebar, alpha):
    """ loop through bars (ebar[1]) and caps (ebar[2]) and set the alpha value """
    for bars_or_caps in ebar[1:]:
        for bar_or_cap in bars_or_caps:
            bar_or_cap.set_alpha(alpha)



# Script Mode ################################################################


if __name__ == "__main__":
    plot()
