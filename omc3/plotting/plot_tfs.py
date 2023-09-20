"""
Plot TFS
--------

Easily plot tfs-files with all kinds of additional functionality and ways to combine plots.

.. code-block:: python

    from omc3.plotting.plot_tfs import plot

    figs = plot(
        files=['beta_phase_{0}.tfs', 'beta_amp_{0}.tfs'],
        same_figure='planes',
        same_axes=['files'],
        x_columns=['S'],
        y_columns=['BET{0}'],
        error_columns=None,
        planes=['X', 'Y'],
        x_labels=['Location [m]'],
        file_labels=[r'$\\beta$ from phase', r'$\beta$ from amplitude'],
        # column_labels=[r'$\\beta_{0}$'],    # would have bx in legend
        column_labels=[''],                  # removes COLUMNX COLUMNY from legend-names
        y_labels=[[r'$\\beta_x$', r'$\\beta_y$']],  # axes labels (outer = figures, inner = axes)
        output='output_dir',
        show=False,
        single_legend=True,
        change_marker=True,
    )


**Arguments:**

*--Required--*

- **files** *(MultiClass)*: Path to files to plot.
  If planes are used, replace the plane in the filename with '{0}'

- **x_columns** *(str)*: List of column names to use as x-values.

- **y_columns** *(str)*: List of column names to plot (e.g. BETX, BETY or BET{0} if `planes` is used.)


*--Optional--*

- **change_marker**: Changes marker for each line in the plot.

  Action: ``store_true``
- **column_labels** *(str)*: Column-Labels for the plots, default: y_columns.

- **error_columns** *(str)*: List of parameters to get error values from.

- **errorbar_alpha** *(float)*: Alpha value for error bars

  Default: ``0.6``
- **file_labels** *(str)*: Labels for the files, default: filenames.

- **manual_style** *(DictAsString)*: Additional style rcParameters which update the set of predefined ones.

  Default: ``{}``
- **ncol_legend** *(int)*: Number of bpm legend-columns. If < 1 no legend is shown.

  Default: ``3``
- **output** *(MultiClass)*: Folder to output the plots to.

- **output_prefix** *(str)*: Prefix for the output filename.

  Default: ``plot_``
- **planes** *(str)*: Works only with filenames ending in 'x' and 'y' and columns ending in X or Y.
  These suffixes will be attached to the given files and y_columns.

  Choices: ``('X', 'Y')``
- **plot_styles** *(str)*: Which plotting styles to use,
  either from plotting.styles.*.mplstyles or default mpl.

  Default: ``['standard']``
- **same_axes** *(str)*: Combine plots into single axes. Multiple choices possible.

  Choices: ``['files', 'columns', 'planes']``
- **same_figure** *(str)*: Plot two axes into the same figure (can't be the same as 'same_axes').
  Has no effect if there is only one of the given thing.

  Choices: ``['files', 'columns', 'planes']``
- **share_xaxis**: In case of multiple axes per figure, share x-axis.

  Action: ``store_true``
- **show**: Shows plots.

  Action: ``store_true``
- **single_legend**: Show only one legend instance (at the top plot).

  Action: ``store_true``
- **vertical_lines** *(DictAsString)*: List of vertical lines (e.g. IR positions) to plot.
  Need to contain arguments for axvline, and may contain the additional keys "text" and "loc"
  which is one of  ['bottom', 'top', 'line bottom', 'line top'] and places the text at the given location.

  Default: ``[]``
- **x_labels** *(str)*: Labels for the x-axis, default: x_columns.

- **x_lim** *(float, int, None)*: Limits on the x axis (Tupel)

- **y_labels**: Override labels for the y-axis, default: file_labels or column_labels (depending on same_axes).
  Needs to be a list of lists, where the inner list goes over the axes in one figure and the outer over the figures.
  If the respective length is 1, the same label will be used for all figures or axes.

- **y_lim** *(float, int None)*: Limits on the y axis (Tupel)
"""
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import matplotlib
from matplotlib import pyplot as plt, rcParams
from matplotlib.axes import Axes

import tfs
from generic_parser import EntryPointParameters, entrypoint
from generic_parser.entry_datatypes import DictAsString

from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import EXT
from omc3.plotting.optics_measurements.constants import DEFAULTS
from omc3.plotting.optics_measurements.utils import FigureCollector, DataSet, IDMap, safe_format
from omc3.plotting.utils.windows import (
    PlotWidget, SimpleTabWindow, is_qtpy_installed, log_no_qtpy_many_windows, create_pyplot_window_from_fig
)
from omc3.plotting.spectrum.utils import get_unique_filenames, output_plot
from omc3.plotting.utils import (
    annotations as pannot, 
    lines as plines,
    style as pstyle, 
    colors as pcolors,
)
from omc3.plotting.utils.lines import VERTICAL_LINES_TEXT_LOCATIONS
from omc3.utils.iotools import PathOrStr, save_config, OptionalStr, OptionalFloat
from omc3.utils.logging_tools import get_logger, list2str

LOG = get_logger(__name__)


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="files",
        help=("Path to files to plot. "
              "If planes are used, replace the plane in the filename with '{0}'"),
        required=True,
        nargs="+",
        type=PathOrStr,
    )
    params.add_parameter(
        name="y_columns",
        help="List of column names to plot (e.g. BETX, BETY or BET{0} if `planes` is used.)",
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
        type=OptionalStr,
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
        type=PathOrStr,
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
        type=OptionalFloat,
        help='Limits on the x axis (Tupel)'
    )
    params.add_parameter(
        name="y_lim",
        nargs=2,
        type=OptionalFloat,
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
        help='Which plotting styles to use, either from plotting.styles.*.mplstyles or default mpl.'
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
    params.add_parameter(
        name="share_xaxis",
        help="In case of multiple axes per figure, share x-axis.",
        action="store_true",
    )
    return params


# Main invocation ############################################################


@entrypoint(get_params(), strict=True)
def plot(opt):
    """Main plotting function."""
    LOG.debug(f"Starting plotting of tfs files: {list2str(opt.files):s}")
    if opt.output is not None:
        save_config(Path(opt.output), opt, __file__)

    # preparations
    opt = _check_opt(opt)
    pstyle.set_style(opt.plot_styles, opt.manual_style)

    # extract data
    fig_collection = sort_data(opt.get_subdict(
        ['files', 'planes', 'x_columns', 'y_columns', 'error_columns',
         'file_labels', 'column_labels', 'x_labels', 'y_labels',
         'same_axes', 'same_figure', 'output', 'output_prefix',
         'share_xaxis']))

    # plotting
    _create_plots(fig_collection,
                  opt.get_subdict(['x_lim', 'y_lim',
                                   'ncol_legend', 'single_legend',
                                   'change_marker', 'errorbar_alpha',
                                   'vertical_lines', 'show']))

    return fig_collection.fig_dict


# Sorting ----------------------------------------------------------------------

def sort_data(opt):
    """Load all data from files and sort into figures."""
    collector = FigureCollector()
    axes_ids = _get_axes_ids(opt)

    if len(axes_ids) == 1 and opt.share_xaxis:
        LOG.warning("Option `share_xaxis` is activated but only one axes is present. The option has no effect.")

    same_axes_set = frozenset()
    if opt.same_axes:
        same_axes_set = frozenset(opt.same_axes)

    for (file_path, filename), file_label in zip(get_unique_filenames(opt.files), opt.file_labels):
        for x_col, y_col, err_col, x_label, column_label in zip(
                opt.x_columns, opt.y_columns, opt.error_columns, opt.x_labels, opt.column_labels):

            if opt.planes is None:
                id_map = get_id(filename, y_col, file_label, column_label, same_axes_set,
                                opt.same_figure, opt.output_prefix)
                output_path = get_full_output_path(opt.output, id_map.figure_id)

                collector.add_data_for_id(
                    figure_id=id_map.figure_id,
                    label=id_map.legend_label,
                    data=_read_data(file_path, x_col, y_col, err_col),
                    path=output_path,
                    x_label=x_label,
                    y_label=id_map.ylabel,
                    axes_id=id_map.axes_id,
                    axes_ids=axes_ids,
                )

            else:
                for plane in opt.planes:
                    id_map = get_id(filename, y_col, file_label, column_label, same_axes_set,
                                    opt.same_figure, opt.output_prefix, plane, opt.planes)
                    output_path = get_full_output_path(opt.output, id_map.figure_id)

                    file_path_plane = file_path.with_name(f'{safe_format(file_path.name, plane.lower())}')
                    y_col_plane = safe_format(y_col, plane.upper())
                    err_col_plane = safe_format(err_col, plane.upper())
                    x_col_plane = safe_format(x_col, plane.upper())

                    collector.add_data_for_id(
                        figure_id=id_map.figure_id,
                        label=id_map.legend_label,
                        data=_read_data(file_path_plane, x_col_plane, y_col_plane, err_col_plane),
                        path=output_path,
                        x_label=x_label,
                        y_label=id_map.ylabel,
                        axes_id=id_map.axes_id,
                        axes_ids=axes_ids,
                    )
    if opt.y_labels:
        collector = _update_y_labels(collector, opt.y_labels)

    if opt.share_xaxis:
        collector = _share_xaxis(collector)

    return collector


def get_id(filename_parts, column, file_label, column_label, same_axes, same_figure, prefix, plane='', planes=[]):
    """
    Get the right IDs for the current sorting way.
    This is where the actual sorting happens, by mapping the right IDs according to the chosen
    options.
    """
    planes = "".join(planes)

    file_last = filename_parts[-1].replace(EXT, "").strip("_")
    file_output = "_".join(filename_parts).replace(EXT, "").strip("_")
    if file_label is not None:
        file_output = file_label if file_last in file_label else f'{file_label}_{file_last}'
    else:
        file_label = file_output
    file_output_planes = safe_format(file_output, planes.lower())
    file_output = safe_format(file_output, plane)

    column_planes = safe_format(column, planes)
    column = safe_format(column, plane)
    if column_label is None:
        column_label = column
    else:
        column_label = safe_format(column_label, plane.lower())

    axes_id = {'files': '_'.join(filename_parts),
               'columns': f'{column}',
               'planes': f'{plane.lower()}'
               }.get(same_figure, '')

    key = same_axes if same_figure is None else same_axes.union({same_figure})

    out_idmap = {
        frozenset(['files', 'columns', 'planes']): IDMap(
            figure_id=f'{prefix}{planes}',
            axes_id=axes_id,
            legend_label=f'{file_label} {column_label}',
            ylabel='',
        ),
        frozenset(['files', 'columns']): IDMap(
            figure_id=f'{prefix}{plane.lower()}',
            axes_id=axes_id,
            legend_label=f'{file_label} {column_label}',
            ylabel='',
        ),
        frozenset(['planes', 'columns']): IDMap(
            figure_id=f'{prefix}{file_output_planes}',
            axes_id=axes_id,
            legend_label=column_label,
            ylabel=column_label,
        ),
        frozenset(['planes', 'files']): IDMap(
            figure_id=f'{prefix}{column_planes}',
            axes_id=axes_id,
            legend_label=f'{file_label} {column_label}',
            ylabel=column_label,
        ),
        frozenset(['planes']): IDMap(
            figure_id=f'{prefix}{file_output_planes}_{column_planes}',
            axes_id=axes_id,
            legend_label=column_label,
            ylabel=column_label,
        ),
        frozenset(['files']): IDMap(
            figure_id=f'{prefix}{column}',
            axes_id=axes_id,
            legend_label=f'{file_label}',
            ylabel=column_label,
        ),
        frozenset(['columns']): IDMap(
            figure_id=f'{prefix}{file_output}',
            axes_id=axes_id,
            legend_label=column_label,
            ylabel=f'{file_label}',
        ),
        frozenset([]): IDMap(
            figure_id=f'{prefix}{file_output}_{column}',
            axes_id=axes_id,
            ylabel=column_label,
            legend_label='',
        ),
    }[key]

    out_idmap.figure_id = out_idmap.figure_id.strip("_")
    return out_idmap


def _read_data(file_path, x_col, y_col, err_col):
    tfs_data = tfs.read(file_path)
    return DataSet(
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
            fig_container.ylabels[ax_id] = safe_format(axes_labels[idx_ax], ax_id)
    return fig_collection


def _share_xaxis(fig_collection):
    """Shared xaxis at last axes and remove all xlabels, ticks of other axes."""
    for fig_container in fig_collection.figs.values():
        axs = fig_container.axes.values()

        # Axes.get_shared_x_axes() does not work here as it returns a GrouperView
        # instead of the Grouper (i.e. _shared_axes['x'], matplotlib < 3.8), 
        # so we access _shared_axes directly
        fig_container.axes[fig_container.axes_ids[-1]]._shared_axes["x"].join(*axs)

        # remove ticks and labels for all but the last axes.
        for ax_id in fig_container.axes_ids[:-1]:
            fig_container.axes[ax_id].set_xticklabels([])
            fig_container.xlabels[ax_id] = None

    return fig_collection


# Plotting ---------------------------------------------------------------------


def _create_plots(fig_collection, opt):
    """Main plotting routine."""
    for fig_container in fig_collection.figs.values():
        for idx_ax, ax_id in enumerate(fig_container.axes_ids):
            ax, data, xlabel, ylabel = fig_container[ax_id]
            _plot_vlines(ax, opt.vertical_lines)
            _plot_data(ax, data, opt.change_marker, opt.errorbar_alpha)
            _set_axes_layout(ax, x_lim=opt.x_lim, y_lim=opt.y_lim, xlabel=xlabel, ylabel=ylabel)

            # plt.show(block=False)  # for debugging
            if idx_ax == 0 or not opt.single_legend:
                pannot.make_top_legend(ax, opt.ncol_legend)

        output_plot(fig_container)

    if opt.show:
        if len(fig_collection) > 1 and is_qtpy_installed():
            window = SimpleTabWindow("Tfs Plots")
            for fig_container in fig_collection.figs.values():
                tab = PlotWidget(fig_container.fig, title=fig_container.id)
                window.add_tab(tab)
            window.show()
            return
        
        if len(fig_collection) > rcParams['figure.max_open_warning']:
            log_no_qtpy_many_windows()

        for fig_container in fig_collection.figs.values():
            create_pyplot_window_from_fig(fig_container.fig, title=fig_container.id)
        plt.show()

            

def _plot_data(ax: Axes, data: Dict[str, DataSet], change_marker: bool, ebar_alpha: float):
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


def _set_axes_layout(ax, x_lim, y_lim,  xlabel, ylabel):
    ax.set_xlim(x_lim, auto=not x_lim or any(x_lim))
    ax.set_xlabel(xlabel)
    ax.set_ylim(y_lim, auto=not y_lim or any(y_lim))
    ax.set_ylabel(ylabel)


# Output ---


def get_full_output_path(folder, filename):
    if folder is None or filename is None:
        return None
    return Path(folder) / f"{filename}.{matplotlib.rcParams['savefig.format']}"


# Helper -----------------------------------------------------------------------


def _check_opt(opt):
    """Sanity checks for the opt structure and broadcasting of parameters."""
    if opt.file_labels is None:
        opt.file_labels = [None] * len(opt.files)
    elif len(opt.file_labels) != len(opt.files):
        raise AttributeError("The number of file-labels and number of files differ!")

    if len(opt.x_columns) == 1 and len(opt.y_columns) > 1:
        opt.x_columns *= len(opt.y_columns)
    elif len(opt.x_columns) != len(opt.y_columns):
        raise AttributeError("The number of x-columns and y-columns differ!")

    if opt.x_labels is None:
        opt.x_labels = [None] * len(opt.x_columns)
    elif len(opt.x_labels) == 1 and len(opt.x_columns) > 1:
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
    """
    Get's all id's first (to know how many axes to use). So the `get_id` is done later for
    actual plotting again.
    The order matters, as this function determines the order in which the data is distributed on
    the figure axes (if multiple), which is important if one gives the manual y-labels option.
    Couldn't find a quicker way... (jdilly)
    """
    axes_ids = []  # needs to be something ordered
    same_axes_set = frozenset()
    if opt.same_axes:
        same_axes_set = frozenset(opt.same_axes)
    for (file_path, filename), file_label in zip(get_unique_filenames(opt.files), opt.file_labels):
        for x_col, y_col, err_col, x_label, column_label in zip(
                opt.x_columns, opt.y_columns, opt.error_columns, opt.x_labels, opt.column_labels):

            if opt.planes is None:
                id_ = get_id(filename, y_col, file_label, column_label, same_axes_set,
                             opt.same_figure, opt.output_prefix).axes_id
                axes_ids.append(id_)
            else:
                for plane in opt.planes:
                    id_ = get_id(filename, y_col, file_label, column_label, same_axes_set,
                                 opt.same_figure, opt.output_prefix, plane, opt.planes).axes_id
                    axes_ids.append(id_)
    return list(OrderedDict.fromkeys(axes_ids).keys())  # unique list with preserved order


def _get_marker(idx, change):
    """Return the marker used"""
    if change:
        return plines.MarkerList.get_marker(idx)
    else:
        return rcParams['lines.marker']


# Script Mode ------------------------------------------------------------------


if __name__ == "__main__":
    plot()
