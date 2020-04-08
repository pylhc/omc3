"""
Plot Optics Measurements
--------------------------

Wrapper for `plot_tfs` to easily plot the results from optics measurements.


"""

import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import tfs
from generic_parser import EntryPointParameters, entrypoint
from generic_parser.entry_datatypes import DictAsString, get_multi_class
from generic_parser.entrypoint_parser import save_options_to_config
from omc3.optics_measurements.rdt import _rdt_to_order_and_type

from omc3.plotting.spectrum.utils import get_unique_filenames

from omc3.definitions import formats
from omc3.plotting.optics_measurements.constants import (DEFAULTS,
                                                         XAXIS, YAXIS,
                                                         IP_POS_DEFAULT)
from omc3.optics_measurements.constants import ERR, DELTA, AMPLITUDE, PHASE, EXT
from omc3.plotting.plot_tfs import plot as plot_tfs
from omc3.utils.logging_tools import get_logger, list2str
from omc3.plotting.utils.lines import VERTICAL_LINES_TEXT_LOCATIONS

LOG = get_logger(__name__)


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="folders",
        help="Optics Measurements folders containing the analysed data.",
        required=True,
        nargs="+",
        type=str,
    )
    params.add_parameter(
        name="optics_parameters",
        help=("Optics parameters to plot, e.g. 'beta_amplitude'. "
              "RDTs need to be specified with plane, e.g. 'f1001_x'"),
        required=True,
        type=str,
        nargs="+",
    )
    params.add_parameter(
        name="delta",
        help="Plot the difference to model instead of the parameter.",
        action="store_true"
    )
    params.add_parameter(
        name="ip_positions",
        help=("Input to plot IP-Positions into the plots. "
              "Either 'LHCB1' or 'LHCB2' for LHC defaults, "
              "a dictionary of labels and positions "
              "or path to TFS file of a model."),
    )
    params.add_parameter(name="lines_manual",
                         nargs="*",
                         default=[],
                         type=DictAsString,
                         help='List of manual lines to plot. Need to contain arguments for axvline, and may contain '
                              'the additional keys "text" and "loc" which is one of '
                              f'{list(VERTICAL_LINES_TEXT_LOCATIONS.keys())} and places the text at the given location.'
                         )
    params.add_parameter(
        name="ip_search_pattern",
        default=r"IP\d",
        help="In case your IPs have a weird name. Specify regex pattern.",
    )
    params.add_parameter(
        name="x_axis",
        help="Which parameter to use for the x axis.",
        choices=list(XAXIS.keys()),
        default='location',
    )
    # Parameters that are only passed on ---
    params.add_parameter(
        name="combine_by",
        help="Combine plots into one. Either files, planes (not separated into two axes) or both.",
        nargs="+",
        choices=['files', 'planes']  # combine by columns does not really make sense
    )
    params.add_parameter(
        name="output",
        help="Folder to output the results to.",
        type=str,
    )
    params.add_parameter(
        name="show",
        help="Shows plots.",
        action="store_true",
    )
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


@entrypoint(get_params(), strict=True)
def plot(opt):
    LOG.info("Starting plotting of optics measurement data: "
             f"{list2str(opt.optics_parameters)} in {list2str(opt.folders):s}")

    if opt.output is not None:
        _save_options_to_config(opt)

    # opt = _check_opt(opt)

    ip_positions = _get_ip_positions(opt.ip_positions, opt.x_axis, opt.ip_search_pattern)
    x_column, x_label = _get_x_options(opt.x_axis)

    for optics_parameter in opt.optics_parameters:
        plot_opts = _get_plotting_options(
            opt.folders, optics_parameter, opt.delta
        )

        if plot_opts.pop('is_rdt'):
            fig_dict = {}
            for idxs in (slice(0, 2), slice(2, 4)):  # amp,phase - real,imag
                fig_dict.update(
                    plot_tfs(
                        files=plot_opts['files'],
                        file_labels=plot_opts['file_labels'],
                        y_columns=plot_opts['y_columns'][idxs],
                        y_labels=plot_opts['y_labels'][idxs],
                        error_columns=plot_opts['error_columns'][idxs],
                        x_columns=[x_column],
                        xlabels=[x_label],
                        vertical_lines=ip_positions + opt.lines_manual,
                        same_figure="columns",
                        same_axes=opt.comibe_by,
                        **opt.get_subdict(['show', 'output',
                                           'plot_styles', 'manual_style',
                                           'change_marker', 'errorbar_alpha', 'ncol_legend'])
                ))
            return fig_dict

        else:
            same_fig = None
            if opt.combine_by is not None and 'planes' not in opt.combine_by:
                same_fig = 'planes'

            return plot_tfs(
                **plot_opts,
                x_columns=[x_column],
                x_labels=[x_label],
                planes=['X', 'Y'],
                vertical_lines=ip_positions + opt.lines_manual,
                same_figure=same_fig,
                same_axes=opt.combine_by,
                **opt.get_subdict(['show', 'output',
                                   'plot_styles', 'manual_style',
                                   'change_marker', 'errorbar_alpha', 'ncol_legend'])
            )

# IP-Positions -----------------------------------------------------------------


def _get_ip_positions(ip_positions, xaxis, ip_pattern):
    if isinstance(ip_positions, str):
        try:
            positions = IP_POS_DEFAULT[ip_positions]
        except KeyError:
            return _get_ip_positions_from_file(ip_positions, xaxis, ip_pattern)
        else:
            if xaxis != 'location':
                raise NotImplementedError("No default IP positions for "
                                          f"{xaxis} implemented.")

            return [{'text': name, 'x': x,
                     'loc': 'top', 'color': 'k'} for name, x in positions.items()]
    return ip_positions


def _get_ip_positions_from_file(path, axis, pattern):
    model = tfs.read(path, index="NAME")
    ip_mask = model.index.str.match(pattern)
    column = XAXIS[axis][1]
    return model.loc[ip_mask, column].to_dict()


# X-Axis -----------------------------------------------------------------------

def _get_x_options(x_axis):
    return XAXIS[x_axis]


# Y-Axis -----------------------------------------------------------------------


def _get_plotting_options(folders: Iterable, optics_parameter: str, delta: bool):
    is_rdt = optics_parameter.lower().startswith("f")
    files, file_labels = zip(*get_unique_filenames(folders))
    plot_opts = {'is_rdt': is_rdt,
                 'file_labels': list(file_labels),
                 }
    if is_rdt:
        if delta:
            LOG.warning('Delta Columns for RDTs not implemented. Using normal columns.')
        subfolder = _rdt_to_order_and_type(optics_parameter[1:5])
        plot_opts.update(_get_rdt_columns())
        plot_opts['files'] = [f/'rdt'/subfolder/f'{optics_parameter}{EXT}' for f in files]

    else:
        if not optics_parameter.endswith("_"):
            optics_parameter += "_"
        y_column, error_column, y_label = _get_columns_and_label(optics_parameter, delta)
        plot_opts['files'] = [f/optics_parameter for f in files]
        if delta:
            plot_opts['file_labels'] = [f"delta_{label}" for label in plot_opts['file_labels']]
        plot_opts['y_columns'] = [y_column]
        plot_opts['column_labels'] = [y_label]
        plot_opts['error_columns'] = [error_column]

    plot_opts['files'] = [str(path.absolute()) for path in plot_opts['files']]
    return plot_opts


def _get_columns_and_label(parameter, delta):
    column_and_label = YAXIS[parameter]
    column = column_and_label[0]
    label = column_and_label[1]
    if delta:
        column = f"{DELTA}{column}"
        try:
            label = column_and_label[2]
        except IndexError:
            label = _default_delta_from_label(label)

    return column, f"{ERR}{column}", label


def _default_delta_from_label(label):
    return fr'$\Delta {label[1:]}'


def _get_rdt_columns():
    rdt_measures = ['rdt_amp', 'rdt_phase', 'rdt_real', 'rdt_imag']
    result = {key: [None] * len(rdt_measures) for key in ['y_columns', 'y_labels', 'error_columns']}
    for idx, meas in enumerate(rdt_measures):
        column, label = YAXIS[meas]
        result['y_columns'][idx] = column
        result['error_columns'][idx] = f"{ERR}{AMPLITUDE}"
        result['column_labels'][idx] = label
    result['error_columns'][2] = f"{ERR}{PHASE}"
    return result


def _get_auto_scale(y_val, scaling):
    """ Find the y-limits so that scaling% of the points are visible """
    y_sorted = sorted(y_val)
    n_points = len(y_val)
    y_min = y_sorted[int(((1 - scaling/100.) / 2.) * n_points)]
    y_max = y_sorted[int(((1 + scaling/100.) / 2.) * n_points)]
    return y_min, y_max


    # # things to do only once
    # if last_line:
    #     # setting the y_label
    #     if y_label is None:
    #         _set_ylabel(ax, y_col, y_label_from_col, y_plane, chromatic)
    #     else:
    #         y_label_from_label = ""
    #         if y_label:
    #             y_label_from_label, y_plane, _, _, chromatic = _get_names_and_columns(
    #                 idx_plot, xy, y_label, "")
    #         if xy:
    #             y_label = f"{y_label:s} {y_plane:s}"
    #         _set_ylabel(ax, y_label, y_label_from_label, y_plane, chromatic)
    #
    #     # setting x limits
    #     if x_is_position:
    #         with suppress(AttributeError, ValueError):
    #             post_processing.set_xlimits(data.SEQUENCE, ax)
    #
    #     # setting visibility, ir-markers and label
    #     if xy and idx_plot == 0:
    #         ax.axes.get_xaxis().set_visible(False)
    #         if x_is_position and ir_positions:
    #             annotations.show_ir(ir_positions, ax, mode='lines')
    #     else:
    #         if x_is_position:
    #             annotations.set_xaxis_label(ax)
    #             if ir_positions:
    #                 annotations.show_ir(ir_positions, ax, mode='outside')


def _save_options_to_config(opt):
    output_dir = Path(opt.output)
    os.makedirs(output_dir, exist_ok=True)
    save_options_to_config(output_dir / formats.get_config_filename(__file__),
                           OrderedDict(sorted(opt.items()))
                           )


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    # plot()
    import matplotlib
    matplotlib.use('qt5agg')
    plot(folders=['/home/josch/Software/myomc3/optics93'],
         optics_parameters=['beta_amplitude', 'orbit', 'f0012_y', 'f3000_x'],
         output="temp/", show=True, ip_positions="LHCB1")