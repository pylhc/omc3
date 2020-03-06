"""
Plot Optics Measurements
--------------------------

Wrapper for `plot_tfs` to easily plot the results from optics measurements.


"""

import os
from collections import OrderedDict
from pathlib import Path

from generic_parser import EntryPointParameters, entrypoint
from generic_parser.entry_datatypes import DictAsString
from generic_parser.entrypoint_parser import save_options_to_config

from omc3.definitions import formats
from omc3.plotting.optics_measurements.constants import DEFAULTS
from omc3.plotting.plot_tfs import plot as plot_tfs
from omc3.utils.logging_tools import get_logger, list2str

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
        help="Optics parameters to plot.",
        required=True,
        type=str,
        nargs="+",
    )
    params.add_parameter(
        name="ip_positions",
        help=("Input to plot IP-Positions into the plots. "
              "Either 'LHCB1' or 'LHCB2' for LHC defaults, "
              "a dictionary of labels and positions "
              "or path to TFS file of a model."),
    )
    params.add_parameter(
        name="x_axis",
        help="Which parameter to use for the x axis.",
        choices=['location', 'phase-advance'],
        default='location',
    )
    # Parameters that are only passed on ---
    params.add_parameter(
        name="combine_by",
        help="Combine plots into one. Either files, planes (not separated into two axes) or both.",
        action="store_true",
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

    opt = _check_opt(opt)

    ip_positions = _get_ip_positions(opt.ip_positions)
    x_column, x_label = _get_x_options(opt.x_axis)
    files, file_labels, y_columns, y_labels = _get_plotting_options(opt.folders, opt.optics_parameters)

    return plot_tfs(
        files=files,
        file_labels=file_labels,
        y_columns=y_columns,
        y_labels=y_labels,
        x_columns=[x_column],
        xlabels=[x_label],
        planes=['XY'],
        vertical_lines=ip_positions,
        **opt.get_subdict(['show', 'combine_by', 'output',
                           'plot_styles', 'manual_style',
                           'change_marker', 'errorbar_alpha', 'ncol_legend'])
    )


def _set_ylabel(ax, default, y_label, y_plane, chromatic):
    """ Tries to set a mapped y label, otherwise the default """
    try:
        annotations.set_yaxis_label(_map_proper_name(y_label),
                                    y_plane, ax, chromcoup=chromatic)
    except (KeyError, ValueError):
        ax.set_ylabel(default)


def _map_proper_name(name):
    """ Maps to a name understood by plotstyle. """
    return {
        "BET": "beta",
        "BB": "betabeat",
        "D": "dispersion",
        "ND": "norm_dispersion",
        "MU": "phase",
        "X": "co",
        "Y": "co",
        "PHASE": "phase",
        "I": "imag",
        "R": "real",
    }[name.upper()]


def _get_ir_positions(all_data, x_cols):
    """ Check if x is position around the ring and return ir positions if possible """
    ir_pos = None
    x_is_pos = all([xc == "S" for xc in x_cols])
    if x_is_pos:
        ir_pos = _find_ir_pos(all_data)
    return ir_pos, x_is_pos


def _get_auto_scale(y_val, scaling):
    """ Find the y-limits so that scaling% of the points are visible """
    y_sorted = sorted(y_val)
    n_points = len(y_val)
    y_min = y_sorted[int(((1 - scaling/100.) / 2.) * n_points)]
    y_max = y_sorted[int(((1 + scaling/100.) / 2.) * n_points)]
    return y_min, y_max


def _find_ir_pos(all_data):
    """ Return the middle positions of the interaction regions """
    ip_names = ["IP" + str(i) for i in range(1, 9)]
    for data in all_data:
        try:
            ip_pos = data.loc[ip_names, 'S'].values
        except KeyError:
            try:
                # loading failed, use defaults
                return IR_POS_DEFAULT[data.SEQUENCE]
                # return {}
            except AttributeError:
                # continue looking
                pass
        else:
            return dict(zip(ip_names, ip_pos))

    # did not find ips or defaults
    return {}

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


def _save_options_to_config(opt):
    output_dir = Path(opt.output)
    os.makedirs(output_dir, exist_ok=True)
    save_options_to_config(output_dir / formats.get_config_filename(__file__),
                           OrderedDict(sorted(opt.items()))
                           )
