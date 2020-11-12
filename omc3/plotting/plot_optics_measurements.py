"""
Plot Optics Measurements
------------------------

Wrapper for `plot_tfs` to easily plot the results from optics measurements.

.. code-block:: python

    from omc3.plotting.plot_optics_measurements import plot

    figs = plot(
        folders=['folder1', 'folder2'],
        combine_by=['files'],  # to compare folder1 and folder2
        output='output_directory',
        delta=True,  # delta from reference
        optics_parameters=['orbit', 'beta_phase', 'beta_amplitude',
                           'phase', 'total_phase',
                           'f1001_x', 'f1010_x'],
        x_axis='location',  # or 'phase-advance'
        ip_positions='LHCB1',
        suppress_column_legend=True,
        show=True,
        ncol_legend=2,
    )

**Arguments:**

*--Required--*

- **folders** *(MultiClass)*: Optics Measurements folders containing the analysed data.

- **optics_parameters** *(str)*: Optics parameters to plot, e.g. 'beta_amplitude'.
  RDTs need to be specified with plane, e.g. 'f1001_x'


*--Optional--*

- **change_marker**: Changes marker for each line in the plot.

  Action: ``store_true``
- **combine_by**: Combine plots into one. Either files, planes (not separated into two axes) or both.

  Choices: ``['files', 'planes']``
- **delta**: Plot the difference to model instead of the parameter.

  Action: ``store_true``
- **errorbar_alpha** *(float)*: Alpha value for error bars

  Default: ``0.6``
- **ip_positions**: Input to plot IP-Positions into the plots. Either 'LHCB1'
  or 'LHCB2' for LHC defaults, a dictionary of labels and positions or path to TFS file of a model.

- **ip_search_pattern**: In case your IPs have a weird name. Specify regex pattern.

  Default: ``IP\d$``
- **lines_manual** *(DictAsString)*: List of manual lines to plot.
  Need to contain arguments for axvline, and may contain the additional keys "text"
  and "loc" which is one of ['bottom', 'top', 'line bottom', 'line top']
  and places the text at the given location.

  Default: ``[]``
- **manual_style** *(DictAsString)*: Additional style rcParameters which
  update the set of predefined ones.

  Default: ``{}``
- **ncol_legend** *(int)*: Number of bpm legend-columns. If < 1 no legend is shown.

  Default: ``3``
- **output** *(MultiClass)*: Folder to output the results to.

- **plot_styles** *(str)*: Which plotting styles to use,
  either from plotting.styles.*.mplstyles or default mpl.

  Default: ``['standard']``
- **share_xaxis**: In case of multiple axes per figure, share x-axis.

  Action: ``store_true``
- **show**: Shows plots.

  Action: ``store_true``
- **suppress_column_legend**: Does not show column name in legend
  e.g. when combining by files (see also `ncol_legend`).

  Action: ``store_true``
- **x_axis**: Which parameter to use for the x axis.

  Choices: ``['location', 'phase-advance']``
  Default: ``location``
- **x_lim** *(float, int, None)*: Limits on the x axis (Tupel)

- **y_lim** *(float, int, None)*: Limits on the y axis (Tupel)
"""
from pathlib import Path

import tfs
from generic_parser import EntryPointParameters, entrypoint
from generic_parser.entry_datatypes import DictAsString

from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import ERR, DELTA, AMPLITUDE, PHASE, EXT
from omc3.optics_measurements.rdt import _rdt_to_order_and_type
from omc3.plotting.optics_measurements.constants import (DEFAULTS,
                                                         XAXIS, YAXIS,
                                                         IP_POS_DEFAULT)
from omc3.plotting.plot_tfs import plot as plot_tfs, float_or_none
from omc3.plotting.spectrum.utils import get_unique_filenames
from omc3.plotting.utils.lines import VERTICAL_LINES_TEXT_LOCATIONS
from omc3.utils.iotools import PathOrStr, save_config
from omc3.utils.logging_tools import get_logger, list2str

LOG = get_logger(__name__)


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="folders",
        help="Optics Measurements folders containing the analysed data.",
        required=True,
        nargs="+",
        type=PathOrStr,
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
        default=r"IP\d$",
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
        choices=['files', 'planes'],  # combine by columns does not really make sense
    )
    params.add_parameter(
        name="output",
        help="Folder to output the results to.",
        type=PathOrStr,
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
        name="suppress_column_legend",
        help=("Does not show column name in legend "
              "e.g. when combining by files (see also `ncol_legend`)."),
        action="store_true",
    )
    params.add_parameter(
        name="errorbar_alpha",
        help="Alpha value for error bars",
        type=float,
        default=DEFAULTS['errorbar_alpha'],
    )
    params.add_parameter(
        name="x_lim",
        nargs=2,
        type=float_or_none,
        help='Limits on the x axis (Tupel)'
    )
    params.add_parameter(
        name="y_lim",
        nargs=2,
        type=float_or_none,
        help='Limits on the y axis (Tupel)'
    )
    params.add_parameter(
        name="share_xaxis",
        help="In case of multiple axes per figure, share x-axis.",
        action="store_true",
    )
    return params


@entrypoint(get_params(), strict=True)
def plot(opt):
    """Main plotting function."""
    LOG.info("Starting plotting of optics measurement data: "
             f"{list2str(opt.optics_parameters)} in {list2str(opt.folders):s}")

    if opt.output is not None:
        save_config(Path(opt.output), opt, __file__)

    opt = _check_opt(opt)

    ip_positions = _get_ip_positions(opt.ip_positions, opt.x_axis, opt.ip_search_pattern)
    x_column, x_label = _get_x_options(opt.x_axis)

    fig_dict = {}
    for optics_parameter in opt.optics_parameters:

        is_rdt = optics_parameter.lower().startswith("f")
        files, file_labels = zip(*get_unique_filenames(opt.folders))
        file_labels = ["_".join(flabels) for flabels in file_labels]

        if is_rdt:
            fig_dict.update(_plot_rdt(
                optics_parameter, files, file_labels, x_column, x_label,
                ip_positions, opt,)
            )
        else:
            if not optics_parameter.endswith("_"):
                optics_parameter += "_"

            fig_dict.update(_plot_param(
                optics_parameter, files, file_labels, x_column, x_label,
                ip_positions, opt,)
            )
    return fig_dict


def _check_opt(opt):
    return opt


# Plot RDTs --------------------------------------------------------------------


def _plot_rdt(optics_parameter, files, file_labels, x_column, x_label, ip_positions, opt):
    """Main plotting function for RDTs."""
    fig_dict = {}
    if opt.x_axis != 'location':
        LOG.error("Phase advance not yet implemented in RDT-files. Skipping.")
        return fig_dict

    if opt.delta:
        LOG.warning('Delta Columns for RDTs not implemented. Using normal columns.')

    subfolder = _rdt_to_order_and_type([int(n) for n in optics_parameter[1:5]])
    files = [str(f.absolute()/'rdt'/subfolder/f'{optics_parameter}{EXT}') for f in files]
    columns = _get_rdt_columns()

    combine_by = []
    if opt.combine_by is not None:
        combine_by = list(opt.combine_by)

    prefix = ''
    if "files" in combine_by:
        prefix = f'{optics_parameter}_'

    combine_planes = False
    if "planes" in combine_by:
        combine_by[combine_by.index("planes")] = "columns"
        combine_planes = True

    for idxs, mode in ((slice(0, 2), "amplitude"), (slice(2, 4), "complex")):
        if combine_planes:
            y_labels = [optics_parameter.upper()]
            if opt.suppress_column_legend:
                column_labels = [l.format('F') for l in columns['y_labels'][idxs]]
            else:
                column_labels = [l.format(optics_parameter.upper()) for l in columns['y_labels'][idxs]]

        else:
            y_labels = [l.format(optics_parameter.upper()) for l in columns['y_labels'][idxs]]
            column_labels = [optics_parameter.upper()]
            if opt.suppress_column_legend:
                column_labels = ['']

        fig_dict.update(
            plot_tfs(
                files=files,
                file_labels=list(file_labels),
                y_columns=columns['y_columns'][idxs],
                column_labels=column_labels,
                y_labels=[y_labels],
                error_columns=columns['error_columns'][idxs],
                x_columns=[x_column],
                x_labels=[x_label],
                vertical_lines=ip_positions + opt.lines_manual,
                same_figure="columns" if "columns" not in combine_by else None,
                same_axes=combine_by if len(combine_by) else None,
                single_legend=True,
                output_prefix=f"plot_{mode}_{prefix}",
                **opt.get_subdict(['show', 'output',
                                   'plot_styles', 'manual_style',
                                   'change_marker', 'errorbar_alpha',
                                   'ncol_legend', 'x_lim', 'y_lim',
                                   'share_xaxis'])
            ))

    return fig_dict


def _get_rdt_columns():
    rdt_measures = ['rdt_amp', 'rdt_phase', 'rdt_real', 'rdt_imag']
    result = {key: [None] * len(rdt_measures) for key in ['y_columns', 'y_labels', 'error_columns']}
    for idx, meas in enumerate(rdt_measures):
        column, _, label = YAXIS[meas]
        result['y_columns'][idx] = column
        result['error_columns'][idx] = f"{ERR}{AMPLITUDE}"  # replaced for phase below
        result['y_labels'][idx] = label
    result['error_columns'][rdt_measures.index('rdt_phase')] = f"{ERR}{PHASE}"
    return result


# Plot Other Parameter ---------------------------------------------------------


def _plot_param(optics_parameter, files, file_labels, x_column, x_label, ip_positions, opt):
    """Main plotting function for all parameters but RDTs."""
    y_column, error_column, column_label, y_label = _get_columns_and_label(optics_parameter, opt.delta)

    same_fig = None
    column_labels = [column_label]
    if opt.combine_by is None or 'planes' not in opt.combine_by:
        same_fig = 'planes'

    if opt.combine_by is not None and "planes" in opt.combine_by:  # not else!
        column_labels = [f'{column_label} {{0}}']

    if opt.suppress_column_legend:
        if same_fig == 'planes':
            column_labels = ['']
        else:
            column_labels = ['{0}']   #  show planes in labels as all are in same axes

    prefix = ''
    if opt.delta:
        prefix += f'delta_'
    if opt.combine_by and "files" in opt.combine_by:
        prefix += f'{optics_parameter}'

    return plot_tfs(
        files=[f.absolute()/f'{optics_parameter}{{0}}{EXT}' for f in files],
        file_labels=list(file_labels),
        y_columns=[f'{y_column}{{0}}'],
        y_labels=[[y_label]],
        column_labels=column_labels,
        error_columns=[f'{error_column}{{0}}'],
        x_columns=[x_column],
        x_labels=[x_label],
        planes=list(PLANES),
        vertical_lines=ip_positions + opt.lines_manual,
        same_figure=same_fig,
        same_axes=opt.combine_by,
        single_legend=True,
        output_prefix=f"plot_{prefix}",
        **opt.get_subdict(['show', 'output',
                           'plot_styles', 'manual_style',
                           'change_marker', 'errorbar_alpha',
                           'ncol_legend', 'x_lim', 'y_lim',
                           'share_xaxis'])
    )


def _get_columns_and_label(parameter, delta):
    column_and_label = YAXIS[parameter]
    column = column_and_label[0]
    column_label = column_and_label[1]
    y_label = column_and_label[2]
    if delta:
        column = f"{DELTA}{column}"
        try:
            y_label = column_and_label[3]
        except IndexError:
            y_label = fr'$\Delta {y_label[1:]}'

    return column, f"{ERR}{column}", column_label, y_label


# IP-Positions -----------------------------------------------------------------


def _get_ip_positions(ip_positions, xaxis, ip_pattern):
    if isinstance(ip_positions, (str, Path)):
        try:
            positions = IP_POS_DEFAULT[ip_positions]
        except KeyError:
            return _get_ip_positions_from_file(ip_positions, xaxis, ip_pattern)
        else:
            if xaxis != 'location':
                raise NotImplementedError("No default IP positions for "
                                          f"{xaxis} implemented.")
        return _create_ip_list(positions)

    if ip_positions is None:
        return []

    return ip_positions


def _get_ip_positions_from_file(path, axis, pattern):
    model = tfs.read(path, index="NAME")
    ip_mask = model.index.str.match(pattern)
    column = XAXIS[axis][0]
    return _create_ip_list(model.loc[ip_mask, column])


def _create_ip_list(ip_dict):
    """ip_dict can be series as well."""
    return [{'text': name, 'x': x,
             'loc': 'top', 'color': 'k'} for name, x in ip_dict.items()]


# Other ------------------------------------------------------------------------


def _get_x_options(x_axis):
    return XAXIS[x_axis]


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    plot()
