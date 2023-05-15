"""
Plot Correction Test
--------------------

Create plots for the correction tests performed with `omc3.scripts.correction_test`.

**Arguments:**

*--Required--*

- **input_dir** *(PathOrStr)*:

    Path to the `output_dir` from `omc3.correction_test`.


*--Optional--*

- **change_marker**:

    Changes marker for each line in the plot.

    action: ``store_true``


- **combine_by**:

    Combine plots into one. Either files, planes (not separated into two
    axes) or both.

    choices: ``['files', 'planes']``


- **corrections** *(str)*:

    Corrections to plot (assumed to be subfolders in `input_dir`).

    default: ``['']``


- **errorbar_alpha** *(float)*:

    Alpha value for error bars

    default: ``0.6``


- **individual_to_input**:

    Save plots for the individual corrections into the corrections input
    folders. Otherwise they go with suffix into the output_folders.

    action: ``store_true``


- **ip_positions**:

    Input to plot IP-Positions into the plots. Either 'LHCB1' or 'LHCB2'
    for LHC defaults, a dictionary of labels and positions or path to TFS
    file of a model.


- **ip_search_pattern**:

    In case your IPs have a weird name. Specify regex pattern.

    default: ``IP\d$``


- **lines_manual** *(DictAsString)*:

    List of manual lines to plot. Need to contain arguments for axvline,
    and may contain the additional keys "text" and "loc" which is one of
    ['bottom', 'top', 'line bottom', 'line top'] and places the text at
    the given location.

    default: ``[]``


- **manual_style** *(DictAsString)*:

    Additional style rcParameters which update the set of predefined ones.

    default: ``{}``


- **meas_dir** *(PathOrStr)*:

    Path to the directory containing the measurement filesto plot the
    measurement as comparison.If not given, the data from the first
    corrections directory will be used.


- **ncol_legend** *(int)*:

    Number of bpm legend-columns. If < 1 no legend is shown.

    default: ``3``


- **output_dir** *(PathOrStr)*:

    Path to save the plots into. If not given, no plots will be saved.


- **plot_styles** *(str)*:

    Which plotting styles to use, either from plotting.styles.*.mplstyles
    or default mpl.

    default: ``['standard', 'correction_test']``


- **share_xaxis**:

    In case of multiple axes per figure, share x-axis.

    action: ``store_true``


- **show**:

    Shows plots.

    action: ``store_true``


- **suppress_column_legend**:

    Does not show column name in legend e.g. when combining by files (see
    also `ncol_legend`).

    action: ``store_true``


- **x_axis**:

    Which parameter to use for the x axis.

    choices: ``['location', 'phase-advance']``

    default: ``location``


- **x_lim** *(MultiClass)*:

    Limits on the x axis (Tupel)


- **y_lim** *(MultiClass)*:

    Limits on the y axis (Tupel)


"""
from pathlib import Path
from typing import Dict, Iterable, Set

from matplotlib import pyplot as plt

import tfs
from generic_parser import DotDict, EntryPointParameters, entrypoint
from omc3.correction.constants import (CORRECTED_LABEL, UNCORRECTED_LABEL, CORRECTION_LABEL, EXPECTED_LABEL,
                                       COUPLING_NAME_TO_MODEL_COLUMN_SUFFIX)
from omc3.definitions.optics import FILE_COLUMN_MAPPING, ColumnsAndLabels, RDT_COLUMN_MAPPING
from omc3.optics_measurements.constants import EXT
from omc3.plotting.plot_optics_measurements import (_get_x_axis_column_and_label, _get_ip_positions,
                                                    get_optics_style_params, get_plottfs_style_params)
from omc3.plotting.plot_tfs import plot as plot_tfs, get_full_output_path
from omc3.plotting.utils import (annotations as pannot)
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr

LOG = logging_tools.get_logger(__name__)

SPLIT_ID = "#_#"  # will appear in the figure ID, but should be fine to read
PREFIX = "plot_corrections"

def get_plotting_params() -> EntryPointParameters:
    params = EntryPointParameters()
    params.add_parameter(name="input_dir",
                         required=True,
                         type=PathOrStr,
                         help="Path to the `output_dir` from `omc3.correction_test`.",
                         )
    params.add_parameter(name="corrections",
                         nargs="+",
                         type=str,
                         help="Corrections to plot (assumed to be subfolders in `input_dir`).",
                         default=[""],  # empty string means "directly in input-dir"
                         )
    params.add_parameter(name="meas_dir",
                         type=PathOrStr,
                         help="Path to the directory containing the measurement files"
                              "to plot the measurement as comparison."
                              "If not given, the data from the first corrections directory will be used.",)
    params.add_parameter(name="output_dir",
                         type=PathOrStr,
                         help="Path to save the plots into. If not given, no plots will be saved.",
                         )
    params.update(get_plotting_style_parameters())
    return params


def get_plotting_style_parameters():
    """ Parameters related to the style of the plots. """
    params = EntryPointParameters()
    params.add_parameter(name="individual_to_input",
                         action="store_true",
                         help="Save plots for the individual corrections "
                              "into the corrections input folders. "
                              "Otherwise they go with suffix into the output_folders."
                         )
    params.update(get_optics_style_params())
    params.update(get_plottfs_style_params())
    params["plot_styles"]["default"] = params["plot_styles"]["default"] + ["correction_test"]
    return params


@entrypoint(get_plotting_params(), strict=True)
def plot_checked_corrections(opt: DotDict):
    """ Entrypoint for the plotting function. """
    LOG.info("Plotting checked corrections.")
    # Preparations -------------------------------------------------------------
    correction_dirs: Dict[str, Path] = {}
    if len(opt.corrections) == 1 and not opt.corrections[0]:
        correction_dirs[CORRECTED_LABEL] = opt.input_dir
        opt.individual_to_input = False  # we save into output directory
    else:
        for correction in opt.corrections:
            correction_dirs[correction] = opt.input_dir / correction

    measurements: Path = opt.meas_dir or list(correction_dirs.values())[0]

    files = _get_corrected_measurement_names(correction_dirs.values())
    ip_positions = _get_ip_positions(opt.ip_positions, opt.x_axis, opt.ip_search_pattern)
    x_colmap = _get_x_axis_column_and_label(opt.x_axis)

    # Plotting -----------------------------------------------------------------
    fig_dict = {}
    for filename in files:
        try:
            y_colmap = FILE_COLUMN_MAPPING[filename[:-1]].set_plane(filename[-1].upper())
        except KeyError:
            if filename not in COUPLING_NAME_TO_MODEL_COLUMN_SUFFIX.keys():
                LOG.debug(f"Checked file {filename} will not be plotted.")
                continue
            # Coupling RDTS:
            # get F1### column map without the I, R, A, P part based on the rdt-filename:
            LOG.debug(f"Plotting coupling correction for {filename}")
            new_figs = {}
            for idx, y_colmap in enumerate(RDT_COLUMN_MAPPING.values()):  # AMP, PHASE, REAL or IMAG as column-map
                y_colmap = ColumnsAndLabels(
                    _column=y_colmap.column,
                    _label=y_colmap.label.format(filename),  # this one needs additional info
                    _text_label=y_colmap.text_label,
                    needs_plane=False
                )
                new_figs.update(
                    _create_correction_plots_per_filename(
                        filename=filename,
                        measurements=measurements,
                        correction_dirs=correction_dirs,
                        x_colmap=x_colmap,
                        y_colmap=y_colmap,
                        ip_positions=ip_positions,
                        opt=opt
                    )
                )
        else:
            LOG.debug(f"Plotting correction for {filename}")
            new_figs = _create_correction_plots_per_filename(
                filename=filename,
                measurements=measurements,
                correction_dirs=correction_dirs,
                x_colmap=x_colmap,
                y_colmap=y_colmap,
                ip_positions=ip_positions,
                opt=opt
            )

        fig_dict.update(new_figs)

    # Output -------------------------------------------------------------------
    save_plots(opt.output_dir, figure_dict=fig_dict, input_dir=opt.input_dir if opt.individual_to_input else None)
    show_plots(opt.show)
    return fig_dict

def _create_correction_plots_per_filename(filename, measurements, correction_dirs, x_colmap, y_colmap, ip_positions, opt):
    """ Plot measurements and all different correction scenarios into a single plot. """
    full_filename = f"{filename}{EXT}"
    file_label = filename
    if filename in COUPLING_NAME_TO_MODEL_COLUMN_SUFFIX.keys():
        file_label = f"{file_label}_{y_colmap.text_label}"

    # Plot corrections via plot_tfs (as they all have the same column names) ---

    # Plot expectation of all correction scenarios into one plot
    figs = plot_tfs(
        files=[path / full_filename for path in correction_dirs.values()],
        file_labels=list(correction_dirs.keys()),  # defines the legend
        y_columns=[y_colmap.expected_column],
        column_labels=[y_colmap.delta_label],  # defines y-axis label
        error_columns=[y_colmap.error_expected_column],
        x_columns=[x_colmap.column],
        x_labels=[x_colmap.label],
        vertical_lines=ip_positions + opt.lines_manual,
        same_axes=["files"],
        output_prefix=f"{PREFIX}_{file_label}_",  # used in the id, which is the fig_dict key
        **opt.get_subdict([
            'plot_styles', 'manual_style',
            'change_marker', 'errorbar_alpha',
            'ncol_legend', 'x_lim', 'y_lim',
            'share_xaxis'
        ])
    )

    # Plot expectation and correction individually
    for name, path in correction_dirs.items():
        figs.update(
            plot_tfs(
                files=[path / full_filename],
                file_labels=[file_label],
                y_labels=[[y_colmap.delta_label]],  # defines y-axis label
                y_columns=[y_colmap.expected_column, y_colmap.diff_correction_column],
                column_labels=[EXPECTED_LABEL, CORRECTION_LABEL],
                error_columns=[y_colmap.error_expected_column, None],
                x_columns=[x_colmap.column],
                x_labels=[x_colmap.label],
                vertical_lines=ip_positions + opt.lines_manual,
                same_axes=["columns"],
                output_prefix=f"{name}{SPLIT_ID}",  # used in the id, which is the fig_dict key
                **opt.get_subdict([
                    'plot_styles', 'manual_style',
                    'change_marker', 'errorbar_alpha',
                    'ncol_legend', 'x_lim', 'y_lim',
                    'share_xaxis'
                ])
            )
        )

    # Add the measurement data to the plots (has different column names) -------
    df_measurement = tfs.read_tfs(measurements / full_filename)
    xlim = opt.x_lim or (df_measurement[x_colmap.column].min(), df_measurement[x_colmap.column].max())
    errors = None
    try:
        errors = df_measurement[y_colmap.error_delta_column]
    except KeyError:
        LOG.warning(
            f"Could not find {y_colmap.error_delta_column} in {full_filename}. "
            f"Probably an old file? Assuming zero errors."
        )

    for fig in figs.values():
        ax = fig.gca()
        ax.errorbar(
            df_measurement[x_colmap.column],
            df_measurement[y_colmap.delta_column],
            errors,
            label=UNCORRECTED_LABEL,
            color="k",
            zorder=-1,
        )
        ax.set_xlim(xlim)
        pannot.make_top_legend(ax, opt.ncol_legend)

    return figs


def show_plots(show: bool):
    """ Show plots if so desired. """
    # plt.show()
    if show:
        plt.show()


def save_plots(output_dir, figure_dict, input_dir=None):
    """ Save the plots. """
    if not output_dir and not input_dir:
        return

    for figname, fig in figure_dict.items():
        outdir = output_dir
        figname_parts = figname.split(SPLIT_ID)
        if len(figname_parts) == 1:  # no SPLIT_ID
            # these are the combined plots. They have the column name at the end,
            # which we do not care for here at the moment.
            # In case of multiple columns per file, this could be brought back
            # (then we would also not need the RDT check).
            figname = "_".join(figname.split("_")[:-1])
        else:
            # this is then the individual plots
            if input_dir:
                # files go directly into the correction-scenario folders
                outdir = input_dir / figname_parts[0]
                figname = f"{PREFIX}_{figname_parts[1]}"
            else:
                # everything goes into the output-dir (if given), but needs prefix
                figname = "_".join([PREFIX] + figname_parts)

        output_path = get_full_output_path(outdir, figname)
        if output_path is not None:
            LOG.info(f"Saving Corrections Plot to '{output_path}'")
            fig.savefig(output_path)


def _get_corrected_measurement_names(correction_dirs: Iterable[Path]) -> Set[str]:
    """ Check all the corrections dirs for common tfs files."""
    tfs_files = None
    for idx, correction in enumerate(correction_dirs):
        new_files = set(f.stem for f in correction.glob(f"*{EXT}"))
        if not idx:
            tfs_files = new_files
            continue

        tfs_files &= new_files
    # tfs_files -= {Path(MODEL_MATCHED_FILENAME).stem, Path(MODEL_NOMINAL_FILENAME).stem}  # no need, filtered later anyway
    return tfs_files
