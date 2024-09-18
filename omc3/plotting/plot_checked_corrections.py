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

    default: ``IP\\d$``


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
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from generic_parser import EntryPointParameters, entrypoint
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from omc3.correction.constants import (
    CORRECTED_LABEL,
    CORRECTION_LABEL,
    COUPLING_NAME_TO_MODEL_COLUMN_SUFFIX,
    EXPECTED_LABEL,
    UNCORRECTED_LABEL,
)
from omc3.definitions.optics import (
    FILE_COLUMN_MAPPING,
    RDT_AMPLITUDE_COLUMN,
    RDT_COLUMN_MAPPING,
    RDT_IMAG_COLUMN,
    RDT_PHASE_COLUMN,
    RDT_REAL_COLUMN,
    ColumnsAndLabels,
    NORM_DISP_NAME,
)
from omc3.optics_measurements.constants import EXT
from omc3.plotting.plot_optics_measurements import (
    _get_ip_positions,
    _get_x_axis_column_and_label,
    get_optics_style_params,
    get_plottfs_style_params,
)
from omc3.plotting.plot_tfs import get_full_output_path
from omc3.plotting.plot_tfs import plot as plot_tfs
from omc3.plotting.utils import annotations as pannot
from omc3.plotting.utils.windows import (
    PlotWidget,
    TabWidget,
    VerticalTabWindow,
    create_pyplot_window_from_fig,
    log_no_qtpy_many_windows,
)
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config

if TYPE_CHECKING:
    from collections.abc import Iterable
    from generic_parser import DotDict


LOG = logging_tools.get_logger(__name__)

SPLIT_ID: str = "#_#"  # will appear in the figure ID, but should be fine to read
PREFIX: str = "plot_corrections"

SINGLE_PLANE_FILES = (NORM_DISP_NAME,)

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

    opt.input_dir = Path(opt.input_dir)
    if opt.output_dir:
        opt.output_dir = Path(opt.output_dir)
        save_config(opt.output_dir, opt, __file__)

    # Preparations -------------------------------------------------------------
    correction_dirs: dict[str, Path] = {}
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
            for y_colmap in RDT_COLUMN_MAPPING.values():  # AMP, PHASE, REAL or IMAG as column-map
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
    if opt.output_dir:
        save_plots(
            opt.output_dir, 
            figure_dict=fig_dict, 
            input_dir=opt.input_dir if opt.individual_to_input else None
        )

    if opt.show:
        show_plots(fig_dict)
    return fig_dict


def _create_correction_plots_per_filename(
        filename: str, 
        measurements: Path, 
        correction_dirs: dict[str, Path], 
        x_colmap: ColumnsAndLabels, 
        y_colmap: ColumnsAndLabels, 
        ip_positions: str | dict[str, float] | Path, 
        opt: DotDict
    ):
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
        output_prefix=f"{file_label}_",  # used in the id, which is the fig_dict key
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


def save_plots(output_dir: Path, figure_dict: dict[str, Figure], input_dir: Path = None):
    """ Save the plots. """
    for figname, fig in figure_dict.items():
        outdir = output_dir
        figname_parts = figname.split(SPLIT_ID)
        if len(figname_parts) == 1:  # no SPLIT_ID
            # these are the combined plots. They have the column name at the end,
            # which we do not care for here at the moment.
            # In case of multiple columns per file, this could be brought back
            figname = "_".join([PREFIX] + figname.split("_")[:-1])
        else:
            # this is then the individual plots
            if input_dir:
                # files go directly into the correction-scenario folders
                outdir = input_dir / figname_parts[0]
                figname = f"{PREFIX}_{figname_parts[1]}"
            else:
                # everything goes into the output-dir (if given),
                # but with correction-name as additional prefix
                figname = "_".join([PREFIX] + figname_parts)

        output_path = get_full_output_path(outdir, figname)
        LOG.debug(f"Saving corrections plot to '{output_path}'")
        fig.savefig(output_path)
    
    if input_dir:
        LOG.info(f"Saved all correction plots in '{output_dir}'\n"
                 f"and into the correction-scenario in '{input_dir}'.")
    else:
        LOG.info(f"Saved all correction plots in '{output_dir}'.")


def show_plots(figure_dict: dict[str, Figure]):
    """Displays the provided figures.
    If `qtpy` is installed, they are shown in a single window.
    The individual corrections are sorted into vertical tabs,
    the optics parameter into horizontal tabs. 
    If `qtpy` is not installed, they are simply shown as individual figures.
    This is not recommended
    """
    try:
        window = VerticalTabWindow("Correction Check")
    except TypeError:
        log_no_qtpy_many_windows()
        for name, fig in figure_dict.items():
            create_pyplot_window_from_fig(fig, title=name.replace(SPLIT_ID, " "))
        plt.show()
        return

    rdt_pattern = re.compile(r"f\d{4}")
    rdt_complement = {
        RDT_REAL_COLUMN.text_label: RDT_IMAG_COLUMN,
        RDT_AMPLITUDE_COLUMN.text_label: RDT_PHASE_COLUMN,
    }

    figure_names = tuple(figure_dict.keys())
    correction_names = sorted(set([k.split(SPLIT_ID)[0] for k in figure_names if SPLIT_ID in k]))
    for correction_name in [None] + list(correction_names):
        if not correction_name:
            parameter_names = iter(sorted(k for k in figure_names if SPLIT_ID not in k))
            correction_tab_name = "All Corrections"
        else:
            parameter_names = iter(sorted(k for k in figure_names if correction_name in k))
            correction_tab_name = correction_name

        current_tab = TabWidget(title=correction_tab_name)
        window.add_tab(current_tab)

        for name_x in parameter_names:
            # extract the filename (and column-name in case of multi-correction-file)
            tab_prename = name_x.split(SPLIT_ID)[-1] 

            if rdt_pattern.match(tab_prename):
                # Handle RDTs: Get the rdt column and if it's amplitude or real, 
                # we look for the respective complement column (phase, imag).
                # Both, column and complement column are then added to the tab,
                # which is named after the rdt followed by either AP (amp/phase) or RI (real/imag)).
                rdt, column = tab_prename.split("_")[:2]
                try:
                    complement_column: ColumnsAndLabels = rdt_complement[column]
                except KeyError:
                    # skip phase and imag as they will become name_y for amp and real.
                    continue

                if not correction_name:
                    name_y = "_".join([rdt, complement_column.text_label, complement_column.expected_column])
                else:
                    name_y = "_".join(name_x.split("_")[:-1] + [complement_column.text_label,])

                tab_figs = (figure_dict[name_x], figure_dict[name_y])
                tab_name = f"{rdt} {column[0].upper()}/{complement_column.text_label[0].upper()}"

            else:
                # Handle non-RDT columns: As they are sorted alphabetically, the current column
                # is x and the following column is y. They are added to the tab, which 
                # is named by the optics parameter without plane.
                tab_name = " ".join(tab_prename.split("_")[:-1 if correction_name else -2])  # remove plane (and column-name)
                tab_figs = [figure_dict[name_x]]
                if not any(file_name in name_x for file_name in SINGLE_PLANE_FILES):
                    name_y = next(parameter_names)
                    tab_figs.append(figure_dict[name_y])

            current_tab.add_tab(PlotWidget(*tab_figs, title=tab_name))
    window.show()


def _get_corrected_measurement_names(correction_dirs: Iterable[Path]) -> set[str]:
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


if __name__ == "__main__":
    plot_checked_corrections()
