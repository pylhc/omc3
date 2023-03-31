"""
Plot Correction Test
--------------------

Create plots for the correction tests performed with `omc3.scripts.correction_test`.
"""
from pathlib import Path
from typing import Dict, List, Iterable, Set

from matplotlib.figure import Figure

from generic_parser import DotDict, EntryPointParameters, entrypoint
from omc3.correction.constants import (DIFF, ERROR, CORRECTED_LABEL, UNCORRECTED_LABEL, MODEL_MATCHED_FILENAME,
                                       MODEL_NOMINAL_FILENAME)
from omc3.correction.utils_check import get_plotting_style_parameters
from omc3.definitions.optics import FILE_COLUMN_MAPPING, ColumnsAndLabels, OpticsMeasurement
from omc3.optics_measurements.constants import EXT
from omc3.plotting.plot_optics_measurements import (_get_x_axis_column_and_label, _get_ip_positions)
from omc3.plotting.plot_tfs import plot as plot_tfs
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr

LOG = logging_tools.get_logger(__name__)


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


@entrypoint(get_plotting_params(), strict=True)
def plot_correction_test(opt: DotDict):
    """ Entrypoint for the plotting function. """
    correction_dirs: Dict[str, Path] = {}

    if len(opt.corrections) == 1 and not opt.corrections[0]:
        correction_dirs[CORRECTED_LABEL] = opt.input_dir
    else:
        for correction in opt.corrections:
            correction_dirs[correction] = opt.input_dir / correction

    measurements: Path = opt.meas_dir or list(correction_dirs.values())[0]

    fig_dict: Dict[Path, Figure] = {filename: None for filename in _get_corrected_measurement_names(correction_dirs.values())}


    ip_positions = _get_ip_positions(opt.ip_positions, opt.x_axis, opt.ip_search_pattern)
    x_axis = _get_x_axis_column_and_label(opt.x_axis)

    for tfs_file in fig_dict.keys():
        if tfs_file.startswith("f1"):
            continue

        cal: ColumnsAndLabels = FILE_COLUMN_MAPPING[tfs_file[:-1]].set_plane(tfs_file[-1].upper())

        fig_dict[tfs_file] = plot_tfs(
            files=[path / f"{tfs_file}{EXT}" for path in correction_dirs.values()],
            file_labels=list(correction_dirs.keys()),  # defines the legend
            y_columns=[cal.expected_column],
            column_labels=[cal.delta_label],  # defines y-axis label
            error_columns=[cal.error_expected_column],
            x_columns=[x_axis.column],
            x_labels=[x_axis.label],
            vertical_lines=ip_positions + opt.lines_manual,
            same_axes=["files"],
            output_prefix=f"plot_{tfs_file}_",
            output=opt.output_dir,
            **opt.get_subdict(['show',
                               'plot_styles', 'manual_style',
                               'change_marker', 'errorbar_alpha',
                               'ncol_legend', 'x_lim', 'y_lim',
                               'share_xaxis'])
        )

    return fig_dict

def _get_corrected_measurement_names(correction_dirs: Iterable[Path]) -> Set[str]:
    tfs_files = None
    for idx, correction in enumerate(correction_dirs):
        new_files = set(f.stem for f in correction.glob(f"*{EXT}"))
        if not idx:
            tfs_files = new_files
            continue

        tfs_files &= new_files
    tfs_files -= {Path(MODEL_MATCHED_FILENAME).stem, Path(MODEL_NOMINAL_FILENAME).stem}
    return tfs_files



