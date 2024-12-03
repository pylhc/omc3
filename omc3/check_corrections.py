"""
Correction Test
---------------

In a MAD-X simulation (pre-calculated) corrections are applied to the (nominal)
model and the difference between this new matched model and the nominal model
are evaluated. These are the changes in the optics parameters as
expected from the corrections and are later subtracted from the original
model-measurement DELTA (see below).
In new folders, either the given output folder or sub-folders,
based on the name of the correction applied, the measurement data will be
written out into the standard optics tfs-files with three additional columns:

DIFF-column:
               Difference between the corrected and uncorrected model,
               i.e. the expected correction influence (with inverted sign).
               The beta-diff is beta-beating.
               The difference normalized-dispersion is also properly calculated.
EXP-column :
               This column is calculated by subtracting the DIFF-column
               from the original DELTA-column (i.e. the difference between
               measurement and model) and is therefore the
               expected difference of model and measurement (i.e.
               DELTA-Measurement) after correction.
               For beta-beating this might be only approximate
               as it assumes, that the model used to calculate
               the DELTA-columns in the measurement and the
               nominal model used for the corrections are identical
               (if e.g. best-knowledge model is used, the
               expectation is only approximate).
ERREXP-column:
               This is the error on the EXP-column.
               This is the measurement error in most cases,
               apart from in beta and normalized dispersion,
               where we account for beating and normalization.

There will also be the RMS values of the DIFF and EXP columns
in the headers of the files.
If error- and/or measurement cuts were given, additional RMS headers
will be present, taking these cuts into account.
This is the only use for the given cuts and optics parameters.

If plotting is activated, also plots for each correction (DIFFerence and EXPected) as
well as a plot for all corrections (only EXPected) will be saved into the output folder(s).


**Arguments:**

*--Required--*

- **corrections** *(PathOrStr)*:

    Paths to the correction files/directories to use. If files are given,
    these will all be applied as corrections at the same time. If folders
    are given, these are assumed to individually containing the correction
    files. See then also ``file_pattern``.


- **meas_dir** *(PathOrStr)*:

    Path to the directory containing the measurement files.


- **output_dir** *(PathOrStr)*:

    Path to the directory where to write the output files. If the
    ``corrections`` input consists of multiple folders, their name will be
    used to sort the output data into subfolders.


*--Optional--*

- **beta_filename**:

    Prefix of the beta file to use. E.g.: ``beta_phase_``

    default: ``beta_phase_``


- **change_marker**:

    Changes marker for each line in the plot.

    action: ``store_true``


- **combine_by**:

    Combine plots into one. Either files, planes (not separated into two
    axes) or both.

    choices: ``['files', 'planes']``


- **errorbar_alpha** *(float)*:

    Alpha value for error bars

    default: ``0.6``


- **errorcut** *(float)*:

    Reject BPMs whose error bar is higher than the corresponding input.
    Input in order of optics_params.


- **file_pattern** *(str)*:

    Filepattern to use to find correction files in folders (as regex).

    default: ``^changeparameters*?\\.madx$``


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


- **modelcut** *(float)*:

    Reject BPMs whose deviation to the model is higher than the
    corresponding input. Input in order of optics_params.


- **ncol_legend** *(int)*:

    Number of bpm legend-columns. If < 1 no legend is shown.

    default: ``3``


- **optics_params** *(str)*:

    List of parameters for which the cuts should be applied (e.g. BETX
    BETY)

    choices: ``('PHASEX', 'PHASEY', 'BETX', 'BETY', 'NDX', 'Q', 'DX', 'DY', 'F1001R', 'F1001I', 'F1010R', 'F1010I')``


- **plot**:

    Activate plotting.

    action: ``store_true``


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

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

import tfs
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint
from omc3.correction import filters
from omc3.correction import handler as global_correction
from omc3.correction.constants import MODEL_MATCHED_FILENAME, COUPLING_NAME_TO_MODEL_COLUMN_SUFFIX
from omc3.correction.model_appenders import add_coupling_to_model
from omc3.correction.model_diff import diff_twiss_parameters
from omc3.correction.response_twiss import PLANES
from omc3.definitions.optics import (
    OpticsMeasurement, ColumnsAndLabels,
    FILE_COLUMN_MAPPING, RDT_COLUMN_MAPPING, TUNE_COLUMN
)
from omc3.global_correction import _get_default_values, CORRECTION_DEFAULTS, OPTICS_PARAMS_CHOICES
from omc3.model import manager
from omc3.model.accelerators.accelerator import Accelerator
from omc3.optics_measurements.constants import EXT, F1010_NAME, F1001_NAME, BETA, F1001, F1010, PHASE, TUNE
from omc3.optics_measurements.toolbox import ang_diff
from omc3.plotting.plot_checked_corrections import plot_checked_corrections, get_plotting_style_parameters
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, glob_regex, save_config
from omc3.utils.stats import rms, circular_rms
from tfs import TfsDataFrame

if TYPE_CHECKING:
    from collections.abc import Sequence
    from generic_parser import DotDict


LOG = logging_tools.get_logger(__name__)


def get_params():
    params = EntryPointParameters()
    # IO ---
    params.add_parameter(name="meas_dir",
                         required=True,
                         type=PathOrStr,
                         help="Path to the directory containing the measurement files.",)
    params.add_parameter(name="output_dir",
                         required=True,
                         type=PathOrStr,
                         help="Path to the directory where to write the output files. "
                              "If the ``corrections`` input consists of multiple folders, their name will "
                               "be used to sort the output data into subfolders.",)
    params.add_parameter(name="corrections",
                         required=True,
                         nargs="+",
                         type=PathOrStr,
                         help="Paths to the correction files/directories to use. "
                              "If files are given, these will all be applied as corrections at the same time. "
                              "If folders are given, these are assumed to individually containing "
                              "the correction files. See then also ``file_pattern``.",)
    params.add_parameter(name="file_pattern",
                         help="Filepattern to use to find correction files in folders (as regex).",
                         type=str,
                         default=r"^changeparameters*?\.madx$",
                         )
    # Parameters (similar/same as in global correction) ---
    params.add_parameter(name="optics_params",
                         type=str,
                         nargs="+",
                         choices=OPTICS_PARAMS_CHOICES,
                         help=f"List of parameters for which the cuts should be applied (e.g. {BETA}X {BETA}Y)", )
    params.add_parameter(name="modelcut",
                         nargs="+",
                         type=float,
                         help="Reject BPMs whose deviation to the model is higher "
                              "than the corresponding input. Input in order of optics_params.",)
    params.add_parameter(name="errorcut",
                         nargs="+",
                         type=float,
                         help="Reject BPMs whose error bar is higher than the corresponding "
                              "input. Input in order of optics_params.",)
    params.add_parameter(name="beta_filename",
                         default=CORRECTION_DEFAULTS["beta_filename"],
                         help="Prefix of the beta file to use. E.g.: ``beta_phase_``", )
    # Plotting -----------------------------------------------------------------
    params.add_parameter(name="plot",
                         action="store_true",
                         help="Activate plotting."
                         )
    params.update(get_plotting_style_parameters())
    return params


@entrypoint(get_params())
def correction_test_entrypoint(opt: DotDict, accel_opt) -> None:
    """ Entrypoint function to test the given corrections.

    .. todo:: Instead of writing everything out, it could return
             dictionaries of the TFSDataFrames and Figures.
             But I don't see a usecase at the moment (jdilly 2023)
    """
    LOG.info("Starting Correction Test.")
    save_config(Path(opt.output_dir), opt, __file__, unknown_opt=accel_opt)

    opt = _check_opt_add_dicts(opt)
    opt.output_dir.mkdir(parents=True, exist_ok=True)

    measurement = OpticsMeasurement(opt.meas_dir)

    # get model and filter data
    accel_inst = manager.get_accelerator(accel_opt)
    accel_inst.model = _maybe_add_coupling_to_model(accel_inst.model, measurement)

    meas_masks = _get_measurement_filter(
        accel_inst.model,
        opt.get_subdict(["meas_dir", "optics_params", "modelcut", "errorcut", "beta_filename"])
    )

    # sort corrections input (e.g. by folders)
    corrections = _get_corrections(opt.corrections, opt.file_pattern)

    # run through all scenarios
    for correction_name, correction_files in corrections.items():
        _create_model_and_write_diff_to_measurements(
            opt.output_dir,
            measurement,
            correction_name,
            correction_files,
            accel_inst,
            meas_masks,
        )

    if opt.plot:
        _do_plots(corrections, opt)


# Input Parameters -------------------------------------------------------------

def _check_opt_add_dicts(opt: DotDict) -> DotDict:
    """ Check on options and put in missing values. """
    opt = copy.deepcopy(opt)  # not sure if I trust this (jdilly)
    def_dict = _get_default_values()

    # Check cuts and fill defaults
    if opt.optics_params:
        for key in ("modelcut", "errorcut"):
            if opt[key] is not None and len(opt[key]) != len(opt.optics_params):
                raise AttributeError(f"Length of {key} is not the same as of the optical parameters!")
            if opt[key] is not None:
                opt[key] = dict(zip(opt.optics_params, opt[key]))
            else:
                opt[key] = {param: def_dict[key][param] for param in opt.optics_params}

    # Convert Strings to Paths
    opt.meas_dir = Path(opt.meas_dir)
    opt.output_dir = Path(opt.output_dir)
    opt.corrections = [Path(c) for c in opt.corrections]
    all_files = all(c.is_file() for c in opt.corrections)
    all_dirs = all(c.is_dir() for c in opt.corrections)
    if not (all_files or all_dirs):
        raise AttributeError("Corrections need to be either all paths to files or all paths to directories!")

    if all_dirs and not opt.file_pattern:
        raise AttributeError("Parameter 'file_pattern' is missing, when giving directories as input.")

    return opt


def _get_corrections(corrections: Sequence[Path], file_pattern: str = None) -> dict[str, Sequence[Path]]:
    """ Sort the given correction files:
    If given by individual files, they all go into one bucket,
    if given by folders (i.e. scenarios) they are sorted by its name.
    It is also checked, that they are valid!"""
    # create correction mapping
    if corrections[0].is_file():  # checked above, that all are files or all are dirs
        corr_dict = {"": corrections}
    else:
        corr_dict = {c.name:  _glob_regex_paths(c, file_pattern) for c in corrections}

    # check correction files
    for name, corr_files in corr_dict.items():
        if not len(corr_files):
            raise IOError(f"No corrections found for scenario {name}.")

        do_not_exist = [f for f in corr_files if not f.exists()]
        if len(do_not_exist):
            raise IOError(f"Some correction files do not exist for scenario {name}:"
                          f" {str(do_not_exist)}")

    return corr_dict


def _glob_regex_paths(path: Path, pattern: str) -> list[Path]:
    """ Filter the files in path by pattern and return a list of paths. """
    return [path / f for f in glob_regex(path, pattern)]


# Main and Output --------------------------------------------------------------

def _get_measurement_filter(nominal_model: TfsDataFrame, opt: DotDict) -> dict[str, pd.Index]:
    """ Get the filtered measurement based on the cuts as done in the correction calculation.
    As we need this only for RMS calculations later on, we only care about the
    BPM-names. So the returned dict contains the index to be used for this
    calculation per optics measurement file."""
    if not opt.optics_params:
        LOG.debug("No filters selected, returning empty dict.")
        return {}

    if TUNE in opt.optics_params:
        LOG.warning("Filtering RMS on tune does not make sense. Ignoring.")
        opt.optics_params.remove(TUNE)

    optics_params, meas_dict = global_correction.get_measurement_data(
        opt.optics_params,
        opt.meas_dir,
        opt.beta_filename,
    )

    # add weights and use_errorbars as these are required in filters
    opt.use_errorbars = False
    opt.weights = {param: 1.0 for param in opt.optics_params}
    meas_dict = filters.filter_measurement(optics_params, meas_dict, nominal_model, opt)

    # use the indices as filters
    filter_dict = {
        global_correction.get_filename_from_parameter(k, opt.beta_filename): v.index for k, v in meas_dict.items()
    }

    LOG.debug(f"Filters selected for keys: {list(filter_dict.keys())}")
    return filter_dict


def _create_model_and_write_diff_to_measurements(
        output_dir: Path, measurement: OpticsMeasurement, correction_name: str, correction_files: Sequence[Path],
        accel_inst: Accelerator, rms_masks: dict) -> OpticsMeasurement:
    """ Create a new model with the corrections (well, the "matchings") applied and calculate
    the difference to the nominal model, i.e. the expected improvement of the measurements
    (for detail see main docstring in this file).
    This will be written out then into individual tfs-files in the output folder(s). """
    if correction_name:
        output_dir = output_dir / correction_name
    LOG.info(f"Checking correction for {output_dir.name}")

    # Created matched model
    corr_model_path = output_dir / MODEL_MATCHED_FILENAME
    corr_model_elements = global_correction._create_corrected_model(corr_model_path, correction_files, accel_inst)  # writes out twiss file!
    corr_model_elements = _maybe_add_coupling_to_model(corr_model_elements, measurement)
    LOG.debug(f"Matched model created in {str(corr_model_path.absolute())}.")

    # Get diff to nominal model
    diff_columns = (
            list(OPTICS_PARAMS_CHOICES[:-4]) +
            [col for col in corr_model_elements.columns if col.startswith("F1")] +
            list(PLANES)
    )
    diff_models = diff_twiss_parameters(corr_model_elements, accel_inst.model, parameters=diff_columns)
    LOG.debug("Differences to nominal model calculated.")

     # Create new "measurement" with additional columns
    output_measurement = OpticsMeasurement(directory=output_dir, allow_write=True)

    for attribute, filename in measurement.filenames(exist=True).items():
        rms_mask = rms_masks.get(filename, None)  # keys in rms_masks still with extension
        filename = filename.replace(EXT, "")

        try:
            colmap = FILE_COLUMN_MAPPING[filename[:-1]]
        except KeyError:
            if filename not in (F1001_NAME, F1010_NAME):
                LOG.debug(f"Attribute {attribute} will not be checked.")
                continue
            # Coupling RDTS:
            # get F1### column map without the I, R, A, P part based on the rdt-filename:
            LOG.debug(f"Checking coupling correction for {attribute}")
            colmap_model_generic = ColumnsAndLabels(COUPLING_NAME_TO_MODEL_COLUMN_SUFFIX[filename])
            for idx, colmap_meas in enumerate(RDT_COLUMN_MAPPING.values()):  # AMP, PHASE, REAL or IMAG as column-map
                colmap_model = colmap_model_generic.set_plane(colmap_meas.column[0])  # F1### with I, R, A, P
                output_measurement.allow_write = (idx == 3)  # write out only after the last column is set
                _create_check_columns(
                    measurement=measurement,
                    output_measurement=output_measurement,
                    diff_models=diff_models,
                    colmap_meas=colmap_meas,
                    colmap_model=colmap_model,
                    attribute=attribute,
                    rms_mask=rms_mask,
                )
        else:
            # "Normal" optics parameters
            LOG.debug(f"Checking correction for {attribute}")
            plane = filename[-1].upper()
            cols = colmap.set_plane(plane)
            _create_check_columns(
                measurement=measurement,
                output_measurement=output_measurement,
                diff_models=diff_models,
                colmap_meas=cols,
                colmap_model=cols,
                attribute=attribute,
                rms_mask=rms_mask,
            )
    return output_measurement


def _create_check_columns(measurement: OpticsMeasurement, output_measurement: OpticsMeasurement, diff_models: TfsDataFrame,
                          colmap_meas: ColumnsAndLabels, colmap_model: ColumnsAndLabels, attribute: str,
                          rms_mask: dict = None) -> None:
    """Creates the columns in the measurements, that allow for checking the corrections.
    These are:
        diff_correction_column: Difference between the corrected and uncorrected model,
                                i.e. the expected correction influence (with inverted sign).
                                This is calculated beforehand (see `omc3.correction.model_appenders`)
                                and given via `diff_models`.
                                The beta-diff is beta-beating, and the difference normalized-dispersion
                                is also properly calculated.
        expected_column: The expected difference of model and measurement (i.e.
                         DELTA-Measurement) after correction.
                         For beta-beating this might be only approximate
                         as it assumes, that the model used to calculate
                         the DELTA-columns in the measurement and the
                         nominal model used for the corrections are identical.
                         (If e.g. best-knowledge model is used, the
                         expectation is only approximate)
        error-column: As error column later the ERRDELTA column (which
                      is basically the normal error apart from in beta and
                      normalized dispersion, where this is calculated to account
                      for beating and normalization) is used. Nothing done here.

    Args:
        measurement (OpticsMeasurement): The original optics measurement
        output_measurement (OpticsMeasurement): The OM-object gathering the output dataframes
        diff_models (TfsDataframe): Diff-Dataframe between the matched and nominal model
        colmap_meas (ColumnsAndLabels): Columns of the measurement
        colmap_model (ColumnsAndLabels): Columns of the model
        attribute (str): attribute/property name of the OpticsMeasurement of the current measurement
        rms_mask (pd.Index): Indices to use when calculating RMS

    """
    LOG.debug(
        f"Creating columns for {attribute} ({colmap_meas.column}):\n"
        f"Model Diff: {colmap_model.delta_column} -> {colmap_meas.diff_correction_column}\n"
        f"Expected: {colmap_meas.delta_column} - diff ->  {colmap_meas.expected_column}\n"
        f"Error: {colmap_meas.error_delta_column} -> {colmap_meas.error_expected_column}"
    )
    df = measurement[attribute]

    diff = diff_models.loc[df.index, colmap_model.delta_column]

    df[colmap_meas.diff_correction_column] = diff
    if colmap_meas.column == PHASE:
        df[colmap_meas.expected_column] = pd.to_numeric(ang_diff(df[colmap_meas.delta_column], diff))  # assumes period 1
        df.headers[colmap_meas.delta_rms_header] =  circular_rms(df[colmap_meas.delta_column], period=1)
        df.headers[colmap_meas.expected_rms_header] = circular_rms(df[colmap_meas.expected_column], period=1)
        if rms_mask is not None:
            df.headers[colmap_meas.delta_masked_rms_header] = circular_rms(df.loc[rms_mask, colmap_meas.delta_column], period=1)
            df.headers[colmap_meas.expected_masked_rms_header] = circular_rms(df.loc[rms_mask, colmap_meas.expected_column], period=1)

    else:
        df[colmap_meas.expected_column] = pd.to_numeric(df[colmap_meas.delta_column] - diff)
        df.headers[colmap_meas.delta_rms_header] = rms(df[colmap_meas.delta_column])
        df.headers[colmap_meas.expected_rms_header] = rms(df[colmap_meas.expected_column])
        if rms_mask is not None:
            df.headers[colmap_meas.delta_masked_rms_header] = rms(df.loc[rms_mask, colmap_meas.delta_column])
            df.headers[colmap_meas.expected_masked_rms_header] = rms(df.loc[rms_mask, colmap_meas.expected_column])

    LOG.info(
        f"\nRMS {attribute} ({colmap_meas.column}):\n"
        f"    measured {df.headers[colmap_meas.delta_rms_header]:.2e}\n"
        f"    expected {df.headers[colmap_meas.expected_rms_header]:.2e}"
    )

    if rms_mask is not None:
        LOG.info(
            f"\nRMS {attribute} ({colmap_meas.column}) after filtering:\n"
            f"    measured {df.headers[colmap_meas.delta_masked_rms_header]:.2e}\n"
            f"    expected {df.headers[colmap_meas.expected_masked_rms_header]:.2e}"
        )

    try:
        df[colmap_meas.error_expected_column] = df[colmap_meas.error_delta_column]
    except KeyError:
        LOG.warning(f"The error-delta column {colmap_meas.error_delta_column} was "
                    f"not found in {attribute}. Probably an old file. Assuming zero errors.")
        df[colmap_meas.error_expected_column] = 0.0

    for tune_map in (TUNE_COLUMN.set_plane(n) for n in (1, 2)):
        LOG.debug(
            f"Creating columns for tune {tune_map.column}:\n"
            f"Model Diff: {tune_map.delta_column} -> {tune_map.diff_correction_column}\n"
            f"Expected: {tune_map.column} - diff ->  {tune_map.expected_column}"
        )

        diff_tune = diff_models.headers[tune_map.delta_column]
        df.headers[tune_map.diff_correction_column] = diff_tune
        df.headers[tune_map.expected_column] = df.headers[tune_map.column] - diff_tune

    output_measurement[attribute] = df  # writes file if allow_write is set.


# Helper -----------------------------------------------------------------------


def _maybe_add_coupling_to_model(model: tfs.TfsDataFrame, measurement: OpticsMeasurement) -> tfs.TfsDataFrame:
    """ Add coupling to the model, if terms corresponding to coupling RDTs are
    found in the provided measurements.

    Args:
        model (tfs.TfsDataFrame): Twiss dataframe.
        keys (Sequence[str]):

    Returns:
        A TfsDataFrame with the added columns.
    """
    if any((measurement.directory / measurement.filenames[rdt.lower()]).exists() for rdt in (F1001, F1010)):
        return add_coupling_to_model(model)
    return model



# Plotting ---------------------------------------------------------------------

def _do_plots(corrections: dict[str, Any], opt: DotDict):
    """ Plot the differences of the matched models to the measurement. """
    opt_plot = {k: v for k, v in opt.items() if k in get_plotting_style_parameters().keys()}
    opt_plot["input_dir"] = opt.output_dir
    opt_plot["output_dir"] = opt.output_dir
    opt_plot["corrections"] = list(corrections.keys())
    plot_checked_corrections(**opt_plot)


if __name__ == '__main__':
    correction_test_entrypoint()