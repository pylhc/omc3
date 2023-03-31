"""
Correction Test
---------------

Run a test, i.e. a MAD-X simulation to check how well the correction settings work.
"""
from pathlib import Path
from typing import Dict, Sequence, Any

import pandas as pd

import tfs
from generic_parser import DotDict
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint
from omc3.correction import filters
from omc3.correction import handler as global_correction
from omc3.correction.constants import MODEL_MATCHED_FILENAME, COUPLING_NAME_TO_MODEL_COLUMN_SUFFIX
from omc3.correction.model_appenders import add_coupling_to_model
from omc3.correction.model_diff import diff_twiss_parameters
from omc3.correction.utils_check import get_plotting_style_parameters
from omc3.definitions.optics import OpticsMeasurement, FILE_COLUMN_MAPPING, ColumnsAndLabels, RDT_COLUMN_MAPPING
from omc3.global_correction import _get_default_values, CORRECTION_DEFAULTS, OPTICS_PARAMS_CHOICES
from omc3.model import manager
from omc3.model.accelerators.accelerator import Accelerator
from omc3.optics_measurements.constants import EXT, F1010_NAME, F1001_NAME, BETA, TUNE, F1001, F1010
from omc3.plotting.plot_correction_test import plot_correction_test
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config, glob_regex
from tfs import TfsDataFrame

LOG = logging_tools.get_logger(__name__)


def get_correction_test_params():
    params = EntryPointParameters()
    # IO ---
    params.add_parameter(name="meas_dir",
                         required=True,
                         type=PathOrStr,
                         help="Path to the directory containing the measurement files.",)
    params.add_parameter(name="output_dir",
                         required=True,
                         type=PathOrStr,
                         help=("Path to the directory where to write the output files. "
                              "If the input ``corrections`` input consists of multiple folders, their name will "
                               "be used to sort the output data into subfolders.", ))
    # Parameters (similar/same as in global correction) ---
    params.add_parameter(name="optics_params",
                         type=str,
                         nargs="+",
                         default=list(CORRECTION_DEFAULTS["optics_params"]),
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
    params.update(get_plotting_style_parameters())
    return params



@entrypoint(get_correction_test_params())
def correction_test_entrypoint(opt: DotDict, accel_opt) -> None:
    """ Test the given corrections.
    TODO: Instead of writing everything out, it could return
          dictionaries of the TFSDataFrames and Figures.
          But I don't see a usecase at the moment (jdilly 2023)
    """
    LOG.info("Starting Correction Test.")
    save_config(Path(opt.output_dir), {**opt, **accel_opt}, __file__)

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

    # plotting (maybe make optional, e.g. if run from the GUI?)
    _do_plots(corrections, opt)


# Input Parameters -------------------------------------------------------------

def _check_opt_add_dicts(opt: DotDict) -> DotDict:  # acts inplace...
    """ Check on options and put in missing values. """
    def_dict = _get_default_values()

    # Check cuts and fill defaults
    for key in ("modelcut", "errorcut"):
        if opt[key] is not None and len(opt[key]) != len(opt.optics_params):
            raise AttributeError(f"Length of {key} is not the same as of the optical parameters!")
        if opt[key] is None:
            given_keys = {}
        else:
            given_keys = dict(zip(opt.optics_params, opt[key]))
        opt[key] = {param: given_keys.get(param, def_dict[key][param]) for param in OPTICS_PARAMS_CHOICES}
    opt.optics_params = OPTICS_PARAMS_CHOICES


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

def _get_corrections(corrections: Sequence[Path], file_pattern: str = None) -> Dict[str, Sequence[Path]]:
    """ Sort the given correction files:
    If given by individual files, they all go into one bucket,
    if given by folders (i.e. scenarios) they are sorted by its name."""
    if corrections[0].is_file():  # checked above, that all are files or all are dirs
        return {"": corrections}
    return {c.name: glob_regex(c, file_pattern) for c in corrections}


# Main and Output --------------------------------------------------------------

def _get_measurement_filter(nominal_model: TfsDataFrame, opt: DotDict) -> Dict[str, pd.Index]:
    """ Writes the filtered original data out, so that it can be easily plotted against
    the corrections later. We now have the same BPMs and Column-Names. """
    if not opt.optics_params:
        LOG.debug("No filters selected, returning empty dict.")
        return {}


    optics_params, meas_dict = global_correction.get_measurement_data(
        opt.optics_params,
        opt.meas_dir,
        opt.beta_filename,
    )

    # add weights and use_errorbars as these are required in filters
    opt.use_errorbars = False
    opt.weights = {param: 1.0 for param in opt.optics_params}
    meas_dict = filters.filter_measurement(optics_params, meas_dict, nominal_model, opt)
    filter_dict = {
        global_correction.get_filename_from_parameter(k, opt.beta_filename): v.index for k, v in meas_dict.items()
    }

    LOG.debug(f"Filters selected for keys: {list(filter_dict.keys())}")
    return filter_dict


def _create_model_and_write_diff_to_measurements(
        output_dir: Path, measurement: OpticsMeasurement, correction_name: str, correction_files: Sequence[Path],
        accel_inst: Accelerator, rms_masks: Dict) -> OpticsMeasurement:
    """ Create a new model with the corrections (well, the "matchings") inserted and calculate
    the difference to the measurements. This will be written out then into individual tfs-files in the output
    folder(s). """
    if correction_name:
        output_dir = output_dir / correction_name
    LOG.info(f"Checking correction for {output_dir.name}")

    # Created matched model
    corr_model_path = output_dir / MODEL_MATCHED_FILENAME
    corr_model_elements = global_correction._create_corrected_model(corr_model_path, correction_files, accel_inst)  # writes out twiss file!
    corr_model_elements = _maybe_add_coupling_to_model(corr_model_elements, measurement)
    LOG.debug(f"Matched model created in {str(corr_model_path)}.")

    # Get diff to nominal model
    diff_columns = list(OPTICS_PARAMS_CHOICES[:-4]) + [col for col in corr_model_elements.columns if col.startswith("F1")]
    diff_models = diff_twiss_parameters(corr_model_elements, accel_inst.model, parameters=diff_columns)
    LOG.debug(f"Differences to nominal model calculated.")

     # Crate new "measurement" with additional columns
    output_measurement = OpticsMeasurement(directory=output_dir, allow_write=True)

    for attribute, filename in measurement.filenames(exist=True).items():
        filename = filename.replace(EXT, "")
        try:
            colmap = FILE_COLUMN_MAPPING[filename[:-1]]
        except KeyError:
            if filename not in (F1001_NAME, F1010_NAME):
                LOG.debug(f"Attribute {attribute} will not be checked.")
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
            )

    # df_rms_mask = rms_masks.get(filename, df.index)

    return output_measurement


def _create_check_columns(measurement: OpticsMeasurement, output_measurement: OpticsMeasurement, diff_models: TfsDataFrame,
                          colmap_meas: ColumnsAndLabels, colmap_model: ColumnsAndLabels, attribute: str):
    """
    Creates the columns in the measurements, that allow for checking the corrections.
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

    """
    LOG.debug(
        f"Creating columns for {attribute} ({colmap_meas.column}):\n"
        f"Model Diff: {colmap_model.delta_column} -> {colmap_meas.diff_correction_column}\n"
        f"Expected: {colmap_meas.delta_column} - diff ->  {colmap_meas.expected_column}\n"
        f"Error: {colmap_meas.error_delta_column} -> {colmap_meas.error_expected_column}"
    )
    df = measurement[attribute]

    diff = diff_models.loc[:, colmap_model.delta_column]

    df[colmap_meas.diff_correction_column] = diff
    df[colmap_meas.expected_column] = df[colmap_meas.delta_column] - diff
    df[colmap_meas.error_expected_column] = df[colmap_meas.error_delta_column]

    for tune_map in (FILE_COLUMN_MAPPING[TUNE].set_plane(n) for n in (1, 2)):
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
    found in the provided keys.

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

def _do_plots(corrections: Dict[str, Any], opt: DotDict):
    """ Plot the differences of the matched models to the measurement. """
    opt_plot = {k: v for k, v in opt.items() if k in get_plotting_style_parameters().keys()}
    opt_plot["input_dir"] = opt.output_dir
    opt_plot["output_dir"] = opt.output_dir
    opt_plot["corrections"] = list(corrections.keys())
    opt_plot["show"] = True  # debug
    plot_correction_test(**opt_plot)


if __name__ == '__main__':
    correction_test_entrypoint()