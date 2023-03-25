"""
Correction Test
---------------

Run a test, i.e. a MAD-X simulation to check how well the correction settings work.
"""
from pathlib import Path
from typing import Dict, Sequence, Iterable, Any

import pandas as pd

import tfs
from generic_parser import DotDict
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint
from omc3.correction import filters, model_appenders
from omc3.correction.constants import (BETA, TUNE, DIFF, NAME, NOMINAL_MEASUREMENT, S, MDL, PHASE_ADV, WEIGHT,
                                       MODEL_MATCHED_FILENAME)
from omc3.correction.correction_test_utils import get_plotting_style_parameters, Measurements
from omc3.correction.handler import (get_measurement_data, _create_corrected_model,
                                     _maybe_add_coupling_to_model, get_filename_from_parameter)
from omc3.definitions.constants import PLANES
from omc3.global_correction import _get_default_values, CORRECTION_DEFAULTS, OPTICS_PARAMS_CHOICES
from omc3.model import manager
from omc3.model.accelerators.accelerator import Accelerator
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
                              "If the input consists of multiple folders, their name will "
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
    params.add_parameter(name="beta_file_name",
                         default=CORRECTION_DEFAULTS["beta_file_name"],
                         help="Prefix of the beta file to use. E.g.: getkmodbeta", )
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

    # read data
    optics_params, meas_dict = get_measurement_data(
        opt.optics_params,
        opt.meas_dir,
        opt.beta_file_name,
    )

    # get model and filter data
    accel_inst = manager.get_accelerator(accel_opt)
    accel_inst.model = _maybe_add_coupling_to_model(accel_inst.model, optics_params)
    meas_dict = _filter_and_write_original_measurements(meas_dict, optics_params, accel_inst.model, opt)

    # sort corrections input (e.g. by folders)
    corrections = _get_corrections(opt.corrections, opt.file_pattern)

    # run through all scenarios
    for correction_name, correction_files in corrections.items():
        _create_model_and_write_diff_to_measurements(
            opt.output_dir,
            meas_dict,
            correction_name,
            correction_files,
            accel_inst,
            opt.beta_file_name
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

    # add weights and use_errorbars as used in filters (but we don't care here in the tests)
    opt.use_errorbars = False
    opt.weights = {param: 1.0 for param in opt.optics_params}

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

def _filter_and_write_original_measurements(
        meas_dict: Measurements, optics_params: Iterable[str], nominal_model: TfsDataFrame, opt: DotDict
) -> Measurements:
    """ Writes the filtered original data out, so that it can be easily plotted against
    the corrections later. We now have the same BPMs and Column-Names. """
    meas_dict = filters.filter_measurement(optics_params, meas_dict, nominal_model, opt)
    meas_dict = model_appenders.add_differences_to_model_to_measurements(nominal_model, meas_dict)
    _write_corrected_measurement_data(opt.output_dir / NOMINAL_MEASUREMENT, meas_dict, opt.beta_file_name, nominal_model)
    return meas_dict


def _create_model_and_write_diff_to_measurements(
        output_dir: Path, meas_dict: Measurements, correction_name: str, correction_files: Sequence[Path],
        accel_inst: Accelerator, beta_file_name: str) -> Measurements:
    """ Create a new model with the corrections (well, the "matchings") inserted and calculate
    the difference to the measurements. This will be written out then into individual tfs-files in the output
    folder(s). """
    if correction_name:
        output_dir = output_dir / correction_name

    corr_model_path = output_dir / MODEL_MATCHED_FILENAME
    corr_model_elements = _create_corrected_model(corr_model_path, correction_files, accel_inst)  # also writes out twiss file!
    corr_model_elements = _maybe_add_coupling_to_model(corr_model_elements, accel_inst.model.columns)
    corr_meas_dict = model_appenders.add_differences_to_model_to_measurements(corr_model_elements, meas_dict)

    _write_corrected_measurement_data(output_dir, corr_meas_dict, beta_file_name, accel_inst.model)
    return corr_meas_dict


def _write_corrected_measurement_data(output_dir: Path, meas_dict: Measurements, beta_file_name: str, nominal_model: pd.DataFrame):
    """ Write the measurement data out.
    These files look a bit different from the optics measurement outputs,
    as they still use the VALUE, ERROR, DIFF etc. column names. """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update Tunes in all files
    tune_headers = {
        f"{DIFF}{TUNE}1": meas_dict[TUNE].loc[f"{TUNE}1", DIFF],
        f"{DIFF}{TUNE}2": meas_dict[TUNE].loc[f"{TUNE}2", DIFF],
    }

    for key, data in meas_dict.items():
        if key == f"{TUNE}":
            continue
        file_name = get_filename_from_parameter(key, beta_file_name)

        # insert DIFF-Tunes
        data.headers.update(tune_headers)

        # insert S and MU data (for x-axis of plotting)
        data[S] = nominal_model.loc[data.index, S]
        for plane in PLANES:
            data[f"{PHASE_ADV}{plane.upper()}{MDL}"] = nominal_model.loc[data.index, f"{PHASE_ADV}{plane.upper()}"]

        # drop weights (not used for tests and possibly confusing)
        data = data.drop(columns=[WEIGHT])

        tfs.write(output_dir / file_name, data, save_index=NAME)


# Plotting ---------------------------------------------------------------------

def _do_plots(corrections: Dict[str, Any], opt: DotDict):
    """ Plot the differences of the matched models to the measurement. """
    opt_plot = {k: v for k, v in opt.items() if k in get_plotting_style_parameters().keys()}
    opt_plot["input_dir"] = opt.output_dir
    opt_plot["output_dir"] = opt.output_dir
    opt_plot["corrections"] = list(corrections.keys())
    plot_correction_test(opt_plot)


if __name__ == '__main__':
    correction_test_entrypoint()