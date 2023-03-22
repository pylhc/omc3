"""
Correction Test
---------------

Run a test, i.e. a MAD-X simulation to check how well the correction settings work.
"""
from pathlib import Path
from typing import Dict, Sequence

import tfs
from generic_parser import DotDict
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint

from omc3.correction import handler, filters, model_appenders
from omc3.correction.constants import (BETA, BETABEAT, DISP, F1001, F1010,
                                       NORM_DISP, PHASE, TUNE, DIFF, NAME)
from omc3.correction.handler import get_measurement_data, _create_corrected_model, _maybe_add_coupling_to_model, \
    get_filename_from_parameter
from omc3.global_correction import _get_default_values, CORRECTION_DEFAULTS, OPTICS_PARAMS_CHOICES
from omc3.model import manager
from omc3.model.accelerators.accelerator import Accelerator
from omc3.optics_measurements.constants import EXT
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config, glob_regex
from tfs import TfsDataFrame

LOG = logging_tools.get_logger(__name__)

MATCHED_MODEL_NAME = f"twiss_matched{EXT}"


def correction_test_params():
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
    # Parameters ---
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
    # Plotting ---
    params.add_parameter(name="show_plots",
                         help="Show plots.",
                         action="store_true",
                         )
    params.add_parameter(name="change_marker",
                         help="Changes marker for each line in the plot.",
                         action="store_true",
                         )
    params.add_parameter(name="auto_scale",
                         help="Scales the plot, so that this percentage of points is inside the picture.",
                         type=float,
                         )
    return params


@entrypoint(correction_test_params())
def correction_test_entrypoint(opt: DotDict, accel_opt) -> None:
    """ Test the given corrections. """
    LOG.info("Starting Correction Test.")
    save_config(Path(opt.output_dir), {**opt, **accel_opt}, __file__)

    opt = _check_opt_add_dicts(opt)
    opt.output_dir.mkdir(parents=True, exist_ok=True)
    accel_inst = manager.get_accelerator(accel_opt)

    # read data from files
    optics_params, meas_dict = get_measurement_data(
        opt.optics_params,
        opt.meas_dir,
        opt.beta_file_name,
    )

    # get nominal model (for filtering)
    nominal_model = _maybe_add_coupling_to_model(accel_inst.model, optics_params)
    meas_dict = filters.filter_measurement(optics_params, meas_dict, nominal_model, opt)

    # sort the given correction files, either all files in one scenario or by folder
    corrections = _get_corrections(opt.corrections, opt.file_pattern)
    corrected_measurements = {"nominal": meas_dict}

    # loop over different correction scenarios and create a new corrected/matched
    # model with the given changed parameters. Then compare the new model with
    # the measurement and calculate differences. These are written out into the
    # scenario folders, so that also the GUI can plot them.
    for correction_name, correction_files in corrections.items():
        output_dir = opt.output_dir
        if correction_name:
            output_dir = output_dir / correction_name

        corr_model_path = output_dir / MATCHED_MODEL_NAME
        corr_model_elements = _create_corrected_model(corr_model_path, correction_files, accel_inst)  # writes twiss files!
        corr_model_elements = _maybe_add_coupling_to_model(corr_model_elements, nominal_model.columns)
        corr_meas_dict = model_appenders.add_differences_to_model_to_measurements(corr_model_elements, meas_dict)
        corrected_measurements[correction_name] = corr_meas_dict

        _write_corrected_measurement_data(output_dir, corr_meas_dict, opt.beta_file_name)

    # TODO plot corrections


def _get_corrections(corrections: Sequence[Path], file_pattern: str = None) -> Dict[str, Sequence[Path]]:
    """ Get the correction files and correction names """
    if corrections[0].is_file():
        return {"": corrections}
    return {c.name: glob_regex(c, file_pattern) for c in corrections}


def _write_corrected_measurement_data(output_dir: Path, meas_dict: Dict[str, TfsDataFrame], beta_file_name: str):
    # Update Tunes in all files
    tune_headers = {
            f"{DIFF}{TUNE}1": meas_dict[TUNE].loc[f"{TUNE}1", DIFF],
            f"{DIFF}{TUNE}2": meas_dict[TUNE].loc[f"{TUNE}2", DIFF],
        }

    for key, data in meas_dict.items():
        if key == f"{TUNE}":
            continue
        data.headers.update(tune_headers)
        file_name = get_filename_from_parameter(key, beta_file_name)
        tfs.write(output_dir / file_name, data, save_index=NAME)


def _check_opt_add_dicts(opt: dict) -> dict:  # acts inplace...
    """ Check on options and put in missing values """
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
