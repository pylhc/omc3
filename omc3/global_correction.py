r"""
Global Correction
-----------------

Iterative Correction Scheme.

The response matrices :math:`R_{O}` for the observables :math:`O` (e.g. BBX, PHASEX, ...)
are loaded from a file and then the equation

.. math:: R_{O} \cdot \delta var = O_{meas} - O_{model}
    :label: eq1

is being solved for :math:`\delta var` via a chosen method (at the moment only numpys pinv,
which creates a pseudo-inverse via svd is used).

The response matrices are hereby merged into one matrix for all observables to solve for all
:math:`\delta var` at the same time.

To normalize the observables to another ``weights`` (W) can be applied.

Furthermore, an ``errorcut``, specifying the maximum errorbar for a BPM to be used, and
``modelcut``, specifying the maximum distance between measurement and model for a BPM to be used,
can be defined. Data from BPMs outside of those cut-values will be discarded.
These cuts are defined for each observable separately.

After each iteration the model variables are changed by :math:`-\delta var` and the
observables are recalculated by Mad-X.
:eq:`eq1` is then solved again.

Input arguments are split into correction arguments and accelerator arguments.
The former are listed below, the latter depend on the accelerator you want
to use. Check :ref:`modules/model:Model` to see which ones are needed.

**Arguments:**

*--Required--*

- **meas_dir**:

    Path to the directory containing the measurement files.


- **output_dir**:

    Path to the directory where to write the output files.


*--Optional--*

- **beta_filename**:

    Prefix of the beta file to use. E.g.: getkmodbeta

    default: ``beta_phase_``


- **errorcut** *(float)*:

    Reject BPMs whose error bar is higher than the corresponding input.
    Input in order of optics_params.


- **fullresponse_path**:

    Path to the fullresponse binary file. If not given, calculates the
    response analytically.


- **max_iter** *(int)*:

    Maximum number of correction re-iterations to perform. A value of `0`
    means the correction is calculated once.

    default: ``3``


- **method** *(str)*:

    Optimization method to use.

    choices: ``('pinv', 'omp')``

    default: ``pinv``


- **min_corrector_strength** *(float)*:

    Minimum (absolute) strength of correctors.

    default: ``0.0``


- **modelcut** *(float)*:

    Reject BPMs whose deviation to the model is higher than the
    corresponding input. Input in order of optics_params.


- **n_correctors** *(int)*:

    Maximum number of correctors to use. (Method: 'omp')


- **optics_params** *(str)*:

    List of parameters to correct upon (e.g. BETX BETY)

    choices: ``('PHASEX', 'PHASEY', 'BETX', 'BETY', 'DX', 'DY', 'NDX', 'Q', 'F1001R', 'F1001I', 'F1010R', 'F1010I')``

    default: ``['PHASEX', 'PHASEY', 'BETX', 'BETY', 'NDX', 'Q']``


- **output_filename**:

    Identifier of the output files.

    default: ``changeparameters_iter``


- **svd_cut** *(float)*:

    Cutoff for small singular values of the pseudo inverse. (Method:
    'pinv')Singular values smaller than rcond*largest_singular_value are
    set to zero

    default: ``0.01``


- **update_response**:

    Update the (analytical) response per iteration.

    action: ``store_true``


- **use_errorbars**:

    Take into account the measured errorbars in the correction.

    action: ``store_true``


- **variable_categories**:

    List of names of the variables classes to use.

    default: ``['MQM', 'MQT', 'MQTL', 'MQY']``


- **weights** *(float)*:

    Weight to apply to each measured quantity. Input in order of
    optics_params.



Possible problems and notes (lmalina, 2020):
 * error-based weights default? likely - but be careful with low tune errors vs svd cut in pseudoinverse
 * manual creation of pd.DataFrame varslist, deltas? maybe tunes in tfs_pandas single value or a column?
 * There should be some summation/renaming for iterations
 * For two beam correction
 * The two beams can be treated separately until the calculation of correction
 * Values missing in the response (i.e. correctors of the other beam) shall be treated as zeros
 * Missing a part that treats the output from LSA

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint
from omc3.correction import handler
from omc3.optics_measurements.constants import (BETA, DISPERSION, F1001, F1010,
                                                NORM_DISPERSION, PHASE, TUNE)
from omc3.model import manager
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config

if TYPE_CHECKING:
    from generic_parser import DotDict

LOG = logging_tools.get_logger(__name__)

OPTICS_PARAMS_CHOICES = (f"{PHASE}X", f"{PHASE}Y",
                         f"{BETA}X", f"{BETA}Y",
                         f"{NORM_DISPERSION}X",
                         f"{TUNE}",
                         f"{DISPERSION}X", f"{DISPERSION}Y",
                         f"{F1001}R", f"{F1001}I", f"{F1010}R", f"{F1010}I")

CORRECTION_DEFAULTS = {
    "optics_file": None,
    "output_filename": "changeparameters_iter",
    "svd_cut": 0.01,
    "optics_params": OPTICS_PARAMS_CHOICES[:6],
    "variable_categories": ["MQM", "MQT", "MQTL", "MQY"],
    "beta_filename": "beta_phase_",
    "method": "pinv",
    "iterations": 4,
}


def correction_params():
    params = EntryPointParameters()
    params.add_parameter(name="meas_dir",
                         required=True,
                         type=PathOrStr,
                         help="Path to the directory containing the measurement files.",)
    params.add_parameter(name="output_dir",
                         required=True,
                         type=PathOrStr,
                         help="Path to the directory where to write the output files.", )
    params.add_parameter(name="fullresponse_path",
                         type=PathOrStr,
                         help="Path to the fullresponse binary file.If not given, "
                              "calculates the response analytically.",)
    params.add_parameter(name="optics_params",
                         type=str,
                         nargs="+",
                         default=list(CORRECTION_DEFAULTS["optics_params"]),
                         choices=OPTICS_PARAMS_CHOICES,
                         help=f"List of parameters to correct upon (e.g. {BETA}X {BETA}Y)", )
    params.add_parameter(name="output_filename",
                         default=CORRECTION_DEFAULTS["output_filename"],
                         help="Identifier of the output files.", )
    params.add_parameter(name="min_corrector_strength",
                         type=float,
                         default=0.,
                         help="Minimum (absolute) strength of correctors.",)
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
    params.add_parameter(name="weights",
                         nargs="+", type=float,
                         help="Weight to apply to each measured quantity. "
                              "Input in order of optics_params.",)
    params.add_parameter(name="variable_categories",
                         nargs="+",
                         default=CORRECTION_DEFAULTS["variable_categories"],
                         help="List of names of the variables classes to use.", )
    params.add_parameter(name="beta_filename",
                         default=CORRECTION_DEFAULTS["beta_filename"],
                         help="Prefix of the beta file to use. E.g.: beta_phase_", )
    params.add_parameter(name="method",
                         type=str,
                         choices=("pinv", "omp"),
                         default=CORRECTION_DEFAULTS["method"],
                         help="Optimization method to use.", )
    params.add_parameter(name="svd_cut",
                         type=float,
                         default=CORRECTION_DEFAULTS["svd_cut"],
                         help="Cutoff for small singular values of the pseudo inverse. "
                              "(Method: 'pinv')Singular values smaller than "
                              "rcond*largest_singular_value are set to zero", )
    params.add_parameter(name="n_correctors",
                         type=int,
                         help="Maximum number of correctors to use. (Method: 'omp')")
    params.add_parameter(name="iterations",
                         type=int,
                         default=CORRECTION_DEFAULTS["iterations"],
                         help="Maximum number of correction iterations to perform. "
                              "A value of `1` means the correction is calculated once."
                              "In this case, the accelerator instance does not need to be able"
                              "to produce a new model.", )
    params.add_parameter(name="use_errorbars",
                         action="store_true",
                         help="Take into account the measured errorbars as weights.", )
    params.add_parameter(name="update_response",
                         action="store_true",
                         help="Update the (analytical) response per iteration.", )
    return params


@entrypoint(correction_params())
def global_correction_entrypoint(opt: DotDict, accel_opt) -> None:
    """Do the global correction. Iteratively."""
    LOG.info("Starting Iterative Global Correction.")
    save_config(Path(opt.output_dir), opt, __file__, unknown_opt=accel_opt)

    opt = _check_opt_add_dicts(opt)
    opt = _add_hardcoded_paths(opt)
    opt.output_dir.mkdir(parents=True, exist_ok=True)
    accel_inst = manager.get_accelerator(accel_opt)
    handler.correct(accel_inst, opt)


def _check_opt_add_dicts(opt: dict) -> dict:  # acts inplace...
    """ Check on options and put in missing values """
    def_dict = _get_default_values()

    # Check cuts and fill defaults
    for key in ("modelcut", "errorcut", "weights"):
        if opt[key] is None:
            opt[key] = [def_dict[key][p] for p in opt.optics_params]
        elif len(opt[key]) != len(opt.optics_params):
            raise AttributeError(f"Length of {key} is not the same as of the optical parameters!")
        opt[key] = dict(zip(opt.optics_params, opt[key]))

    # Convert Strings to Paths
    opt.meas_dir = Path(opt.meas_dir)
    opt.output_dir = Path(opt.output_dir)
    if opt.fullresponse_path:
        opt.fullresponse_path = Path(opt.fullresponse_path)
    return opt


def _add_hardcoded_paths(opt: DotDict) -> DotDict:  # acts inplace...
    opt.change_params_path = opt.output_dir / f"{opt.output_filename}.madx"
    opt.change_params_correct_path = opt.output_dir / f"{opt.output_filename}_correct.madx"
    opt.knob_path = opt.output_dir / f"{opt.output_filename}.tfs"
    return opt


# Define functions here, to new optics params
def _get_default_values() -> dict[str, dict[str, float]]:
    return {
        "modelcut": {
            f"{PHASE}X": 0.05,
            f"{PHASE}Y": 0.05,
            f"{BETA}X": 0.2,
            f"{BETA}Y": 0.2,
            f"{DISPERSION}X": 0.2,
            f"{DISPERSION}Y": 0.2,
            f"{NORM_DISPERSION}X": 0.2,
            f"{TUNE}": 0.1,
            f"{F1001}R": 0.2,
            f"{F1001}I": 0.2,
            f"{F1010}R": 0.2,
            f"{F1010}I": 0.2,
        },
        "errorcut": {
            f"{PHASE}X": 0.035,
            f"{PHASE}Y": 0.035,
            f"{BETA}X": 0.02,
            f"{BETA}Y": 0.02,
            f"{DISPERSION}X": 0.02,
            f"{DISPERSION}Y": 0.02,
            f"{NORM_DISPERSION}X": 0.02,
            f"{TUNE}": 0.027,
            f"{F1001}R": 0.02,
            f"{F1001}I": 0.02,
            f"{F1010}R": 0.02,
            f"{F1010}I": 0.02,
        },
        "weights": {
            f"{PHASE}X": 1,
            f"{PHASE}Y": 1,
            f"{BETA}X": 0,
            f"{BETA}Y": 0,
            f"{DISPERSION}X": 0,
            f"{DISPERSION}Y": 0,
            f"{NORM_DISPERSION}X": 0,
            f"{TUNE}": 10,
            f"{F1001}R": 0,
            f"{F1001}I": 0,
            f"{F1010}R": 0,
            f"{F1010}I": 0,
        },
    }


if __name__ == "__main__":
    global_correction_entrypoint()
