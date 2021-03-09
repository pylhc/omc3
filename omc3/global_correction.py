r"""
Iterative Correction Scheme.

The response matrices :math:`R_{O}` for the observables :math:`O` (e.g. BBX, MUX, ...)
are loaded from a file and then the equation

.. math:: R_{O} \cdot \delta var = O_{meas} - O_{model}
    :label: eq1

is being solved for :math:`\delta var` via a chosen method (at the moment only numpys pinv,
which creates a pseudo-inverse via svd is used).

The response matrices are hereby merged into one matrix for all observables to solve vor all
:math:`\delta var` at the same time.

To normalize the observables to another ``weigths`` (W) can be applied.

Furthermore, an ``errorcut``, specifying the maximum errorbar for a BPM to be used, and
``modelcut``, specifying the maximum distance between measurement and model for a BPM to be used,
can be defined. Data from BPMs outside of those cut-values will be discarded.
These cuts are defined for each observable separately.

After each iteration the model variables are changed by :math:`-\delta var` and the
observables are recalculated by Mad-X.
:eq:`eq1` is then solved again.


:author: Lukas Malina, Joschua Dilly


Possible problems and notes:
 * error-based weights default? likely - but be carefull with low tune errors vs
svd cut in pseudoinverse
 * manual creation of pd.DataFrame varslist, deltas? maybe
tunes in tfs_pandas single value or a column?
 * There should be some summation/renaming for iterations
 * For two beam correction
 * The two beams can be treated separately until the calcultation of correction
 * Values missing in the response (i.e. correctors of the other beam) shall be
treated as zeros
 * Missing a part that treats the output from LSA

"""
import os
from typing import Dict

from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint

from omc3.correction import handler
from omc3.model import manager
from omc3.utils import iotools, logging_tools

LOG = logging_tools.get_logger(__name__)

CORRECTION_DEFAULTS = {
    "optics_file": None,
    "output_filename": "changeparameters_iter",
    "svd_cut": 0.01,
    "optics_params": ["MUX", "MUY", "BETX", "BETY", "NDX", "Q"],
    "variable_categories": ["MQM", "MQT", "MQTL", "MQY"],
    "beta_file_name": "beta_phase_",
    "method": "pinv",
    "max_iter": 3,
}


def correction_params():
    params = EntryPointParameters()
    params.add_parameter(name="meas_dir", required=True,
                         help="Path to the directory containing the measurement files.",)
    params.add_parameter(name="output_dir", required=True,
                         help="Path to the directory where to write the output files.", )
    params.add_parameter(name="fullresponse_path",
                         help="Path to the fullresponse binary file.If not given, "
                              "calculates the response analytically.",)
    params.add_parameter(name="optics_params", type=str, nargs="+",
                         default=CORRECTION_DEFAULTS["optics_params"],
                         choices=('MUX', 'MUY', 'BBX', 'BBY', 'BETX', 'BETY', 'DX', 'DY', 'NDX', 'Q',
                                  'F1001R', 'F1001I', 'F1010R', 'F1010I'),
                         help="List of parameters to correct upon (e.g. BETX BETY)", )
    params.add_parameter(name="output_filename", default=CORRECTION_DEFAULTS["output_filename"],
                         help="Identifier of the output files.", )
    params.add_parameter(name="min_corrector_strength", type=float, default=0.,
                         help="Minimum (absolute) strength of correctors.",)
    params.add_parameter(name="modelcut", nargs="+", type=float,
                         help="Reject BPMs whose deviation to the model is higher "
                              "than the correspoding input. Input in order of optics_params.",)
    params.add_parameter(name="errorcut", nargs="+", type=float,
                         help="Reject BPMs whose error bar is higher than the corresponding "
                              "input. Input in order of optics_params.",)
    params.add_parameter(name="weights", nargs="+", type=float,
                         help="Weight to apply to each measured quantity. "
                              "Input in order of optics_params.",)
    params.add_parameter(name="variable_categories", nargs="+",
                         default=CORRECTION_DEFAULTS["variable_categories"],
                         help="List of names of the variables classes to use.", )
    params.add_parameter(name="beta_file_name",
                         default=CORRECTION_DEFAULTS["beta_file_name"],
                         help="Prefix of the beta file to use. E.g.: getkmodbeta", )
    params.add_parameter(name="method", type=str, choices=("pinv", "omp"),
                         default=CORRECTION_DEFAULTS["method"],
                         help="Optimization method to use.", )
    params.add_parameter(name="svd_cut", type=float,
                         default=CORRECTION_DEFAULTS["svd_cut"],
                         help="Cutoff for small singular values of the pseudo inverse. "
                              "(Method: 'pinv')Singular values smaller than "
                              "rcond*largest_singular_value are set to zero", )
    params.add_parameter(name="n_correctors", type=int,
                         help="Maximum number of correctors to use. (Method: 'omp')")
    params.add_parameter(name="max_iter", type=int,
                         default=CORRECTION_DEFAULTS["max_iter"],
                         help="Maximum number of correction re-iterations to perform. "
                              "A value of `0` means the correction is calculated once.", )
    params.add_parameter(name="use_errorbars", action="store_true",
                         help="Take into account the measured errorbars in the correction.", )
    params.add_parameter(name="update_response", action="store_true",
                         help="Update the (analytical) response per iteration.", )
    return params


@entrypoint(correction_params())
def global_correction_entrypoint(opt: dict, accel_opt: dict) -> None:
    """Do the global correction. Iteratively.
    # TODO auto-generate docstring
    """
    LOG.info("Starting Iterative Global Correction.")
    opt = _check_opt_add_dicts(opt)
    opt = _add_hardcoded_paths(opt)
    iotools.create_dirs(opt.output_dir)
    accel_inst = manager.get_accelerator(accel_opt)
    handler.correct(accel_inst, opt)


def _check_opt_add_dicts(opt: dict) -> dict:  # acts inplace...
    """ Check on options and put in missing values """
    def_dict = _get_default_values()
    opt.optics_params = [p.replace("BB", "BET") for p in opt.optics_params]
    for key in ("modelcut", "errorcut", "weights"):
        if opt[key] is None:
            opt[key] = [def_dict[key][p] for p in opt.optics_params]
        elif len(opt[key]) != len(opt.optics_params):
            raise AttributeError(f"Length of {key} is not the same as of the optical parameters!")
        opt[key] = dict(zip(opt.optics_params, opt[key]))
    return opt


def _add_hardcoded_paths(opt: dict) -> dict:  # acts inplace...
    opt.change_params_path = os.path.join(opt.output_dir, f"{opt.output_filename}.madx")
    opt.change_params_correct_path = os.path.join(opt.output_dir, f"{opt.output_filename}_correct.madx")
    opt.knob_path = os.path.join(opt.output_dir, f"{opt.output_filename}.tfs")
    return opt


OPTICS_PARAMS_CHOICES = ("MUX", "MUY",  "BETX", "BETY", "DX", "DY", "NDX",
                         "Q", "F1001R", "F1001I", "F1010R", "F1010I")


# Define functions here, to new optics params
def _get_default_values() -> Dict[str, Dict[str, float]]:
    return {
        "modelcut": {
            "MUX": 0.05,
            "MUY": 0.05,
            "BETX": 0.2,
            "BETY": 0.2,
            "DX": 0.2,
            "DY": 0.2,
            "NDX": 0.2,
            "Q": 0.1,
            "F1001R": 0.2,
            "F1001I": 0.2,
            "F1010R": 0.2,
            "F1010I": 0.2,
        },
        "errorcut": {
            "MUX": 0.035,
            "MUY": 0.035,
            "BETX": 0.02,
            "BETY": 0.02,
            "DX": 0.02,
            "DY": 0.02,
            "NDX": 0.02,
            "Q": 0.027,
            "F1001R": 0.02,
            "F1001I": 0.02,
            "F1010R": 0.02,
            "F1010I": 0.02,
        },
        "weights": {
            "MUX": 1,
            "MUY": 1,
            "BETX": 0,
            "BETY": 0,
            "DX": 0,
            "DY": 0,
            "NDX": 0,
            "Q": 10,
            "F1001R": 0,
            "F1001I": 0,
            "F1010R": 0,
            "F1010I": 0,
        },
    }


if __name__ == "__main__":
    with logging_tools.DebugMode(active=True, log_file="iterative_correction.log"):
        global_correction_entrypoint()
