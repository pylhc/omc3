"""
.. module: optics_input

Created on 11/07/18

:author: Lukas Malina

Parses options and creates input object for measure_optics.py
"""
import argparse
from model import manager


def parse_args(args=None):
    parser = _get_optics_parser()
    options, acc_args = parser.parse_known_args(args)
    accelerator = manager.get_accel_instance(acc_args)
    options.accelerator = accelerator
    optics_input = OpticsInput.init_from_options(options)
    return optics_input


class OpticsInput(object):
    DEFAULTS = {
        "calibrationdir": None,
        "max_closed_orbit": 4.0,
        "coupling_method": 2,
        "orbit_unit": "mm",
        "range_of_bpms": 11,
        "beta_model_cut": 0.15,
        "union": False,
        "nonlinear": False,
        "three_bpm_method": False,
        "only_coupling": False
    }

    def __init__(self):
        self.files = None
        self.outputdir = None
        self.calibrationdir = OpticsInput.DEFAULTS["calibrationdir"]
        self.max_closed_orbit = OpticsInput.DEFAULTS["max_closed_orbit"]
        self.coupling_method = OpticsInput.DEFAULTS["coupling_method"]
        self.orbit_unit = OpticsInput.DEFAULTS["orbit_unit"]
        self.range_of_bpms = OpticsInput.DEFAULTS["range_of_bpms"]
        self.beta_model_cut = OpticsInput.DEFAULTS["beta_model_cut"]
        self.union = OpticsInput.DEFAULTS["union"]
        self.nonlinear = OpticsInput.DEFAULTS["nonlinear"]
        self.three_bpm_method = OpticsInput.DEFAULTS["three_bpm_method"]
        self.only_coupling = OpticsInput.DEFAULTS["only_coupling"]
        self.accelerator = None

    @staticmethod
    def init_from_options(options):
        self = OpticsInput()
        self.files = options.files
        self.outputdir = options.outputdir
        self.calibrationdir = options.calibrationdir
        self.max_closed_orbit = options.max_closed_orbit
        self.coupling_method = options.coupling_method
        self.orbit_unit = options.orbit_unit
        self.range_of_bpms = options.range_of_bpms
        self.beta_model_cut = options.beta_model_cut
        self.union = options.union
        self.nonlinear = options.nonlinear
        self.three_bpm_method = options.three_bpm_method
        self.only_coupling = options.only_coupling
        self.accelerator = options.accelerator
        return self


def _get_optics_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--files", "--file", dest="files",  required=True,
                        help="Files from analysis, separated by comma")
    parser.add_argument("--outputdir", dest="outputdir", required=True, help="Output directory")
    parser.add_argument("--max_closed_orbit", dest="max_closed_orbit",
                        default=OpticsInput.DEFAULTS["max_closed_orbit"],
                        help="Maximal closed orbit for dispersion measurement in 'orbit_unit'")
    parser.add_argument("--calibrationdir", dest="calibrationdir", type=str,
                        default=OpticsInput.DEFAULTS["calibrationdir"],
                        help="Directory where the calibration files are stored")
    parser.add_argument("--coupling_method", dest="coupling_method", type=int, choices=(0, 1, 2),
                        default=OpticsInput.DEFAULTS["coupling_method"],
                        help="Analysis option for coupling: disabled, 1 BPM or 2 BPMs")
    parser.add_argument("--orbit_unit", dest="orbit_unit", type=str,
                        choices=("um", "mm", "cm", "m"),
                        default=OpticsInput.DEFAULTS["orbit_unit"], help="Unit of orbit position.")
    parser.add_argument("--range_of_bpms", dest="range_of_bpms", choices=(5, 7, 9, 11, 13, 15),
                        type=int, default=OpticsInput.DEFAULTS["range_of_bpms"],
                        help="Range of BPMs for beta from phase calculation")
    parser.add_argument("--beta_model_cut", dest="beta_model_cut", type=float,
                        default=OpticsInput.DEFAULTS["beta_model_cut"],
                        help="Set beta-beating threshold for action calculations")
    parser.add_argument("--union", dest="union", action="store_true",
                        help="The phase per BPM is calculated from at least 3 valid measurements.")
    parser.add_argument("--nonlinear", dest="nonlinear", action="store_true",
                        help="Run the RDT analysis")
    parser.add_argument("--three_bpm_method", dest="three_bpm_method", action="store_true",
                        help="Use 3 BPM method only")  # TODO --no_systematic_errors option instead?
    parser.add_argument("--only_coupling", dest="only_coupling", action="store_true",
                        help="Only coupling is calculated. ")
    return parser
