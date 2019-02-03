"""
.. module: hole_in_one

Created on 27/01/19

:author: Lukas Malina

Top-level script, which computes:
    frequency spectra of Turn-by-Turn BPM data
    various lattice optics parameters from frequency spectra
    various lattice optics parameters from Turn-by-Turn BPM data

Generally, analysis flows as follows:
   Turn-by-Turn BPM data   --->    frequency spectra   --->    various lattice optics parameters

Stages represented by different files:
    Sdds file:  .sdds      --->   Tfs files: .lin[xy]  --->    Tfs files: .out

To run either of the two or both steps, use options:
                          --harpy                     --optics
"""
from os.path import join, dirname, basename, abspath
import tbt
from utils import logging_tools, iotools
from utils.entrypoint import entrypoint, EntryPoint, EntryPointParameters
from utils.contexts import timeit

LOGGER = logging_tools.get_logger(__name__)


def hole_in_one_entrypoint():
    params = EntryPointParameters()
    params.add_parameter(flags="--harpy", name="harpy", action="store_true",
                         help="Runs frequency analysis")
    params.add_parameter(flags="--optics", name="optics", action="store_true",
                         help="Measures the lattice optics")
    return params


@entrypoint(hole_in_one_entrypoint(), strict=False)
def hole_in_one(opt, rest):
    if not opt.harpy and not opt.optics:
        raise SystemError("No module has been chosen.")
    if not rest:
        raise SystemError("No input has been set.")
    harpy_opt, optics_opt = _get_options(opt, rest)
    lins = []
    if harpy_opt is not None:
        lins = _run_harpy(harpy_opt)
    if optics_opt is not None:
        _measure_optics(lins, optics_opt)


def _get_options(opt, rest):
    if opt.harpy:
        harpy_opt, rest = _harpy_entrypoint(rest)
        if opt.optics:
            rest.extend(['--files'] + harpy_opt.file)
            rest.extend(['--outputdir'] + [join(harpy_opt.outputdir, 'optics')])
            rest.extend(['--model_dir'] + [dirname(abspath(harpy_opt.model))])
    else:
        harpy_opt = None
    if opt.optics:
        optics_opt, rest = _optics_entrypoint(rest)
        from model import manager
        optics_opt.accelerator = manager.get_accel_instance(rest)
    else:
        optics_opt = None
    return harpy_opt, optics_opt


def _run_harpy(harpy_options):
    from harpy import handler
    import tbt
    iotools.create_dirs(harpy_options.outputdir)
    with timeit(lambda spanned: LOGGER.info(f"Total time for Harpy: {spanned}")):
        lins = []
        all_options = _replicate_harpy_options_per_file(harpy_options)
        tbt_datas = [(tbt.read(option.file), option) for option in all_options]
        for tbt_data, option in tbt_datas:
            lins.extend([handler.run_per_bunch(bunch_data, bunch_options)
                         for bunch_options, bunch_data in _multibunch(option, tbt_data)])
    return lins


def _replicate_harpy_options_per_file(options):
    list_of_options = []
    from copy import copy
    for input_file in options.file:
        new_options = copy(options)
        new_options.file = input_file
        list_of_options.append(new_options)
    return list_of_options


def _multibunch(options, tbt_datas):
    if tbt_datas.nbunches == 1:
        yield options, tbt_datas
        return
    from copy import copy
    for index in range(tbt_datas.nbunches):
        new_options = copy(options)
        new_file_name = f"bunchid{tbt_datas.bunch_ids[index]}_{basename(new_options.file)}"
        new_options.file = join(dirname(options.file), new_file_name)
        yield new_options, tbt.TbtData([tbt_datas.matrices[index]], tbt_datas.date,
                                       [tbt_datas.bunch_ids[index]], tbt_datas.nturns)


def _measure_optics(lins, optics_opt):
    from optics_measurements import measure_optics
    if len(lins) == 0:
        lins = optics_opt.files
    inputs = measure_optics.InputFiles(lins)
    iotools.create_dirs(optics_opt.outputdir)
    calibrations = measure_optics.copy_calibration_files(optics_opt.outputdir,
                                                         optics_opt.calibrationdir)
    inputs.calibrate(calibrations)
    measure_optics.measure_optics(inputs, optics_opt)


def _harpy_entrypoint(unknown_params):
    options, rest = EntryPoint(harpy_params(), strict=False).parse(unknown_params)
    if options.natdeltas is not None and options.nattunes is not None:
        raise AttributeError("Colliding options found: --nattunes and --natdeltas. Choose only one")
    if options.tunes is not None and options.autotunes is not None:
        raise AttributeError("Colliding options found: --tunes and --autotunes. Choose only one")
    if options.tunes is None and options.autotunes is None:
        raise AttributeError("One of the options --tunes and --autotunes has to be used.")
    if [x for x in options.to_write if x not in ("lin", "spectra", "full_spectra", "bpm_summary")]:
        raise ValueError(f"Unknown options found in to_write")
    if options.bad_bpms is None:
        options.bad_bpms = []
    if options.wrong_polarity_bpms is None:
        options.wrong_polarity_bpms = []
    if options.is_free_kick:
        options.window = "rectangle"
    return options, rest


def harpy_params():
    params = EntryPointParameters()
    params.add_parameter(flags="--file", name="file", required=True, nargs='+',
                         help="TbT files to analyse")
    params.add_parameter(flags="--outputdir", name="outputdir", required=True,
                         help="Output directory. Default: the input file directory.")
    params.add_parameter(flags="--model", name="model", help="Model for BPM locations")
    params.add_parameter(flags="--unit", name="unit", type=str, choices=("m", "cm", "mm", "um"),
                         default=HARPY_DEFAULTS["unit"],
                         help="A unit of TbT BPM orbit data. Default is: %(default)s. "
                              "Cuts and output are in milimeters.")
    params.add_parameter(flags="--turns", name="turns", type=int, nargs=2,
                         default=HARPY_DEFAULTS["turns"],
                         help="Turn index to start and first turn index to be ignored. "
                              "Default: %(default)s")
    params.add_parameter(flags="--sequential", name="sequential", action="store_true",
                         help="If set, it will run in only one process.")
    params.add_parameter(flags="--to_write", name="to_write", nargs='+',
                         default=HARPY_DEFAULTS["to_write"],
                         help="Choose the output: 'lin'  'spectra' 'full_spectra' 'bpm_summary'")

    # Cleaning parameters
    params.add_parameter(flags="--clean", name="clean", action="store_true",
                         help="If present, the data are first cleaned.")
    params.add_parameter(flags="--sing_val", name="sing_val", type=int,
                         default=HARPY_DEFAULTS["sing_val"],
                         help="Keep this amount of largest singular values. Default: %(default)s")
    params.add_parameter(flags="--peak_to_peak", name="peak_to_peak", type=float,
                         default=HARPY_DEFAULTS["peak_to_peak"],
                         help="Peak to peak amplitude cut. This removes BPMs, "
                              "where abs(max(turn values) - min(turn values)) <= threshold. "
                              "Default: %(default)s")
    params.add_parameter(flags="--max_peak", name="max_peak", type=float,
                         default=HARPY_DEFAULTS["max_peak"],
                         help="Removes BPMs where the maximum orbit > limit. Default: %(default)s")
    params.add_parameter(flags="--svd_dominance_limit", name="svd_dominance_limit",
                         type=float, default=HARPY_DEFAULTS["svd_dominance_limit"],
                         help="Threshold for single BPM dominating a mode. Default: %(default)s")
    params.add_parameter(flags="--bad_bpms", name="bad_bpms", nargs='*', help="Bad BPMs to clean.")
    params.add_parameter(flags="--wrong_polarity_bpms", name="wrong_polarity_bpms", nargs='*',
                         help="BPMs with swapped polarity in both planes.")
    params.add_parameter(flags="--no_exact_zeros", name="no_exact_zeros", action="store_true",
                         help="If present, will not remove BPMs with a single zero.")
    params.add_parameter(flags="--first_bpm", name="first_bpm", type=str,
                         help="First BPM in the measurement. "
                              "Used to resynchronise the TbT data with model.")
    params.add_parameter(flags="--opposite_direction", name="opposite_direction",
                         action="store_true",
                         help="If present, beam in the opposite direction to model"
                              " is assumed for resynchronisation of BPMs.")

    # Harmonic analysis parameters
    params.add_parameter(flags="--tunes", name="tunes", type=float, nargs=3,
                         help="Guess for the main tunes(x, y, z). Tunez is disabled when set to 0")
    params.add_parameter(flags="--nattunes", name="nattunes", type=float, nargs=3,
                         help="Guess for the natural tunes (x, y, z).  Disabled when set to 0.")
    params.add_parameter(flags="--natdeltas", name="natdeltas", type=float, nargs=3,
                         help="Guess for the offsets of natural tunes from the driven tunes"
                              " (x, y, z). Disabled when set to zero.")
    params.add_parameter(flags="--autotunes", name="autotunes", type=str,
                         choices=("all", "transverse"),
                         help="If present, the main tunes are guessed as "
                              "the strongest line in SV^T matrix. "
                              "In (0.0, 0.03) resp (0.03,0.5) for sychrpotron resp betatron tunes")
    params.add_parameter(flags="--tune_clean_limit", name="tune_clean_limit", type=float,
                         default=HARPY_DEFAULTS["tune_clean_limit"],
                         help="The autoclean wont remove tune deviation lower than this limit.")
    params.add_parameter(flags="--tolerance", name="tolerance", type=float,
                         default=HARPY_DEFAULTS["tolerance"],
                         help="Tolerance on the guess for the tunes.")
    params.add_parameter(flags="--free_kick", name="is_free_kick", action="store_true",
                         help="If present, it will perform the free kick phase correction")
    params.add_parameter(flags="--window", name="window", type=str,
                         choices=("nuttal3", "nuttal4", "hamming", "rectangle"),
                         default=HARPY_DEFAULTS["window"],
                         help="Windowing function to be used for frequency analysis.")
    params.add_parameter(flags="--turn_bits", name="turn_bits", type=int,
                         default=HARPY_DEFAULTS["turn_bits"],
                         help="Number of bits of required DFT peaks in the calculation [0, 0.5)")
    params.add_parameter(flags="--output_bits", name="output_bits", type=int,
                         default=HARPY_DEFAULTS["output_bits"],
                         help="Number of bits of required DFT peaks in the output [0, 0.5)")
    return params


def _optics_entrypoint(unknown_params):
    return EntryPoint(optics_params(), strict=False).parse(unknown_params)


def optics_params():
    params = EntryPointParameters()
    params.add_parameter(flags=["--files", "--file"], name="files",  required=True, nargs='+',
                         help="Files for analysis")
    params.add_parameter(flags="--outputdir", name="outputdir", required=True,
                         help="Output directory")
    params.add_parameter(flags="--calibrationdir", name="calibrationdir", type=str,
                         default=OPTICS_DEFAULTS["calibrationdir"],
                         help="Directory where the calibration files are stored")
    params.add_parameter(flags="--coupling_method", name="coupling_method", type=int,
                         choices=(0, 1, 2), default=OPTICS_DEFAULTS["coupling_method"],
                         help="Analysis option for coupling: disabled, 1 BPM or 2 BPMs")
    params.add_parameter(flags="--range_of_bpms", name="range_of_bpms", type=int,
                         choices=(5, 7, 9, 11, 13, 15),  default=OPTICS_DEFAULTS["range_of_bpms"],
                         help="Range of BPMs for beta from phase calculation")
    params.add_parameter(flags="--beta_model_cut", name="beta_model_cut", type=float,
                         default=OPTICS_DEFAULTS["beta_model_cut"],
                         help="Set beta-beating threshold for action calculations")
    params.add_parameter(flags="--max_closed_orbit", name="max_closed_orbit", type=float,
                         default=OPTICS_DEFAULTS["max_closed_orbit"],
                         help="Maximal closed orbit for dispersion measurement in 'orbit_unit'")
    params.add_parameter(flags="--union", name="union", action="store_true",
                         help="The phase per BPM is calculated from at least 3 valid measurements.")
    params.add_parameter(flags="--nonlinear", name="nonlinear", action="store_true",
                         help="Run the RDT analysis")
    params.add_parameter(flags="--three_bpm_method", name="three_bpm_method", action="store_true",
                         help="Use 3 BPM method only")
    params.add_parameter(flags="--only_coupling", name="only_coupling", action="store_true",
                         help="Only coupling is calculated. ")
    return params


HARPY_DEFAULTS = {
    "turns": [0, 50000],
    "unit": "mm",
    "sing_val": 12,
    "peak_to_peak": 1e-5,
    "max_peak": 20.0,
    "svd_dominance_limit": 0.925,
    "tolerance": 0.01,
    "tune_clean_limit": 1e-5,
    "window": "hamming",
    "turn_bits": 20,
    "output_bits": 12,
    "to_write": ["lin", "bpm_summary"]
}

OPTICS_DEFAULTS = {
        "calibrationdir": None,
        "max_closed_orbit": 4.0,
        "coupling_method": 2,
        "orbit_unit": "mm",
        "range_of_bpms": 11,
        "beta_model_cut": 0.15,
}

if __name__ == "__main__":
    hole_in_one()
