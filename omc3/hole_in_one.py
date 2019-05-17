"""
Entrypoint hole_in_one
------------------------

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
from copy import deepcopy
import tbt
from utils import logging_tools, iotools
from parser.entrypoint import entrypoint, EntryPoint, EntryPointParameters, add_to_arguments
from utils.contexts import timeit

LOGGER = logging_tools.get_logger(__name__)


def hole_in_one_params():
    params = EntryPointParameters()
    params.add_parameter(flags="--harpy", name="harpy", action="store_true",
                         help="Runs frequency analysis")
    params.add_parameter(flags="--optics", name="optics", action="store_true",
                         help="Measures the lattice optics")
    return params


@entrypoint(hole_in_one_params(), strict=False)
def hole_in_one_entrypoint(opt, rest):
    """
    Runs frequency analysis and measures lattice optics.

    Hole_in_one Kwargs:
      - **harpy**: Runs frequency analysis

        Flags: **--harpy**
        Action: ``store_true``
      - **optics**: Measures the lattice optics

        Flags: **--optics**
        Action: ``store_true``

    Harpy Kwargs:
      - **files**: TbT files to analyse

        Flags: **--files**
        Required: ``True``
      - **outputdir**: Output directory.

        Flags: **--outputdir**
        Required: ``True``
      - **to_write**: Choose the type of output.

        Flags: **--to_write**
        Choices: ``('lin', 'spectra', 'full_spectra', 'bpm_summary')``
        Default: ``['lin', 'bpm_summary']``
      - **turns** *(int)*: Turn index to start and first turn index to be ignored.

        Flags: **--turns**
        Default: ``[0, 50000]``
      - **unit** *(str)*: A unit of TbT BPM orbit data. All cuts and output are in 'mm'.

        Flags: **--unit**
        Choices: ``('m', 'cm', 'mm', 'um')``
        Default: ``mm``

      *--Cleaning--*

      - **clean**: If present, the data are first cleaned.

        Flags: **--clean**
        Action: ``store_true``
      - **bad_bpms**: Bad BPMs to clean.

        Flags: **--bad_bpms**
      - **first_bpm** *(str)*: First BPM in the measurement.
        Used to resynchronise the TbT data with model.

        Flags: **--first_bpm**
      - **keep_exact_zeros**: If present, will not remove BPMs with exact zeros in TbT data.

        Flags: **--keep_exact_zeros**
        Action: ``store_true``
      - **max_peak** *(float)*: Removes BPMs where the maximum orbit > limit.

        Flags: **--max_peak**
        Default: ``20.0``
      - **model**: Model for BPM locations

        Flags: **--model**
      - **opposite_direction**: If present, beam in the opposite direction to model
        is assumed for resynchronisation of BPMs.

        Flags: **--opposite_direction**
        Action: ``store_true``
      - **peak_to_peak** *(float)*: Peak to peak amplitude cut. This removes BPMs,
        where abs(max(turn values) - min(turn values)) <= threshold.

        Flags: **--peak_to_peak**
        Default: ``1e-05``
      - **sing_val** *(int)*: Keep this amount of largest singular values.

        Flags: **--sing_val**
        Default: ``12``
      - **svd_dominance_limit** *(float)*: Limit for single BPM dominating a mode.

        Flags: **--svd_dominance_limit**
        Default: ``0.925``
      - **wrong_polarity_bpms**: BPMs with swapped polarity in both planes.

        Flags: **--wrong_polarity_bpms**

      *--Frequency Analysis--*

      - **autotunes** *(str)*: The main tunes are guessed as the strongest line in SV^T matrix
        frequency spectrum: Synchrotron tune below ~0.03, betatron tunes above ~0.03.

        Flags: **--autotunes**
        Choices: ``('all', 'transverse')``
      - **is_free_kick**: If present, it will perform the free kick phase correction

        Flags: **--free_kick**
        Action: ``store_true``
      - **natdeltas** *(float)*: Guess for the offsets of natural tunes from
        the driven tunes (x, y, z). Disabled when set to 0.

        Flags: **--natdeltas**
      - **nattunes** *(float)*: Guess for the natural tunes (x, y, z).  Disabled when set to 0.

        Flags: **--nattunes**
      - **output_bits** *(int)*: Number (frequency, complex coefficient) pairs in the output
        is up to 2 ** output_bits (maximal in case full spectra is output).
        There is one pair (with maximal amplitude of complex coefficient) per interval
        of size 2 ** (- output_bits - 1).

        Flags: **--output_bits**
        Default: ``12``
      - **tolerance** *(float)*: Tolerance specifying an interval in frequency domain,
        where to look for the tunes.

        Flags: **--tolerance**
        Default: ``0.01``
      - **tune_clean_limit** *(float)*: The tune cleaning wont remove BPMs because of measured
        tune outliers closer to the average tune than this limit.

        Flags: **--tune_clean_limit**
        Default: ``1e-05``
      - **tunes** *(float)*: Guess for the main tunes [x, y, z]. Tunez is disabled when set to 0

        Flags: **--tunes**
      - **turn_bits** *(int)*: Number (frequency, complex coefficient) pairs in the calculation
        is 2 ** turn_bits, i.e. the difference between two neighbouring frequencies
        is 2 ** (- turn_bits - 1).

        Flags: **--turn_bits**
        Default: ``20``
      - **window** *(str)*: Windowing function to be used for frequency analysis.

        Flags: **--window**
        Choices: ``('rectangle', 'hamming', 'nuttal3', 'nuttal4')``
        Default: ``hamming``


    Optics Kwargs:
      - **files**: Files for analysis

        Flags: **--files**
        Required: ``True``
      - **outputdir**: Output directory

        Flags: **--outputdir**
        Required: ``True``
      - **calibrationdir** *(str)*: Path to calibration files directory.

        Flags: **--calibrationdir**
      - **coupling_method** *(int)*: Coupling analysis option: disabled, 1 BPM or 2 BPMs method

        Flags: **--coupling_method**
        Choices: ``(0, 1, 2)``
        Default: ``2``
      - **max_beta_beating** *(float)*: Maximal beta-beating allowed for action calculation.

        Flags: **--max_beta_beating**
        Default: ``0.15``
      - **max_closed_orbit** *(float)*: Maximal closed orbit in 'mm'
        allowed for dispersion measurement

        Flags: **--max_closed_orbit**
        Default: ``4.0``
      - **nonlinear**: Calculate higher order RDTs

        Flags: **--nonlinear**
        Action: ``store_true``
      - **only_coupling**: Calculate only coupling.

        Flags: **--only_coupling**
        Action: ``store_true``
      - **range_of_bpms** *(int)*: Range of BPMs for beta from phase calculation

        Flags: **--range_of_bpms**
        Choices: ``(5, 7, 9, 11, 13, 15)``
        Default: ``11``
      - **three_bpm_method**: Use 3 BPM method in beta from phase

        Flags: **--three_bpm_method**
        Action: ``store_true``
      - **union**: If present, the phase advances are calculate for union of BPMs
        with at least 3 valid measurements, instead of intersection .

        Flags: **--union**
        Action: ``store_true``


    Accelerator Kwargs:  TODO

    """
    if not opt.harpy and not opt.optics:
        raise SystemError("No module has been chosen.")
    if not rest:
        raise SystemError("No input has been set.")
    harpy_opt, optics_opt = _get_suboptions(opt, rest)
    lins = []
    if harpy_opt is not None:
        lins = _run_harpy(harpy_opt)
    if optics_opt is not None:
        _measure_optics(lins, optics_opt)


def _get_suboptions(opt, rest):
    if opt.harpy:
        harpy_opt, rest = _harpy_entrypoint(rest)
        if opt.optics:
            rest = add_to_arguments(rest, entry_params=optics_params(),
                                    files=harpy_opt.files,
                                    outputdir=harpy_opt.outputdir)
            harpy_opt.outputdir = join(harpy_opt.outputdir, 'lin_files')
            rest = add_to_arguments(rest, entry_params={"model_dir": {"flags": "--model_dir"}},
                                    model_dir=dirname(abspath(harpy_opt.model)))
    else:
        harpy_opt = None
    if opt.optics:
        optics_opt, rest = _optics_entrypoint(rest)
        from model import manager
        optics_opt.accelerator = manager.get_accel_instance(rest)
        if not optics_opt.accelerator.excitation and optics_opt.compensation != "none":
            raise AttributeError("Compensation requested and no driven model was provided.")

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
        tbt_datas = [(tbt.read(option.files), option) for option in all_options]
        for tbt_data, option in tbt_datas:
            lins.extend([handler.run_per_bunch(bunch_data, bunch_options)
                         for bunch_options, bunch_data in _multibunch(option, tbt_data)])
    return lins


def _replicate_harpy_options_per_file(options):
    list_of_options = []
    for input_file in options.files:
        new_options = deepcopy(options)
        new_options.files = input_file
        list_of_options.append(new_options)
    return list_of_options


def _multibunch(options, tbt_datas):
    if tbt_datas.nbunches == 1:
        yield options, tbt_datas
        return
    for index in range(tbt_datas.nbunches):
        new_options = deepcopy(options)
        new_file_name = f"bunchid{tbt_datas.bunch_ids[index]}_{basename(new_options.files)}"
        new_options.files = join(dirname(options.files), new_file_name)
        yield new_options, tbt.TbtData([tbt_datas.matrices[index]], tbt_datas.date,
                                       [tbt_datas.bunch_ids[index]], tbt_datas.nturns)


def _measure_optics(lins, optics_opt):
    from optics_measurements import measure_optics
    if len(lins) == 0:
        lins = optics_opt.files
    inputs = measure_optics.InputFiles(lins, optics_opt)
    iotools.create_dirs(optics_opt.outputdir)
    calibrations = measure_optics.copy_calibration_files(optics_opt.outputdir,
                                                         optics_opt.calibrationdir)
    inputs.calibrate(calibrations)
    with timeit(lambda spanned: LOGGER.info(f"Total time for optics measurements: {spanned}")):
        measure_optics.measure_optics(inputs, optics_opt)


def _harpy_entrypoint(params):
    options, rest = EntryPoint(harpy_params(), strict=False).parse(params)
    if options.natdeltas is not None and options.nattunes is not None:
        raise AttributeError("Colliding options found: --nattunes and --natdeltas. Choose only one")
    if options.tunes is not None and options.autotunes is not None:
        raise AttributeError("Colliding options found: --tunes and --autotunes. Choose only one")
    if options.tunes is None and options.autotunes is None:
        raise AttributeError("One of the options --tunes and --autotunes has to be used.")
    if options.bad_bpms is None:
        options.bad_bpms = []
    if options.wrong_polarity_bpms is None:
        options.wrong_polarity_bpms = []
    if options.is_free_kick:
        options.window = "rectangle"
    return options, rest


def harpy_params():
    params = EntryPointParameters()
    params.add_parameter(flags="--files", name="files", required=True, nargs='+',
                         help="TbT files to analyse")
    params.add_parameter(flags="--outputdir", name="outputdir", required=True,
                         help="Output directory.")
    params.add_parameter(flags="--model", name="model", help="Model for BPM locations")
    params.add_parameter(flags="--unit", name="unit", type=str, choices=("m", "cm", "mm", "um"),
                         default=HARPY_DEFAULTS["unit"],
                         help=f"A unit of TbT BPM orbit data. All cuts and output are in 'mm'.")
    params.add_parameter(flags="--turns", name="turns", type=int, nargs=2,
                         default=HARPY_DEFAULTS["turns"],
                         help="Turn index to start and first turn index to be ignored.")
    params.add_parameter(flags="--to_write", name="to_write", nargs='+',
                         default=HARPY_DEFAULTS["to_write"],
                         choices=('lin', 'spectra', 'full_spectra', 'bpm_summary'),
                         help="Choose the type of output. ")

    # Cleaning parameters
    params.add_parameter(flags="--clean", name="clean", action="store_true",
                         help="If present, the data are first cleaned.")
    params.add_parameter(flags="--sing_val", name="sing_val", type=int,
                         default=HARPY_DEFAULTS["sing_val"],
                         help="Keep this amount of largest singular values.")
    params.add_parameter(flags="--peak_to_peak", name="peak_to_peak", type=float,
                         default=HARPY_DEFAULTS["peak_to_peak"],
                         help="Peak to peak amplitude cut. This removes BPMs, "
                              "where abs(max(turn values) - min(turn values)) <= threshold.")
    params.add_parameter(flags="--max_peak", name="max_peak", type=float,
                         default=HARPY_DEFAULTS["max_peak"],
                         help="Removes BPMs where the maximum orbit > limit.")
    params.add_parameter(flags="--svd_dominance_limit", name="svd_dominance_limit",
                         type=float, default=HARPY_DEFAULTS["svd_dominance_limit"],
                         help="Limit for single BPM dominating a mode.")
    params.add_parameter(flags="--bad_bpms", name="bad_bpms", nargs='*', help="Bad BPMs to clean.")
    params.add_parameter(flags="--wrong_polarity_bpms", name="wrong_polarity_bpms", nargs='*',
                         help="BPMs with swapped polarity in both planes.")
    params.add_parameter(flags="--keep_exact_zeros", name="keep_exact_zeros", action="store_true",
                         help="If present, will not remove BPMs with exact zeros in TbT data.")
    params.add_parameter(flags="--first_bpm", name="first_bpm", type=str,
                         help="First BPM in the measurement. "
                              "Used to resynchronise the TbT data with model.")
    params.add_parameter(flags="--opposite_direction", name="opposite_direction",
                         action="store_true",
                         help="If present, beam in the opposite direction to model"
                              " is assumed for resynchronisation of BPMs.")

    # Harmonic analysis parameters
    params.add_parameter(flags="--tunes", name="tunes", type=float, nargs=3,
                         help="Guess for the main tunes [x, y, z]. Tunez is disabled when set to 0")
    params.add_parameter(flags="--nattunes", name="nattunes", type=float, nargs=3,
                         help="Guess for the natural tunes (x, y, z).  Disabled when set to 0.")
    params.add_parameter(flags="--natdeltas", name="natdeltas", type=float, nargs=3,
                         help="Guess for the offsets of natural tunes from the driven tunes"
                              " (x, y, z). Disabled when set to 0.")
    params.add_parameter(flags="--autotunes", name="autotunes", type=str,
                         choices=("all", "transverse"),
                         help="The main tunes are guessed as "
                              "the strongest line in SV^T matrix frequency spectrum: "
                              "Synchrotron tune below ~0.03, betatron tunes above ~0.03.")
    params.add_parameter(flags="--tune_clean_limit", name="tune_clean_limit", type=float,
                         default=HARPY_DEFAULTS["tune_clean_limit"],
                         help="The tune cleaning wont remove BPMs because of measured tune outliers"
                              " closer to the average tune than this limit.")
    params.add_parameter(flags="--tolerance", name="tolerance", type=float,
                         default=HARPY_DEFAULTS["tolerance"],
                         help="Tolerance specifying an interval in frequency domain, where to look "
                              "for the tunes.")
    params.add_parameter(flags="--free_kick", name="is_free_kick", action="store_true",
                         help="If present, it will perform the free kick phase correction")
    params.add_parameter(flags="--window", name="window", type=str,
                         choices=("rectangle", "hann", "triangle", "welch", "hamming", "nuttal3",
                                  "nuttal4"), default=HARPY_DEFAULTS["window"],
                         help="Windowing function to be used for frequency analysis.")
    params.add_parameter(flags="--turn_bits", name="turn_bits", type=int,
                         default=HARPY_DEFAULTS["turn_bits"],
                         help="Number (frequency, complex coefficient) pairs in the calculation"
                              " is 2 ** turn_bits, i.e. the difference between "
                              "two neighbouring frequencies is 2 ** (- turn_bits - 1).")
    params.add_parameter(flags="--output_bits", name="output_bits", type=int,
                         default=HARPY_DEFAULTS["output_bits"],
                         help="Number (frequency, complex coefficient) pairs in the output "
                              "is up to 2 ** output_bits (maximal in case full spectra is output). "
                              "There is one pair (with maximal amplitude of complex coefficient) "
                              "per interval of size 2 ** (- output_bits - 1).")
    return params


def _optics_entrypoint(params):
    return EntryPoint(optics_params(), strict=False).parse(params)


def optics_params():
    params = EntryPointParameters()
    params.add_parameter(flags="--files", name="files",  required=True, nargs='+',
                         help="Files for analysis")
    params.add_parameter(flags="--outputdir", name="outputdir", required=True,
                         help="Output directory")
    params.add_parameter(flags="--calibrationdir", name="calibrationdir", type=str,
                         help="Path to calibration files directory.")
    params.add_parameter(flags="--coupling_method", name="coupling_method", type=int,
                         choices=(0, 1, 2), default=OPTICS_DEFAULTS["coupling_method"],
                         help="Analysis option for coupling: disabled, 1 BPM or 2 BPMs method")
    params.add_parameter(flags="--range_of_bpms", name="range_of_bpms", type=int,
                         choices=(5, 7, 9, 11, 13, 15),  default=OPTICS_DEFAULTS["range_of_bpms"],
                         help="Range of BPMs for beta from phase calculation")
    params.add_parameter(flags="--max_beta_beating", name="max_beta_beating", type=float,
                         default=OPTICS_DEFAULTS["max_beta_beating"],
                         help="Maximal beta-beating allowed for action calculation.")
    params.add_parameter(flags="--max_closed_orbit", name="max_closed_orbit", type=float,
                         default=OPTICS_DEFAULTS["max_closed_orbit"],
                         help="Maximal closed orbit in 'mm' allowed for dispersion measurement")
    params.add_parameter(flags="--union", name="union", action="store_true",
                         help="If present, the phase advances are calculate for union of BPMs "
                              "with at least 3 valid measurements, instead of intersection .")
    params.add_parameter(flags="--nonlinear", name="nonlinear", action="store_true",
                         help="Calculate higher order RDTs")
    params.add_parameter(flags="--three_bpm_method", name="three_bpm_method", action="store_true",
                         help="Use 3 BPM method in beta from phase")
    params.add_parameter(flags="--only_coupling", name="only_coupling", action="store_true",
                         help="Calculate only coupling. ")
    params.add_parameter(flags="--compensation", name="compensation", type=str,
                         choices=("model", "equation", "none"), default=OPTICS_DEFAULTS["compensation"],
                         help="Mode of compensation for the analysis after driven beam excitation")
    params.add_parameter(flags="--three_d_excitation", name="three_d_excitation",
                         action="store_true", help="Use 3D kicks to calculate dispersion")
    params.add_parameter(flags="--isolation_forest", name="isolation_forest", action="store_true",
                         help="Remove outlying BPMs with isolation forest")
    params.add_parameter(flags="--second_order_dispersion", name="second_order_dispersion",
                         action="store_true", help="Calculate second order dispersion")
    params.add_parameter(flags="--chromatic_beating", name="chromatic_beating",
                         action="store_true", help="Calculate chromatic beatings: W, PHI and coupling")
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
    "window": "hann",
    "turn_bits": 20,
    "output_bits": 12,
    "to_write": ["lin", "bpm_summary"]
}

OPTICS_DEFAULTS = {
        "max_closed_orbit": 4.0,
        "coupling_method": 2,
        "range_of_bpms": 11,
        "max_beta_beating": 0.15,
        "compensation": "model",
}

if __name__ == "__main__":
    hole_in_one_entrypoint()
