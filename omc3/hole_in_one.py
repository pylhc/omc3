"""
Hole in One
-----------

``hole_in_one`` is the top-level script of analysis functionality offered in ``omc3``. In most of
your use cases, this is the file you will want to call. It handles:
- frequency spectra of Turn-by-Turn BPM data,
- various lattice optics parameters from frequency spectra,
- various lattice optics parameters from Turn-by-Turn BPM data,

A general analysis workflow, from straight out turn-by-turn measurement or simulations files to
results, goes as follows:

+-----------------------+--------+---------------------+------+-----------------------------------+
|                      Analysis Workflow                                                          |
+=======================+========+=====================+======+===================================+
| Turn-by-Turn BPM data | --->   |  frequency spectra  | ---> | various lattice optics parameters |
+-----------------------+--------+---------------------+------+-----------------------------------+

The first step above consists in frequency analysis performed by ``harpy``, while the second
one is optics analysis performed by ``measure_optics``. Each corresponding stage is represented
by a different set of files:

+--------------------------+--------+---------------------------+------+-----------------------+
|                     Corresponding Files                                                      |
+==========================+========+===========================+======+=======================+
|  SDDS file:  **.sdds**   | --->   |  Tfs files: **.lin[xy]**  | ---> |  Tfs files: **.tfs**  |
+--------------------------+--------+---------------------------+------+-----------------------+

To run either of the two or both steps, see options ``--harpy`` and ``--optics``.
"""
from __future__ import annotations
import os
from collections.abc import Generator
from copy import deepcopy
from datetime import datetime, timezone
from os.path import abspath, basename, dirname, join

import turn_by_turn as tbt
from generic_parser import DotDict
from generic_parser.entrypoint_parser import (
    EntryPoint,
    EntryPointParameters,
    add_to_arguments,
    entrypoint,
    save_options_to_config,
)

from omc3.definitions import formats
from omc3.harpy import handler
from omc3.harpy.constants import LINFILES_SUBFOLDER
from omc3.model import manager
from omc3.optics_measurements import measure_optics, phase
from omc3.optics_measurements.data_models import InputFiles
from omc3.utils import iotools, logging_tools
from omc3.utils.contexts import timeit

LOGGER = logging_tools.get_logger(__name__)

DEFAULT_CONFIG_FILENAME = "analysis_{time:s}.ini"


def hole_in_one_params():
    params = EntryPointParameters()
    params.add_parameter(name="harpy", action="store_true", help="Runs frequency analysis")
    params.add_parameter(name="optics", action="store_true", help="Measures the lattice optics")
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
      - **suffix** *(str)*: User-defined suffix for the output filenames.

        Flags: **--suffix**
      - **to_write**: Choose the type of output.

        Flags: **--to_write**
        Choices: ``('lin', 'spectra', 'full_spectra', 'bpm_summary')``
        Default: ``['lin', 'bpm_summary']``
      - **turns** *(int)*: Turn index to start and first turn index to be ignored.

        Flags: **--turns**
        Default: ``[0, 50000]``
      - **bunch_ids** *(int)*: Bunches to process in multi-bunch file. If not specified, all bunches
        are processed.

        Flags: **--bunch_ids**
      - **unit** *(str)*: A unit of TbT BPM orbit data. All cuts and output are in 'm'.

        Flags: **--unit**
        Choices: ``('m', 'cm', 'mm', 'um')``
        Default: ``m``
      - **tbt_datatype** *(str)*: Choose datatype from which to import (e.g LHC binary SDDS, numpy npz).

        Flags: **--tbt_datatype**
        Default: ``LHC``

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
        Default: ``0.02``
      - **model**: Model for BPM locations

        Flags: **--model**
      - **num_svd_iterations** *(int)*: Maximal number of iterations of U matrix elements removal
        and renormalisation in iterative SVD cleaning of dominant BPMs.
        This is also equal to maximal number of BPMs removed per SVD mode.

        Flags: **--num_svd_iterations**
        Default: ``3``
      - **opposite_direction**: If present, beam in the opposite direction to model
        is assumed for resynchronisation of BPMs.

        Flags: **--opposite_direction**
        Action: ``store_true``
      - **peak_to_peak** *(float)*: Peak to peak amplitude cut. This removes BPMs,
        where abs(max(turn values) - min(turn values)) <= threshold.

        Flags: **--peak_to_peak**
        Default: ``1e-08``
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
        Choices: ``('rectangle', 'welch', 'triangle', 'hann', 'hamming', 'nuttal3', 'nuttal4')``
        Default: ``hann``
      - **resonances** *(int)*: Maximum magnet order of resonance lines to calculate.

        Flags: **--resonances**
        Choices: ``(2 <= n <= 8)``
        Default: ``4``


    Optics Kwargs:
      - **files**: Files for analysis

        Flags: **--files**
        Required: ``True``
      - **outputdir**: Output directory

        Flags: **--outputdir**
        Required: ``True``
      - **calibrationdir** *(str)*: Path to calibration files directory.

        Flags: **--calibrationdir**
      - **chromatic_beating**: Calculate chromatic beatings: W, PHI and coupling

        Flags: **--chromatic_beating**
        Action: ``store_true``
      -  **compensation** *(str)*: Mode of compensation for the analysis after driven beam excitation.

        Flags: **-compensation**
        Choices: ``("model", "equation", "none")``
        Default: ``model``
      - **coupling_method** *(int)*: Coupling analysis option: disabled, 1 BPM or 2 BPMs method

        Flags: **--coupling_method**
        Choices: ``(0, 1, 2)``
        Default: ``2``
      - **coupling_pairing**: Pairing mode for 2 BPM coupling method. If 0 is given, omc3 
        will try to determine the best candidate. If a number n>=1 is given, then some BPMs are 
        skipped and the n-th following BPM downstream is used for the pairing.

        Flags: **--coupling_pairing**
        Choices: ``(0, n>=1)``
        Default: ``0``.
      - **nonlinear**: Calculate higher order RDTs or CRDT

        Flags: **--nonlinear**
        Choices: ``(rdt, crdt)``
        Default: ``None``
      - **rdt_magnet_order**: Maximum magnet order for RDTs calculation if --nonlinear is given

        Flags: **--rdt_magnet_order**
        Choices: ``(2 <= n <= 8)``
        Default: ``4``
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

      - **three_d_excitation**: Use 3D kicks to calculate dispersion.
        Action: ``store_true``

      - **isolation_forest**: If present, remove outlying BPMs with isolation forest.

        Flags: **--isolation_forest**
        Action: ``store_true``

      - **second_order_dispersion**: If present, calculate second order dispersion.

        Flags: **--second_order_dispersion**
        Action: ``store_true``

      - **union**: If present, the phase advances are calculate for union of BPMs
        with at least 3 valid measurements, instead of intersection .

        Flags: **--union**
        Action: ``store_true``

      - **analyse_dpp** *(float)*: Filter files to analyse by this value 
        (in analysis for tune, phase, rdt and crdt)..

        Flags: **--analyse_dpp**
        Default: ``0``


    Accelerator Kwargs:
      - **accel**: Choose the accelerator to use. More details can be found in omc3/model/manager.py

        Flags: **--accel**
        Required: ``True``
      - **model_dir**: Model directory, specify if ``--model`` option is not used.

        Flags: **--model_dir**

      - For the rest, please see get_parameters() methods in child Accelerator classes,
        which are declared in ``omc3/model/accelerators/*.py``.
    """
    if not opt.harpy and not opt.optics:
        raise SystemError("No module has been chosen.")
    if not rest:
        raise SystemError("No input has been set.")
    harpy_opt, optics_opt, accel_opt = _get_suboptions(opt, rest)
    _write_config_file(harpy_opt, optics_opt, accel_opt)
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
            harpy_opt.outputdir = join(harpy_opt.outputdir, LINFILES_SUBFOLDER)
            if harpy_opt.model is not None:
                rest = add_to_arguments(rest, entry_params={"model_dir": {"flags": "--model_dir"}},
                                        model_dir=dirname(abspath(harpy_opt.model)))
    else:
        harpy_opt = None

    if opt.optics:
        optics_opt, rest = _optics_entrypoint(rest)
        accel_opt = manager.get_parsed_opt(rest)
        optics_opt.accelerator = manager.get_accelerator(rest)
        if not optics_opt.accelerator.excitation and optics_opt.compensation != "none":
            raise AttributeError("Compensation requested and no driven model was provided.")
    else:
        optics_opt = None
        accel_opt = None
    return harpy_opt, optics_opt, accel_opt


def _write_config_file(harpy_opt, optics_opt, accelerator_opt):
    """Write the parsed options into a config file for later use."""
    all_opt = {}
    if harpy_opt is not None:
        all_opt["harpy"] = True
        all_opt.update(sorted(harpy_opt.items()))

    if optics_opt is not None:
        optics_opt = dict(sorted(optics_opt.items()))
        optics_opt.pop('accelerator')

        all_opt["optics"] = True
        all_opt.update(optics_opt)
        all_opt.update(sorted(accelerator_opt.items()))

    out_dir = all_opt["outputdir"]
    file_name = DEFAULT_CONFIG_FILENAME.format(time=datetime.now(timezone.utc).strftime(formats.TIME))
    iotools.create_dirs(out_dir)

    save_options_to_config(os.path.join(out_dir, file_name), all_opt)


def _run_harpy(harpy_options):
    iotools.create_dirs(harpy_options.outputdir)
    with timeit(lambda spanned: LOGGER.info(f"Total time for Harpy: {spanned}")):
        lins = []
        all_options = _replicate_harpy_options_per_file(harpy_options)
        tbt_datas = [(tbt.read_tbt(option.files, datatype=option.tbt_datatype), option) for option in all_options]
        for tbt_data, option in tbt_datas:
            lins.extend([handler.run_per_bunch(bunch_data, bunch_options)
                         for bunch_data, bunch_options in _add_suffix_and_iter_bunches(tbt_data, option)])
    return lins


def _replicate_harpy_options_per_file(options):
    list_of_options = []
    for input_file in options.files:
        new_options = deepcopy(options)
        new_options.files = input_file
        list_of_options.append(new_options)
    return list_of_options


def _add_suffix_and_iter_bunches(tbt_data: tbt.TbtData, options: DotDict
    ) -> Generator[tuple[tbt.TbtData, DotDict], None, None]:
    # hint: options.files is now a single file because of _replicate_harpy_options_per_file
    # it is also only used here to define the output name, as the tbt-data is already loaded.

    dir_name = dirname(options.files)
    file_name = basename(options.files)
    suffix = options.suffix or ""

    # Single bunch ---
    if tbt_data.nbunches == 1:
        if suffix:
            options.files = join(dir_name, f"{file_name}{suffix}")
        yield tbt_data, options
        return

    # Multibunch ---
    if options.bunch_ids is not None:
        unknown_bunches = set(options.bunch_ids) - set(tbt_data.bunch_ids)
        if unknown_bunches:
            LOGGER.warning(
                f"Bunch IDs {unknown_bunches} not present in multi-bunch file {options.files}."
            )

    for index in range(tbt_data.nbunches):
        bunch_id = tbt_data.bunch_ids[index]
        if options.bunch_ids is not None and bunch_id not in options.bunch_ids:
            continue

        new_options = deepcopy(options)
        bunch_id_str = f"_bunchID{bunch_id}"
        new_options.files = join(dir_name, f"{file_name}{bunch_id_str}{suffix}")
        yield (
            tbt.TbtData([tbt_data.matrices[index]], tbt_data.date, [bunch_id], tbt_data.nturns), 
            new_options
        )


def _measure_optics(lins, optics_opt):
    if len(lins) == 0:
        lins = optics_opt.files
    inputs = InputFiles(lins, optics_opt)
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
    if options.svd_dominance_limit <= 0.0:
        raise AttributeError("SVD dominance limit should be positive")
    if options.bad_bpms is None:
        options.bad_bpms = []
    if options.wrong_polarity_bpms is None:
        options.wrong_polarity_bpms = []
    if options.is_free_kick:
        options.window = "rectangle"
    if not 2 <= options.resonances <= 8:
        raise AttributeError("The magnet order for resonance lines calculation should be between 2 and 8 (inclusive).")

    return options, rest


def harpy_params():
    params = EntryPointParameters()
    params.add_parameter(name="files", required=True, nargs='+', help="TbT files to analyse")
    params.add_parameter(name="outputdir", required=True, help="Output directory.")
    params.add_parameter(name="suffix", type=str, help="User-defined suffix for output filenames.")
    params.add_parameter(name="model", help="Model for BPM locations")
    params.add_parameter(name="unit", type=str, default=HARPY_DEFAULTS["unit"],
                         choices=("m", "cm", "mm", "um"),
                         help="A unit of TbT BPM orbit data. All cuts and output are in 'm'.")
    params.add_parameter(name="turns", type=int, nargs=2, default=HARPY_DEFAULTS["turns"],
                         help="Turn index to start and first turn index to be ignored.")
    params.add_parameter(name="bunch_ids", type=int, nargs="+",
                         help="Bunches to process in multi-bunch file. "
                         "If not specified, all bunches are processed.")
    params.add_parameter(name="to_write", nargs='*', default=HARPY_DEFAULTS["to_write"],
                         choices=('lin', 'spectra', 'full_spectra', 'bpm_summary'),
                         help="Choose the type of output.")
    params.add_parameter(name="tbt_datatype", default=HARPY_DEFAULTS["tbt_datatype"],
                         choices=list(tbt.io.TBT_MODULES.keys()),
                         help="Choose the datatype from which to import. ")

    # Cleaning parameters
    params.add_parameter(name="clean", action="store_true",
                         help="If present, the data are first cleaned.")
    params.add_parameter(name="sing_val", type=int, default=HARPY_DEFAULTS["sing_val"],
                         help="Keep this amount of largest singular values.")
    params.add_parameter(name="peak_to_peak", type=float, default=HARPY_DEFAULTS["peak_to_peak"],
                         help="Peak to peak amplitude cut. This removes BPMs, "
                              "where abs(max(turn values) - min(turn values)) <= threshold.")
    params.add_parameter(name="max_peak", type=float, default=HARPY_DEFAULTS["max_peak"],
                         help="Removes BPMs where the maximum orbit > limit.")
    params.add_parameter(name="svd_dominance_limit", type=float,
                         default=HARPY_DEFAULTS["svd_dominance_limit"],
                         help="Limit for single BPM dominating a mode.")
    params.add_parameter(name="num_svd_iterations", type=int,
                         default=HARPY_DEFAULTS["num_svd_iterations"],
                         help="Maximal number of iterations of U matrix elements removal "
                              "and renormalisation in iterative SVD cleaning of dominant BPMs."
                              " This is also equal to maximal number of BPMs removed per SVD mode.")
    params.add_parameter(name="bad_bpms", nargs='*', help="Bad BPMs to clean.")
    params.add_parameter(name="wrong_polarity_bpms", nargs='*',
                         help="BPMs with swapped polarity in both planes.")
    params.add_parameter(name="keep_exact_zeros", action="store_true",
                         help="If present, will not remove BPMs with exact zeros in TbT data.")
    params.add_parameter(name="first_bpm", type=str,
                         help="First BPM in the measurement. "
                              "Used to resynchronise the TbT data with model.")
    params.add_parameter(name="opposite_direction", action="store_true",
                         help="If present, beam in the opposite direction to model"
                              " is assumed for resynchronisation of BPMs.")

    # Harmonic analysis parameters
    params.add_parameter(name="tunes", type=float, nargs=3,
                         help="Guess for the main tunes [x, y, z]. Tunez is disabled when set to 0")
    params.add_parameter(name="nattunes", type=float, nargs=3,
                         help="Guess for the natural tunes (x, y, z).  Disabled when set to 0.")
    params.add_parameter(name="natdeltas", type=float, nargs=3,
                         help="Guess for the offsets of natural tunes from the driven tunes"
                              " (x, y, z). Disabled when set to 0.")
    params.add_parameter(name="autotunes", type=str, choices=("all", "transverse"),
                         help="The main tunes are guessed as "
                              "the strongest line in SV^T matrix frequency spectrum: "
                              "Synchrotron tune below ~0.03, betatron tunes above ~0.03.")
    params.add_parameter(name="tune_clean_limit", type=float,
                         default=HARPY_DEFAULTS["tune_clean_limit"],
                         help="The tune cleaning wont remove BPMs because of measured tune outliers"
                              " closer to the average tune than this limit.")
    params.add_parameter(name="tolerance", type=float,
                         default=HARPY_DEFAULTS["tolerance"],
                         help="Tolerance specifying an interval in frequency domain, where to look "
                              "for the tunes.")
    params.add_parameter(name="is_free_kick", action="store_true",
                         help="If present, it will perform the free kick phase correction")
    params.add_parameter(name="window", type=str, default=HARPY_DEFAULTS["window"],
                         choices=("rectangle", "hann", "triangle", "welch", "hamming", "nuttal3",
                                  "nuttal4"),
                         help="Windowing function to be used for frequency analysis.")
    params.add_parameter(name="turn_bits", type=int, default=HARPY_DEFAULTS["turn_bits"],
                         help="Number (frequency, complex coefficient) pairs in the calculation"
                              " is 2 ** turn_bits, i.e. the difference between "
                              "two neighbouring frequencies is 2 ** (- turn_bits - 1).")
    params.add_parameter(name="output_bits", type=int, default=HARPY_DEFAULTS["output_bits"],
                         help="Number (frequency, complex coefficient) pairs in the output "
                              "is up to 2 ** output_bits (maximal in case full spectra is output). "
                              "There is one pair (with maximal amplitude of complex coefficient) "
                              "per interval of size 2 ** (- output_bits - 1).")
    params.add_parameter(name="resonances", type=int, default=HARPY_DEFAULTS["resonances"],
                        help="Maximum magnet order of resonance lines to calculate.")
    return params


def _optics_entrypoint(params):
    options, rest = EntryPoint(optics_params(), strict=False).parse(params)
    
    if "rdt" in options.nonlinear and not 2 <= options.rdt_magnet_order <= 8:
        raise AttributeError("The magnet order for RDT calculation should be between 2 and 8 (inclusive).")

    return options, rest


def optics_params():
    params = EntryPointParameters()
    params.add_parameter(name="files", required=True, nargs='+',
                         help="Files for analysis")
    params.add_parameter(name="outputdir", required=True,
                         help="Output directory")
    params.add_parameter(name="calibrationdir", type=str,
                         help="Path to calibration files directory.")
    params.add_parameter(name="coupling_method", type=int,
                         choices=(0, 1, 2), default=OPTICS_DEFAULTS["coupling_method"],
                         help="Analysis option for coupling: disabled, 1 BPM or 2 BPMs method")
    params.add_parameter(name="coupling_pairing", type=int,
                         default=OPTICS_DEFAULTS["coupling_pairing"],
                         help="Pairing mode for 2 BPM coupling method. If 0 is given, omc3 will try to "
                              "determine the best candidate. If a number n>=1 is given, then some BPMs are skipped "
                              "and the n-th following BPM downstream is used for the pairing.")
    params.add_parameter(name="range_of_bpms", type=int,
                         choices=(5, 7, 9, 11, 13, 15),  default=OPTICS_DEFAULTS["range_of_bpms"],
                         help="Range of BPMs for beta from phase calculation")
    params.add_parameter(name="union", action="store_true",
                         help="If present, the phase advances are calculate for union of BPMs "
                              "with at least 3 valid measurements, instead of intersection .")
    params.add_parameter(name="nonlinear", nargs='*', default=[],
                         choices=('rdt', 'crdt'),
                         help="Choose which rdt analysis is conducted.")
    params.add_parameter(name="rdt_magnet_order", type=int, default=OPTICS_DEFAULTS["rdt_magnet_order"],
                         help="Maximum magnet order for the RDT calculation.")
    params.add_parameter(name="three_bpm_method", action="store_true",
                         help="Use 3 BPM method in beta from phase")
    params.add_parameter(name="only_coupling", action="store_true", help="Calculate only coupling. ")
    params.add_parameter(name="compensation", type=str, default=OPTICS_DEFAULTS["compensation"],
                         choices=phase.CompensationMode.all(),
                         help="Mode of compensation for the analysis after driven beam excitation")
    params.add_parameter(name="three_d_excitation", action="store_true",
                         help="Use 3D kicks to calculate dispersion")
    params.add_parameter(name="isolation_forest", action="store_true",
                         help="Remove outlying BPMs with isolation forest")
    params.add_parameter(name="second_order_dispersion", action="store_true",
                         help="Calculate second order dispersion")
    params.add_parameter(name="chromatic_beating", action="store_true",
                         help="Calculate chromatic beatings: W, PHI and coupling")
    params.add_parameter(name="analyse_dpp", type=iotools.OptionalFloat, default=OPTICS_DEFAULTS["analyse_dpp"],
                        help="Filter files to analyse by this value (in analysis for tune, phase, rdt and crdt).")
    return params


HARPY_DEFAULTS = {
    "turns": [0, 50000],
    "unit": "m",
    "sing_val": 12,
    "peak_to_peak": 1e-8,
    "max_peak": 0.02,
    "svd_dominance_limit": 0.925,
    "num_svd_iterations": 3,
    "tolerance": 0.01,
    "tune_clean_limit": 1e-5,
    "window": "hann",
    "turn_bits": 20,
    "output_bits": 12,
    "to_write": ["lin", "bpm_summary"],
    "tbt_datatype": "lhc",
    "resonances": 4,
}

OPTICS_DEFAULTS = {
        "coupling_method": 2,
        "coupling_pairing": 0,
        "range_of_bpms": 11,
        "compensation": "model",
        "rdt_magnet_order": 4,
        "analyse_dpp": 0,
}


if __name__ == "__main__":
    hole_in_one_entrypoint()
