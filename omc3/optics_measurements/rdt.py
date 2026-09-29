"""
Resonance Driving Terms
-----------------------

This module contains RDT calculations related functionality of ``optics_measurements``.
It provides functions to compute global resonance driving terms **f_jklm**.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from copy import deepcopy
from itertools import product
from math import comb, factorial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import tfs
from optics_functions.rdt import get_all_to_order, jklm2str
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.sparse import diags

from omc3.definitions.constants import PLANES
from omc3.harpy.constants import COL_AMP, COL_PHASE
from omc3.optics_measurements.constants import (
    AMPLITUDE,
    COUNT,
    CROSS_FRACTION,
    LOCATION,
    DECOMP,
    ERR,
    EXT,
    FREE,
    IMAG,
    LINE_RESIDUAL,
    MEASUREMENT,
    N_CONTRIBUTORS,
    NAME,
    NAME2,
    PHASE,
    PHASE_ADV,
    RDT_FOLDER,
    REAL,
    S2,
    S,
)
from omc3.optics_measurements.data_models import (
    InputFiles,
    check_and_warn_about_offmomentum_data,
    filter_for_dpp,
)
from omc3.optics_measurements.phase import COMPENSATED, UNCOMPENSATED, CompensationMode, PhaseDict
from omc3.optics_measurements.toolbox import df_diff
from omc3.utils import iotools, logging_tools, stats

if TYPE_CHECKING:
    from generic_parser import DotDict
    from numpy.typing import ArrayLike

    from omc3.optics_measurements.tune import TuneDict

NBPMS_FOR_90 = 3
LOGGER = logging_tools.get_logger(__name__)

RDTTuple = tuple[int, int, int, int]
LineTuple = tuple[int, int, int]


def _generate_plane_rdts(order: int) -> tuple[dict[str, list[RDTTuple]], dict[str, list[RDTTuple]]]:
    """
    This helper function generates two dictionnaries representing on what plane(s)
    a RDT can be seen and which tune it is a multiple of.

    For a given j+k+l+m = n multiple order, a line can be seen:
      - on the horizontal axis if j != 0, at:
          (1 - j + k) Qx + (m - l) Qy
      - on the vertical axis if l != 0, at:
          (k - j) Qx + (1 - l + m) Qy

    Reference equations are (3.27) and (3.28) in
    [Andrea Franchi's Thesis](https://repository.gsi.de/record/55413/files/GSI-Diss-2006-07.pdf)
    """
    # Get all the valid RDTs up to a certain order
    all_rdts = get_all_to_order(order)

    single_plane = {"X": [], "Y": []}
    double_plane = {"X": [], "Y": []}
    # Iterate through our RDTs and classify them depending on what plane they act
    for j, k, l, m in all_rdts:  # noqa: E741
        if j == 0 and l == 0:  # the RDT can't be seen on any plane
            continue
        if l + m == 0 and j != 0:  # The line where the RDT is seen is a multiple of the Qx line
            single_plane["X"].append((j, k, l, m))  # e.g. f1400, f3000, f4000
        elif j + k == 0 and l != 0:  # same, but for the Qy line
            single_plane["Y"].append((j, k, l, m))  # e.g. f0030,f0040
        elif j == 0:  # The RDT can only be seen on the vertical plane and uses both Qx and Qy
            double_plane["Y"].append((j, k, l, m))  # e.g. f0111, f0120
        elif l == 0:  # same, but for the horizontal plane
            double_plane["X"].append((j, k, l, m))  # e.g. f1001, f1002
        else:  # The RDT can be seen on both planes and is a multiple of both Qx and Qy
            double_plane["X"].append((j, k, l, m))  # e.g. f1020, f1120
            double_plane["Y"].append((j, k, l, m))

    return single_plane, double_plane


def calculate(
    measure_input: DotDict,
    input_files: InputFiles,
    tunes: TuneDict,
    phases: dict[str, dict[str, PhaseDict]],
    invariants: dict[str, pd.DataFrame],
    header: dict,
) -> None:
    """
    Computes the RDTs for the given input files and settings up to the magnet
    order given in the inputs, and writes the results to file.

    Args:
        measure_input: `OpticsInput` object containing analysis settings.
        input_files: `InputFiles` object containing frequency spectra files (linx/y).
        tunes: `TuneDict` object mapping planes to tunes, as given by
            :func:`omc3.optics_measurements.tune.calculate`.
        phases: dictionary mapping planes to `PhaseDict` for compensated and uncompensated cases.
        invariants: dictionary mapping planes to dataframes
            of actions/errors per kick, e.g. from :func:`omc3.optics_measurements.kick.calculate`.
        header: headers to include to the written result files.
    """
    LOGGER.info("Start of RDT analysis")
    meas_input = deepcopy(measure_input)
    meas_input["compensation"] = CompensationMode.NONE
    LOGGER.info(f"Calculating RDTs up to magnet order {meas_input['rdt_magnet_order']}")

    dpp_value = meas_input.analyse_dpp
    if dpp_value is None:
        for plane in PLANES:
            check_and_warn_about_offmomentum_data(input_files, plane, id_="RDT calculation")
    else:
        invariants = filter_for_dpp(invariants, input_files, dpp_value)

    single_plane_rdts, double_plane_rdts = _generate_plane_rdts(meas_input["rdt_magnet_order"])
    single, double = "Single", "Double"

    deltas = None
    if meas_input.get("rdt_delta_plus_correction"):
        deltas = _delta_pm_at_reference(meas_input, input_files, phases, tunes, invariants, dpp_value)
        if deltas is None:
            LOGGER.warning(
                "rdt_delta_plus_correction requested but the driven tune lines could not be "
                "extracted: falling back to the plain forced-to-free conversion."
            )

    for rdts, plane_str in ((single_plane_rdts, single), (double_plane_rdts, double)):
        for plane in PLANES:
            bpm_names = input_files.bpms(
                plane=plane if (plane_str == single) else None, dpp_value=dpp_value
            )

            phases_for_rdts = _best_90_degree_phases(meas_input, bpm_names, phases, tunes, plane)
            LOGGER.info(
                f"{plane_str} Plane RDTs: "
                f"Average phase advance between BPM pairs: {phases_for_rdts.loc[:, MEASUREMENT].mean()}"
            )

            for line, rdts_on_line in _group_rdts_by_line(rdts[plane], plane).items():
                if len(rdts_on_line) > 1:
                    LOGGER.info(
                        f"Line {line} in plane {plane} is shared by "
                        f"{', '.join(jklm2str(*rdt) for rdt in rdts_on_line)}: decomposing."
                    )
                dfs = None
                with _check_amp_error(rdts_on_line):
                    dfs = _process_line(
                        meas_input,
                        input_files,
                        phases_for_rdts,
                        invariants,
                        tunes,
                        plane,
                        line,
                        rdts_on_line,
                        dpp_value,
                        deltas,
                    )
                if dfs is None:
                    continue
                for rdt, df in dfs.items():
                    write(df, add_freq_to_header(header, plane, rdt, tunes), meas_input, plane, rdt)


def write(df: pd.DataFrame, header: dict[str, Any], meas_input: DotDict, plane: str, rdt: RDTTuple) -> None:
    outputdir = Path(meas_input.outputdir) / RDT_FOLDER / _rdt_to_order_and_type(rdt)
    iotools.create_dirs(outputdir)
    tfs.write(outputdir / f"f{_rdt_to_str(rdt)}_{plane.lower()}{EXT}", df, header, save_index=NAME)


@contextmanager
def _check_amp_error(rdts: list[RDTTuple]):
    """Context manager to catch a specific type of ValueError, regarding the AMP_ column not found in the output,
    as raised by get_line_sign_and_suffix. All RDTs sharing a line fail together: they are read
    from the very same spectral line, so a missing AMP column affects every one of them identically.

    Alternatively, we could just raise a custom Error Type?"""
    try:
        yield
    except ValueError as e:
        error_str = str(e)
        message = f"RDT calculation failed for {', '.join(jklm2str(*rdt) for rdt in rdts)}."

        if "column AMP" not in error_str:  # raise unexpected Value errors
            raise ValueError(message) from e

        LOGGER.warning(f"{message}: {error_str}")


def _rdt_to_str(rdt: RDTTuple) -> str:
    j, k, l, m = rdt  # noqa: E741
    return f"{j}{k}{l}{m}"


def _rdt_to_order_and_type(rdt: RDTTuple) -> str:
    j, k, l, m = rdt  # noqa: E741
    rdt_type = "normal" if (l + m) % 2 == 0 else "skew"
    orders = {
        1: "dipole",
        2: "quadrupole",
        3: "sextupole",
        4: "octupole",
        5: "decapole",
        6: "dodecapole",
        7: "tetradecapole",
        8: "hexadecapole",
    }
    return f"{rdt_type}_{orders[j + k + l + m]}"


def _best_90_degree_phases(
    meas_input: DotDict,
    bpm_names: pd.Index,
    phases: dict[str, dict[str, PhaseDict]],
    tunes: TuneDict,
    plane: str,
) -> pd.DataFrame:
    """
    Finds BPM pairs with phase advance closest to 90 degrees for RDT measurement.

    Args:
        meas_input: `OpticsInput` object containing analysis settings.
        bpm_names: index of BPM names to consider.
        phases: nested dictionary of phase advance data, keyed by plane and
            compensation mode.
        tunes: `TuneDict` with tune values per plane.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        A `DataFrame` indexed by BPM name with columns for the partner BPM name,
        measured phase advance to the partner, its error, and longitudinal position.
    """
    phase_dict: PhaseDict = phases[plane][UNCOMPENSATED]
    if phase_dict is None:
        phase_dict = phases[plane][COMPENSATED]

    # Remove BPMs that are not in model
    bpm_names = phase_dict[MEASUREMENT].index.intersection(bpm_names)
    filtered = phase_dict[MEASUREMENT].loc[bpm_names, bpm_names]

    # fmt: off
    phase_meas = pd.concat((filtered % 1, (filtered.iloc[:, :NBPMS_FOR_90] + tunes[plane]["Q"]) % 1), axis=1)
    second_bmps = np.abs(phase_meas * _get_n_upper_diagonals(NBPMS_FOR_90, phase_meas.shape) - 0.25).idxmin(axis=1)
    filtered.iloc[-NBPMS_FOR_90:, :NBPMS_FOR_90] = (filtered.iloc[-NBPMS_FOR_90:, :NBPMS_FOR_90] + tunes[plane]["Q"]) % 1
    # fmt: on

    # get the pairs zip(bpm_names, second_bpms)
    filtered[NAME2] = second_bmps
    filtered[MEASUREMENT] = np.diag(filtered.loc[bpm_names, second_bmps])
    filtered[f"{ERR}{MEASUREMENT}"] = np.diag(
        phase_dict[f"{ERR}{MEASUREMENT}"].loc[bpm_names, second_bmps]
    )

    # Merge final dataframe
    model: pd.DataFrame = meas_input.accelerator.model
    return pd.merge(
        filtered.loc[:, [NAME2, MEASUREMENT, f"{ERR}{MEASUREMENT}"]],
        model.loc[:, [S]],
        how="inner",
        left_index=True,
        right_index=True,
    )


def _get_n_upper_diagonals(n: int, shape: tuple[int, int]) -> np.ndarray:
    return diags(np.ones((n, shape[0])), np.arange(n) + 1, shape=shape).toarray()


def _determine_line(rdt: RDTTuple, plane: str) -> LineTuple:
    j, k, l, m = rdt  # noqa: E741
    lines = {"X": (1 - j + k, m - l, 0), "Y": (k - j, 1 - l + m, 0)}
    return lines[plane]


def _group_rdts_by_line(rdts: list[RDTTuple], plane: str) -> dict[LineTuple, list[RDTTuple]]:
    """
    Groups RDTs that produce the same spectral line, i.e. share the same resonance
    (j - k, l - m): a line is generically fed by several multipole orders at once, not just
    the single one it is conventionally named after (e.g. f3000 and f4100 both drive the
    -2Qx line). Groups are sorted by ascending magnet order, lowest (usually dominant) first.
    """
    groups: dict[LineTuple, list[RDTTuple]] = {}
    for rdt in rdts:
        groups.setdefault(_determine_line(rdt, plane), []).append(rdt)
    return {line: sorted(group, key=sum) for line, group in groups.items()}


def add_freq_to_header(
    header: dict[str, Any], plane: str, rdt: RDTTuple, tunes: TuneDict | None = None
) -> dict[str, Any]:
    """
    Adds the (folded) frequency of the RDT's spectral line: ``FREQ`` at the measured tunes, i.e.
    the frequency the line was actually extracted at (the drive tunes under AC-dipole
    excitation), and ``FREQFREE`` at the free tunes (``Q1``/``Q2`` of the header).
    """
    def fold(q1, q2):
        freq = np.mod(np.array(line) @ np.array([q1, q2, 0]), 1)
        return freq if freq <= 0.5 else 1 - freq

    mod_header = header.copy()
    line = _determine_line(rdt, plane)
    mod_header["FREQFREE"] = fold(header["Q1"], header["Q2"])
    mod_header["FREQ"] = (fold(tunes["X"]["Q"], tunes["Y"]["Q"]) if tunes is not None
                          else mod_header["FREQFREE"])
    return mod_header


def _extract_line_raw(
    meas_input: DotDict,
    input_files: InputFiles,
    phase_data: pd.DataFrame,
    plane: str,
    line: LineTuple,
    dpp_value: float | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts, once per shared spectral line, the BPM-pair bookkeeping and the raw complex line
    coefficient (amplitude and phase, and their errors), *before* the RDT-specific phase
    realignment (:func:`_calculate_rdt_phases_from_line_phases`) that removes the trivial
    phase-advance term so the result no longer depends on the BPM's position.

    This raw, still position-dependent value is exactly what :func:`_extract_line` builds on for
    the RDT phase, and also what the delta_+ correction (:func:`_delta_pm_at_reference`) needs
    for the driven tune line itself: unlike an RDT (an intrinsic, position-independent machine
    constant once realigned), the tune line genuinely *is* position-dependent (it is the driven
    orbit itself), so it must be kept in its raw, BPM-local form.

    Args:
        meas_input: `OpticsInput` object containing analysis settings.
        input_files: `InputFiles` object containing frequency spectra files (linx/y).
        phase_data: BPM-pair data from :func:`_best_90_degree_phases`.
        plane: marking the horizontal or vertical plane, **X** or **Y**.
        line: the (nu_x, nu_y, 0) tuple identifying the spectral line, from :func:`_determine_line`.
        dpp_value: the off-momentum value to filter on, if any.

    Returns:
        A tuple `(df, line_amp, line_amp_e, line_phase, line_phase_e)`: `df` carries `S`/`COUNT`
        indexed by BPM name, and the arrays are `(n_bpms, n_kick_files)`.
    """
    df = pd.DataFrame(phase_data)
    second_bpms = df.loc[:, NAME2].to_numpy()
    df[S2] = df.loc[second_bpms, S].to_numpy()
    df[COUNT] = len(input_files.dpp_frames(plane, dpp_value=dpp_value))
    phase_sign, suffix = get_line_sign_and_suffix(line, input_files, plane)

    df_all_amps = input_files.joined_frame(plane, [f"{COL_AMP}{suffix}"], dpp_value=dpp_value)
    df_all_phases = phase_sign * input_files.joined_frame(
        plane, [f"{COL_PHASE}{suffix}"], dpp_value=dpp_value
    )

    comp_coeffs1 = to_complex(df_all_amps.loc[df.index, :], df_all_phases.loc[df.index, :])
    # Multiples of tunes needs to be added to phase at second BPM if that is in second turn
    comp_coeffs2 = to_complex(
        df_all_amps.loc[second_bpms, :],
        _add_tunes_if_in_second_turn(
            df, input_files, line, df_all_phases.loc[second_bpms, :].to_numpy(), dpp_value
        ),
    )

    # Get amplitude and phase of the line from linx/liny file
    line_amp, line_phase, line_amp_e, line_phase_e = complex_secondary_lines(  # TODO use the errors
        df.loc[:, MEASUREMENT].to_numpy()[:, np.newaxis] * meas_input.accelerator.beam_direction,
        df.loc[:, f"{ERR}{MEASUREMENT}"].to_numpy()[:, np.newaxis],
        comp_coeffs1,
        comp_coeffs2,
    )
    return df, line_amp, line_amp_e, line_phase, line_phase_e


def _extract_line(
    meas_input: DotDict,
    input_files: InputFiles,
    phase_data: pd.DataFrame,
    plane: str,
    line: LineTuple,
    dpp_value: float | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts, once per shared spectral line, everything that every RDT contributing to that
    line has in common: the BPM-pair bookkeeping, the raw complex line coefficient (amplitude
    and its error) and the RDT phase (the line phase realigned to remove the trivial
    phase-advance term, so that it no longer depends on the BPM's position). Individual RDTs
    sharing the line only differ in how this shared data is normalized by their own action
    powers (see :func:`get_linearized_problem`), which is why it previously had to be
    recomputed identically for every one of them.

    Args:
        meas_input: `OpticsInput` object containing analysis settings.
        input_files: `InputFiles` object containing frequency spectra files (linx/y).
        phase_data: BPM-pair data from :func:`_best_90_degree_phases`.
        plane: marking the horizontal or vertical plane, **X** or **Y**.
        line: the (nu_x, nu_y, 0) tuple identifying the spectral line, from :func:`_determine_line`.
        dpp_value: the off-momentum value to filter on, if any.

    Returns:
        A tuple `(df, line_amp, line_amp_e, rdt_phases_per_file)`: `df` carries `S`/`COUNT`
        indexed by BPM name, and the arrays are `(n_bpms, n_kick_files)`.
    """
    df, line_amp, line_amp_e, line_phase, _line_phase_e = _extract_line_raw(
        meas_input, input_files, phase_data, plane, line, dpp_value
    )
    rdt_phases_per_file = _calculate_rdt_phases_from_line_phases(
        df, input_files, line, line_phase, dpp_value
    )
    return df, line_amp, line_amp_e, rdt_phases_per_file


def _process_line(
    meas_input: DotDict,
    input_files: InputFiles,
    phase_data: pd.DataFrame,
    invariants: dict[str, pd.DataFrame],
    tunes: TuneDict,
    plane: str,
    line: LineTuple,
    rdts_on_line: list[RDTTuple],
    dpp_value: float | None,
    deltas: dict[str, tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> dict[RDTTuple, pd.DataFrame]:
    """
    Computes, for every RDT sharing the spectral `line`, the naive result: the whole line
    read as if it were fed by that single multipole order alone (the historical, and still
    default, behaviour of this module). Two versions of it are reported:

    - **forced**: exactly the quantity extracted from the measured spectral line, with no
      correction for AC-dipole excitation. This equals the free RDT f_jklm when the motion is
      free (no excitation), and the forced RDT f'_jklm when an AC dipole is used -- it was, in
      both cases, already what this module used to return under its plain, unsuffixed column
      names, which are kept unchanged here for backwards compatibility.
    - **free** (columns suffixed `FREE`): always the free-machine RDT f_jklm, obtained by
      applying the validated forced-to-free conversion (:func:`_forced_to_free_factor`) to the
      forced quantity above. That conversion is the identity when there is no AC-dipole
      excitation, so the `FREE` columns are then identical to the plain ones.

    When `deltas` is given (i.e. `--rdt_delta_plus_correction` was requested and the driven tune
    lines could be extracted), the measured complex line is first corrected for the delta_+
    cross-term contamination (:func:`_delta_plus_correction`) before either the naive fit above
    or the decomposition below sees it -- both benefit transparently, since both start from the
    same raw complex line.

    When more than one RDT shares the line, an additional decomposition
    (:func:`_decompose_line`) separates the individual multipole-order contributions; the
    naive, whole-line result described above is always computed and kept regardless.

    Args:
        meas_input: `OpticsInput` object containing analysis settings.
        input_files: `InputFiles` object containing frequency spectra files (linx/y).
        phase_data: BPM-pair data from :func:`_best_90_degree_phases`.
        invariants: dictionary mapping planes to DataFrames of actions and errors per kick.
        tunes: `TuneDict` with measured/driven and free tunes per plane.
        plane: marking the horizontal or vertical plane, **X** or **Y**.
        line: the (nu_x, nu_y, 0) tuple identifying the spectral line shared by `rdts_on_line`.
        rdts_on_line: every RDTTuple sharing `line`, lowest magnet order first.
        dpp_value: the off-momentum value to filter on, if any.
        deltas: the driven tune-line amplitudes from :func:`_delta_pm_at_reference`, or `None`
            to skip the delta_+ correction entirely (the default).

    Returns:
        A dictionary mapping each RDT in `rdts_on_line` to its result `DataFrame`.
    """
    df, line_amp, _line_amp_e, line_phase, _line_phase_e = _extract_line_raw(
        meas_input, input_files, phase_data, plane, line, dpp_value
    )

    cross_fraction = None
    if deltas is not None:
        complex_line = line_amp * np.exp(2j * np.pi * line_phase)
        a_cross = _delta_plus_correction(
            meas_input, plane, line, tunes, deltas, complex_line.shape[1], df.index,
            meas_input["rdt_magnet_order"],
        )
        corrected = complex_line - a_cross
        with np.errstate(invalid="ignore", divide="ignore"):
            cross_fraction = np.where(
                np.abs(complex_line) > 0, np.abs(a_cross) / np.abs(complex_line), np.nan
            )
        line_amp = np.abs(corrected)
        line_phase = (np.angle(corrected) / (2 * np.pi)) % 1

    rdt_phases_per_file = _calculate_rdt_phases_from_line_phases(
        df, input_files, line, line_phase, dpp_value
    )
    rdt_angles = stats.circular_mean(rdt_phases_per_file, period=1, axis=1) % 1
    err_rdt_angles = stats.circular_error(rdt_phases_per_file, period=1, axis=1)
    conversion = _forced_to_free_factor(line, plane, tunes)
    sig_abs, sig_arg = _forced_to_free_factor_error(line, plane, tunes, meas_input)

    dfs = {}
    for rdt in rdts_on_line:
        rdt_df = df.loc[:, [S, COUNT]].copy()
        rdt_df[PHASE] = rdt_angles
        rdt_df[f"{ERR}{PHASE}"] = err_rdt_angles
        rdt_df[AMPLITUDE], rdt_df[f"{ERR}{AMPLITUDE}"] = _fit_rdt_amplitudes(
            invariants, line_amp, plane, rdt
        )
        rdt_df[REAL] = np.cos(2 * np.pi * rdt_angles) * rdt_df.loc[:, AMPLITUDE].to_numpy()
        rdt_df[IMAG] = np.sin(2 * np.pi * rdt_angles) * rdt_df.loc[:, AMPLITUDE].to_numpy()

        loc = None
        if meas_input.get("rdt_location_correction"):
            loc = _location_factor(meas_input, plane, line, rdt, tunes, df.index)
            if loc is None:
                LOGGER.warning(
                    f"Location correction for {_rdt_to_str(rdt)} skipped: the model carries no "
                    "sources for it or no AC-dipole element."
                )
        factor = conversion if loc is None else conversion / loc
        amp_free, err_amp_free, phase_free = _apply_forced_to_free(
            rdt_df.loc[:, AMPLITUDE].to_numpy(),
            rdt_df.loc[:, f"{ERR}{AMPLITUDE}"].to_numpy(),
            rdt_angles,
            factor,
        )
        rdt_df[f"{AMPLITUDE}{FREE}"] = amp_free
        # the conversion depends on the free tunes, known only to sigma(Q_F): propagate
        rdt_df[f"{ERR}{AMPLITUDE}{FREE}"] = np.sqrt(err_amp_free ** 2 + (amp_free * sig_abs) ** 2)
        rdt_df[f"{PHASE}{FREE}"] = phase_free
        rdt_df[f"{ERR}{PHASE}{FREE}"] = np.sqrt(err_rdt_angles ** 2 + (sig_arg / (2 * np.pi)) ** 2)
        rdt_df[f"{REAL}{FREE}"] = np.cos(2 * np.pi * phase_free) * amp_free
        rdt_df[f"{IMAG}{FREE}"] = np.sin(2 * np.pi * phase_free) * amp_free

        columns = [
            S, COUNT, AMPLITUDE, f"{ERR}{AMPLITUDE}", PHASE, f"{ERR}{PHASE}", REAL, IMAG,
            f"{AMPLITUDE}{FREE}", f"{ERR}{AMPLITUDE}{FREE}", f"{PHASE}{FREE}",
            f"{ERR}{PHASE}{FREE}", f"{REAL}{FREE}", f"{IMAG}{FREE}",
        ]
        if cross_fraction is not None:
            with np.errstate(invalid="ignore"):
                rdt_df[CROSS_FRACTION] = np.nanmean(cross_fraction, axis=1)
            columns.append(CROSS_FRACTION)
        if loc is not None:
            rdt_df[f"{LOCATION}{AMPLITUDE}"] = np.abs(loc)
            rdt_df[f"{LOCATION}{PHASE}"] = (np.angle(loc) / (2 * np.pi)) % 1
            columns += [f"{LOCATION}{AMPLITUDE}", f"{LOCATION}{PHASE}"]

        # in old files there was "EAMP" and "PHASE_STD"
        dfs[rdt] = rdt_df.loc[:, columns]

    if len(rdts_on_line) > 1:
        decomposition = _decompose_line(
            invariants, plane, line, rdts_on_line, line_amp, rdt_phases_per_file, df.index, conversion
        )
        for rdt, decomp_df in decomposition.items():
            dfs[rdt] = pd.concat([dfs[rdt], decomp_df], axis=1)

    return dfs


def _forced_to_free_factor(line: LineTuple, plane: str, tunes: TuneDict) -> complex:
    """
    Complex factor converting a forced (AC-dipole driven) RDT into the free-machine RDT, for
    the resonance identified by `line` = (nu_x, nu_y, 0) (see :func:`_determine_line`):
    ``f_free = f_forced * factor``.

    The RDT extracted from the measured spectral line is f' = h / (1 - exp(2i pi Theta_forced)),
    with ``Theta_forced = Q_self - nu_x * Q_Dx - nu_y * Q_Dy``, Q_self the *free* natural tune of
    the readout plane and Q_Dx, Q_Dy the (measured) AC-dipole drive tunes. The free RDT is
    f = h / (1 - exp(2i pi Theta_free)), with ``Theta_free`` using the free tunes throughout.
    Since both share the same numerator h (same sources, same amplitude referencing), their
    ratio reduces to

        factor = sin(pi Theta_forced) / sin(pi Theta_free) * exp(i pi (Theta_free - Theta_forced))

    i.e. the amplitude-only sine ratio found in the literature, together with the half-phase
    term it omits (essential for the RDT *phase*); this sign of the half-phase exponential has
    been validated against tracking.

    When there is no AC-dipole excitation, ``tunes[plane]["Q"] == tunes[plane]["QF"]`` in both
    planes, so ``Theta_forced == Theta_free`` and this factor is exactly `1` (a no-op).

    Args:
        line: the (nu_x, nu_y, 0) tuple identifying the spectral line.
        plane: marking the horizontal or vertical readout plane, **X** or **Y**.
        tunes: `TuneDict` with measured/driven (`Q`) and free (`QF`) tunes per plane.

    Returns:
        The complex forced-to-free conversion factor for this line.
    """
    nu_x, nu_y = line[0], line[1]
    qx_free, qy_free = tunes["X"]["QF"], tunes["Y"]["QF"]
    qdx, qdy = tunes["X"]["Q"], tunes["Y"]["Q"]
    q_self_free = tunes[plane]["QF"]

    theta_free = q_self_free - nu_x * qx_free - nu_y * qy_free
    theta_forced = q_self_free - nu_x * qdx - nu_y * qdy

    sin_free = np.sin(np.pi * theta_free)
    if sin_free == 0.0:
        return complex(np.nan, np.nan)  # exactly on the resonance: the conversion is undefined

    ratio = np.sin(np.pi * theta_forced) / sin_free
    half_phase = np.exp(1j * np.pi * (theta_free - theta_forced))
    return ratio * half_phase


def _forced_to_free_factor_error(
    line: LineTuple, plane: str, tunes: TuneDict, meas_input: DotDict | None = None
) -> tuple[float, float]:
    """
    Uncertainty of the forced-to-free factor from the uncertainty of the free tunes Q_F
    (the drive tunes are known precisely). Q_F is reconstructed as the measured driven tune
    shifted by the model's driven-to-free offset, so its uncertainty is dominated by how well
    the drive was placed relative to the real (amplitude-detuned) natural tune: a systematic
    `rdt_free_tune_error` (per plane) is added in quadrature to the statistical `QFERR`.
    The factor is very sensitive near a resonance (d ln|F|/dQ ~ pi cot(pi Theta)).

    Returns:
        `(sigma(|F|)/|F|, sigma(arg F))`, the latter in radians; `(0, 0)` without excitation.
    """
    sys_err = float((meas_input or {}).get("rdt_free_tune_error") or 0.0)
    sig = {p: np.hypot(tunes[p].get("QFERR", 0.0), sys_err) for p in PLANES}
    if all(tunes[p]["Q"] == tunes[p]["QF"] for p in PLANES) or not any(sig.values()):
        return 0.0, 0.0
    f0 = _forced_to_free_factor(line, plane, tunes)
    var_abs = var_arg = 0.0
    step = 1e-6
    for p in PLANES:
        if not sig[p]:
            continue
        shifted = deepcopy(tunes)
        shifted[p]["QF"] = tunes[p]["QF"] + step
        f1 = _forced_to_free_factor(line, plane, shifted)
        var_abs += ((np.abs(f1) - np.abs(f0)) / step / np.abs(f0) * sig[p]) ** 2
        var_arg += (np.angle(f1 / f0) / step * sig[p]) ** 2
    return float(np.sqrt(var_abs)), float(np.sqrt(var_arg))


def _apply_forced_to_free(
    amp: np.ndarray, err_amp: np.ndarray, phase: np.ndarray, factor: complex
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies the forced-to-free `factor` (:func:`_forced_to_free_factor`) to an amplitude/phase
    pair. The phase error is left untouched by the caller: the conversion factor is a
    deterministic function of the measured tunes, not an extra source of random noise on the
    fitted RDT phase.

    Returns:
        A tuple `(amp_free, err_amp_free, phase_free)`.
    """
    amp_free = amp * np.abs(factor)
    err_amp_free = err_amp * np.abs(factor)
    phase_free = (phase + np.angle(factor) / (2 * np.pi)) % 1
    return amp_free, err_amp_free, phase_free


def _decompose_line(
    invariants: dict[str, pd.DataFrame],
    plane: str,
    line: LineTuple,
    rdts_on_line: list[RDTTuple],
    line_amp: np.ndarray,
    rdt_phases_per_file: np.ndarray,
    index: pd.Index,
    conversion: complex,
) -> dict[RDTTuple, pd.DataFrame]:
    """
    Separates several multipole orders sharing one spectral line.

    The line's phase-aligned complex coefficient, ``line_amp * exp(2i pi rdt_phases_per_file)``
    (one value per BPM per kick file), is to first order the *sum* of every contributor's own
    line coefficient: each one scales with its own action monomial, :func:`get_linearized_problem`
    gives that monomial's design column (the very one the naive, single-order fit above already
    uses), and stacking one column per contributor turns the per-BPM data across kick files into
    a linear least-squares problem for the complex RDTs directly -- amplitude *and* phase fit
    jointly, unlike the naive fit's separate amplitude-only regression and phase circular-average.

    Two quantifications of how much to trust the result are added, per BPM:

    - a formal 1-sigma error per contributor, from the fit's residual variance and the design
      matrix's covariance (identical for every BPM, since the design only depends on the kick
      actions, not on the measured line);
    - ``LINERESIDUAL``, the fraction of the measured line left unexplained by the fitted sum of
      contributors (0 = perfectly explained; a large value points to missing contributors, e.g.
      unmodelled second-order feed-up, or to kick amplitudes that do not vary enough to separate
      the multipole orders -- the latter is also flagged as a condition-number warning below, and
      the number of contributors is kept in ``NCONTRIBUTORS`` for reference).

    Args:
        invariants: dictionary mapping planes to DataFrames of actions and errors per kick.
        plane: marking the horizontal or vertical plane, **X** or **Y**.
        line: the (nu_x, nu_y, 0) tuple identifying the shared spectral line.
        rdts_on_line: every RDTTuple sharing `line`, lowest magnet order first.
        line_amp: `(n_bpms, n_kick_files)` measured line amplitude, from :func:`_extract_line`.
        rdt_phases_per_file: `(n_bpms, n_kick_files)` RDT phase, from :func:`_extract_line`.
        index: BPM-name index to build the result DataFrames on.
        conversion: the forced-to-free factor for this line, from :func:`_forced_to_free_factor`.

    Returns:
        A dictionary mapping each RDT in `rdts_on_line` to a `DataFrame` of the
        decomposition-specific columns (forced and free amplitude/phase/errors, and
        `LINERESIDUAL`/`NCONTRIBUTORS`), meant to be merged with the naive per-RDT result.
    """
    design = np.column_stack(
        [get_linearized_problem(invariants, plane, rdt) for rdt in rdts_on_line]
    )  # (n_kicks, n_orders), real-valued
    complex_line = line_amp * np.exp(2j * np.pi * rdt_phases_per_file)  # (n_bpms, n_kicks)

    n_bpms, n_kicks = complex_line.shape
    n_orders = design.shape[1]
    dof = n_kicks - n_orders

    design_pinv = np.linalg.pinv(design)
    try:
        cov_unit = np.linalg.inv(design.T @ design)
        cond_number = np.linalg.cond(design)
    except np.linalg.LinAlgError:
        cov_unit = np.full((n_orders, n_orders), np.nan)
        cond_number = np.inf

    if not np.isfinite(cond_number) or cond_number > 1e8:
        LOGGER.warning(
            f"Line-sharing decomposition for line {line} in plane {plane} "
            f"({n_orders} contributors, {n_kicks} kick files) is ill-conditioned "
            f"(design-matrix condition number {cond_number:.3g}): the available kicks may not "
            "vary enough in amplitude (or plane) to separate the multipole orders reliably."
        )

    coeffs = np.full((n_bpms, n_orders), np.nan, dtype=complex)
    errs = np.full((n_bpms, n_orders), np.nan, dtype=float)
    residual_fraction = np.full(n_bpms, np.nan, dtype=float)

    for i in range(n_bpms):
        b = complex_line[i, :]
        if np.any(np.isnan(b)):
            continue
        x = design_pinv @ b
        residual = b - design @ x
        coeffs[i, :] = x
        norm_b = np.linalg.norm(b)
        if norm_b > 0:
            residual_fraction[i] = np.linalg.norm(residual) / norm_b
        if dof > 0 and np.all(np.isfinite(cov_unit)):
            sigma2 = np.sum(np.abs(residual) ** 2) / dof
            errs[i, :] = np.sqrt(sigma2 * np.diag(cov_unit).real)

    result = {}
    for order_idx, rdt in enumerate(rdts_on_line):
        amp = np.abs(coeffs[:, order_idx])
        phase = (np.angle(coeffs[:, order_idx]) / (2 * np.pi)) % 1
        err_amp = errs[:, order_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            err_phase = np.where(amp > 0, err_amp / (2 * np.pi * np.maximum(amp, 1e-300)), np.nan)

        amp_free, err_amp_free, phase_free = _apply_forced_to_free(amp, err_amp, phase, conversion)

        decomp_df = pd.DataFrame(index=index)
        decomp_df[f"{AMPLITUDE}{DECOMP}"] = amp
        decomp_df[f"{ERR}{AMPLITUDE}{DECOMP}"] = err_amp
        decomp_df[f"{PHASE}{DECOMP}"] = phase
        decomp_df[f"{ERR}{PHASE}{DECOMP}"] = err_phase
        decomp_df[f"{REAL}{DECOMP}"] = amp * np.cos(2 * np.pi * phase)
        decomp_df[f"{IMAG}{DECOMP}"] = amp * np.sin(2 * np.pi * phase)
        decomp_df[f"{AMPLITUDE}{DECOMP}{FREE}"] = amp_free
        decomp_df[f"{ERR}{AMPLITUDE}{DECOMP}{FREE}"] = err_amp_free
        decomp_df[f"{PHASE}{DECOMP}{FREE}"] = phase_free
        decomp_df[f"{ERR}{PHASE}{DECOMP}{FREE}"] = err_phase  # unaffected: see _apply_forced_to_free
        decomp_df[f"{REAL}{DECOMP}{FREE}"] = amp_free * np.cos(2 * np.pi * phase_free)
        decomp_df[f"{IMAG}{DECOMP}{FREE}"] = amp_free * np.sin(2 * np.pi * phase_free)
        decomp_df[LINE_RESIDUAL] = residual_fraction
        decomp_df[N_CONTRIBUTORS] = n_orders
        result[rdt] = decomp_df
    return result


def _pm_lines(plane: str) -> tuple[LineTuple, LineTuple]:
    """
    The (minus, plus) line tuples for the driven tune line of `plane`: "minus" is the co-rotating
    line, at the same frequency as the plane's own driven tune Q_D (i.e. the tune line itself);
    "plus" is the counter-rotating line, at -Q_D.
    """
    return ((1, 0, 0), (-1, 0, 0)) if plane == "X" else ((0, 1, 0), (0, -1, 0))


def _delta_pm_at_reference(
    meas_input: DotDict,
    input_files: InputFiles,
    phases: dict[str, dict[str, PhaseDict]],
    tunes: TuneDict,
    invariants: dict[str, np.ndarray],
    dpp_value: float | None,
) -> dict[str, tuple[str, np.ndarray, np.ndarray]] | None:
    """
    The complex, co-/counter-rotating driven tune-line amplitudes delta_{u-}, delta_{u+}
    (u = x, y; Eq. 24 of the companion paper), needed by the delta_+ cross-term correction
    (:func:`_delta_plus_correction`). Measured once per plane, at one reference BPM (the first
    BPM of that plane's 90-degree-pair set), rather than at every observation BPM: the model's
    phase advance (already trusted throughout the RDT formalism for propagating the source
    Hamiltonians themselves) is equally trusted to propagate these two extra numbers per plane
    to every magnet source and every observation BPM -- exactly as the tracking studies this
    feature is validated against do (a single NAFF extraction, then propagated analytically).

    By harpy/omc3 convention, `_extract_line_raw` applied to a plane's own tune line (`minus_line`)
    returns an amplitude identically equal to 1 (every other secondary line in that plane's
    spectrum is normalized relative to the tune line's own amplitude), so the true driven-mode
    scale has to come from elsewhere: in normalized coordinates (hat_x = x/sqrt(beta_x), Eq. 1),
    the co-rotating driven line's amplitude is exactly sqrt(2J) (the standard action-angle
    relation x = sqrt(2J beta) cos(phi)), already computed per kick file in `invariants` (from
    :func:`omc3.optics_measurements.kick.calculate`). `plus_line`'s reading is a genuine
    (non-degenerate) ratio relative to that same tune-line reference, so multiplying both lines'
    (amplitude, phase) readings by this same sqrt(2J) converts both to the normalized-coordinate
    driven-mode amplitude the delta_+ Hamiltonian formalism (:func:`_delta_plus_correction`)
    expects.

    Returns `None` if either plane's driven tune-line data cannot be extracted (e.g. missing
    columns, as for a plane that was not AC-dipole driven at all), in which case the correction
    is skipped entirely by the caller (falls back to the plain conversion).
    """
    result = {}
    for plane in PLANES:
        try:
            bpm_names = input_files.bpms(plane=plane, dpp_value=dpp_value)
            phase_data = _best_90_degree_phases(meas_input, bpm_names, phases, tunes, plane)
            minus_line, plus_line = _pm_lines(plane)
            df_m, amp_m, _e, phase_m, _pe = _extract_line_raw(
                meas_input, input_files, phase_data, plane, minus_line, dpp_value
            )
            df_p, amp_p, _e, phase_p, _pe = _extract_line_raw(
                meas_input, input_files, phase_data, plane, plus_line, dpp_value
            )
        except ValueError:
            return None
        ref_bpm = df_m.index[0]
        ref_loc_m, ref_loc_p = df_m.index.get_loc(ref_bpm), df_p.index.get_loc(ref_bpm)
        sqrt2j = np.asarray(invariants[plane]).T[0]
        delta_minus = (amp_m * np.exp(2j * np.pi * phase_m))[ref_loc_m, :] * sqrt2j
        delta_plus = (amp_p * np.exp(2j * np.pi * phase_p))[ref_loc_p, :] * sqrt2j
        result[plane] = (ref_bpm, delta_minus, delta_plus)
    return result


def _alias(f: np.ndarray | float) -> np.ndarray | float:
    """Wraps a frequency (in cycles) to (-0.5, 0.5], the convention spectral lines are aliased
    to by the harmonic analysis."""
    return (f + 0.5) % 1.0 - 0.5


def _wrapped_phase(mu_from: float, mu_to: np.ndarray, tune: float) -> np.ndarray:
    """Forward phase advance (cycles) from a single point `mu_from` to each of `mu_to`, in a
    model's MUX/MUY convention (monotonically increasing from 0 to `tune` over one turn),
    wrapping (+`tune`) whenever `mu_to` is "behind" `mu_from`."""
    dphi = mu_to - mu_from
    return np.where(dphi >= 0, dphi, dphi + tune)


def _wrapped_phase_matrix(mu_from: np.ndarray, mu_to: np.ndarray, tune: float) -> np.ndarray:
    """As :func:`_wrapped_phase`, but for every pair of `mu_from` (sources) and `mu_to`
    (observation points): returns a `(len(mu_to), len(mu_from))` matrix."""
    dphi = mu_to[:, None] - mu_from[None, :]
    return np.where(dphi >= 0, dphi, dphi + tune)


def _model_h_arr(elements: pd.DataFrame, j: int, k: int, l: int, m: int, n: int) -> np.ndarray | None:  # noqa: E741
    """
    Per-source (row of the model `elements` table) Hamiltonian coefficient h_{w,jklm}
    (Eq. 18 of the companion paper), for a normal 2n-pole. Only normal, even-(l+m) terms are
    considered (skew sources are not), matching the tracking-validated reference implementation
    (`xsuite_lhc/full_forced_prediction.py`) this is ported from.

    Returns `None` if the model does not carry the K{n-1}L column for this order (as is typical
    for LHC optics-measurement models beyond the chromaticity sextupoles, which do not encode
    the unknown higher-order field errors being measured).
    """
    col = f"K{n - 1}L"
    if col not in elements.columns or (l + m) % 2:
        return None
    knl = elements[col].to_numpy()
    betx = elements["BETX"].to_numpy()
    bety = elements["BETY"].to_numpy()
    with np.errstate(invalid="ignore"):
        h = (
            -(knl * (1j) ** (l + m) * betx ** ((j + k) / 2) * bety ** ((l + m) / 2)
              / (factorial(j) * factorial(k) * factorial(l) * factorial(m) * 2 ** n))
        ).real
    return np.where(knl != 0, h, 0.0)


def _elements_in_beam_direction(meas_input: DotDict) -> pd.DataFrame | None:
    """
    The model element table with MUX/MUY expressed as phase advances *along the beam's direction
    of travel*. LHC Beam 2 models are ordered in the Beam 1 frame (``beam_direction == -1``): their
    MU increases against the beam, so the beam-direction phase is Q_full - MU. Every "forward"
    propagation (source -> BPM, AC dipole -> source) below relies on this.
    """
    elements = getattr(meas_input.accelerator, "elements", None)
    if elements is None or getattr(meas_input.accelerator, "beam_direction", 1) != -1:
        return elements
    headers = getattr(elements, "headers", {})
    if not {"Q1", "Q2"} <= set(headers):
        return elements
    flipped = elements.copy()
    flipped["MUX"] = float(headers["Q1"]) - elements["MUX"]
    flipped["MUY"] = float(headers["Q2"]) - elements["MUY"]
    return flipped


def _ac_dipole_mu(meas_input: DotDict, elements: pd.DataFrame, index: pd.Index) -> tuple[float, float] | None:
    """
    Model phase advances (MUX, MUY) of the AC dipole (exciter) element, or `None` if the
    accelerator cannot name it or the model element table does not contain it.
    """
    accel = meas_input.accelerator
    try:
        _bpm, exciter = accel.get_exciter_bpm("X", list(index))
        return float(elements.loc[exciter, "MUX"]), float(elements.loc[exciter, "MUY"])
    except (AttributeError, KeyError, TypeError):
        return None


def _forced_line_model(
    meas_input: DotDict,
    plane: str,
    line: LineTuple,
    tunes: TuneDict,
    deltas: dict[str, tuple[str, np.ndarray, np.ndarray]],
    n_kicks: int,
    index: pd.Index,
    nmax: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    First-order model prediction of the forced (AC-dipole driven) spectral `line` at every BPM
    in `index` and for every kick file, split into its clean part (the all-delta_- partitions)
    and the delta_+ cross-terms (every other partition landing on the line).

    Bookkeeping (validated against tracking with the AC dipole mid-ring, see the unit tests):

    - Everything is referred to the exit of the AC dipole, where the driven orbit is continuous
      for one full turn. Past the AC dipole the driven components jump by one drive period
      relative to free propagation -- delta_- by exp(2i pi (Q_D - Q)), delta_+ by
      exp(-2i pi (Q_D + Q)) -- so propagating delta_+- freely from a reference BPM around the
      whole ring is only correct up to the AC dipole.
    - Turn-by-turn data count turns from the model start (MU = 0); a point upstream of the AC
      dipole in that turn was last driven one turn earlier, so its complex coefficient at
      frequency omega carries exp(-i omega) relative to the AC-dipole frame.
    - A kick emitted at source w reaches BPM b in the same AC-dipole turn if w precedes b, else
      in the next one (exp(-i omega)).
    - The measured complex line (`complex_line` in :func:`_process_line`) is the h^- line
      coefficient divided by the *amplitude* of the local tune line, with an absolute phase: the
      prediction is divided by |delta_-(b)| only.

    Returns `(clean, cross)`, each a `(len(index), n_kicks)` complex array in those measured
    units, or `None` when the model lacks the data needed (no MUX/MUY/BETX/BETY, no Q1/Q2
    headers, or BPMs missing). Beam direction is handled by :func:`_elements_in_beam_direction`.
    """
    elements = _elements_in_beam_direction(meas_input)
    if elements is None or not {"MUX", "MUY", "BETX", "BETY"} <= set(elements.columns):
        return None
    headers = getattr(elements, "headers", {})
    try:
        q_full_x, q_full_y = float(headers["Q1"]), float(headers["Q2"])
    except KeyError:
        LOGGER.info("Forced-line model skipped: model element table has no Q1/Q2 header.")
        return None

    (ref_x, dxm, dxp), (ref_y, dym, dyp) = deltas["X"], deltas["Y"]
    qx_free, qy_free = tunes["X"]["QF"], tunes["Y"]["QF"]
    qdx, qdy = tunes["X"]["Q"], tunes["Y"]["Q"]
    two_pi = 2 * np.pi

    try:
        mux_b = elements.loc[index, "MUX"].to_numpy()
        muy_b = elements.loc[index, "MUY"].to_numpy()
        mux_refx, muy_refy = elements.loc[ref_x, "MUX"], elements.loc[ref_y, "MUY"]
    except KeyError:
        LOGGER.info("Forced-line model skipped: BPMs not found in the model's element table.")
        return None

    ac = _ac_dipole_mu(meas_input, elements, index)
    if ac is None:
        LOGGER.warning(
            "AC-dipole element not found in the model: assuming it sits just upstream of the "
            "reference BPM. Sources and BPMs across the real AC dipole then get a wrong phase."
        )
        ac = (mux_refx, elements.loc[ref_x, "MUY"])
    mux_ac, muy_ac = ac

    def pos(mux):  # position within the AC-dipole turn, as the x phase from the AC-dipole exit
        return _wrapped_phase(mux_ac, np.asarray(mux, dtype=float), q_full_x)

    def upstream_of_ac(mux):  # upstream of the AC dipole within a data turn (starting at MU = 0)
        return np.asarray(mux, dtype=float) < mux_ac

    # driven components at the AC-dipole exit, in the AC-dipole turn frame
    def at_exit(dm, dp, mu_ref, mu_ac_plane, q_full, qd, ref_is_up):
        k = -1.0 if ref_is_up else 0.0
        back = np.exp(-1j * two_pi * _wrapped_phase(mu_ac_plane, np.array([mu_ref]), q_full)[0])
        return dm * np.exp(-1j * two_pi * qd * k) * back, dp * np.exp(1j * two_pi * qd * k) * back

    dxm0, dxp0 = at_exit(dxm, dxp, mux_refx, mux_ac, q_full_x, qdx, upstream_of_ac(mux_refx))
    dym0, dyp0 = at_exit(
        dym, dyp, muy_refy, muy_ac, q_full_y, qdy, upstream_of_ac(elements.loc[ref_y, "MUX"])
    )

    pos_b = pos(mux_b)
    k_b = np.where(upstream_of_ac(mux_b), -1.0, 0.0)
    target = round(float(_alias(line[0] * qdx + line[1] * qdy)), 9)
    need = (lambda t: t[0] >= 1) if plane == "X" else (lambda t: t[2] >= 1)
    terms = [
        (j, k, l, m)  # noqa: E741
        for j, k, l, m in product(range(nmax + 1), repeat=4)  # noqa: E741
        if 3 <= j + k + l + m <= nmax and (l + m) % 2 == 0 and need((j, k, l, m))
    ]

    clean = np.zeros((len(index), n_kicks), dtype=complex)
    cross = np.zeros((len(index), n_kicks), dtype=complex)
    for (j, k, l, m) in terms:  # noqa: E741
        h_w = _model_h_arr(elements, j, k, l, m, j + k + l + m)
        if h_w is None or not np.any(h_w):
            continue
        mask = h_w != 0
        h_src = h_w[mask]
        mux_w = elements.loc[mask, "MUX"].to_numpy()
        muy_w = elements.loc[mask, "MUY"].to_numpy()
        p_x = _wrapped_phase(mux_ac, mux_w, q_full_x)  # AC-dipole exit -> source w
        p_y = _wrapped_phase(muy_ac, muy_w, q_full_y)
        if plane == "X":
            nux_term, nuy_term = k - j + 1, m - l
            out_phase = _wrapped_phase_matrix(mux_w, mux_b, q_full_x)  # source w -> BPM b
            coeff, q_r, ax_range, ay_range = -2j * j, qx_free, range(j), range(l + 1)
        else:
            nux_term, nuy_term = k - j, m - l + 1
            out_phase = _wrapped_phase_matrix(muy_w, muy_b, q_full_y)
            coeff, q_r, ax_range, ay_range = -2j * l, qy_free, range(j + 1), range(l)
        # the reduced monomial's spatial phase is the same for every partition: both driven
        # components are transported by the same linear map between the AC dipole and the source
        src = h_src * np.exp(1j * two_pi * (nux_term * p_x + nuy_term * p_y))
        later = pos(mux_w)[None, :] >= pos_b[:, None]  # (n_bpms, n_src): kick reaches b next turn
        prop = np.exp(1j * two_pi * out_phase)
        same_turn, next_turn = np.where(later, 0, prop) @ src, np.where(later, prop, 0) @ src

        for ax_ in ax_range:
            for bx in range(k + 1):
                for ay in ay_range:
                    for by in range(m + 1):
                        if plane == "X":
                            nux, nuy = (2 * ax_ - (j - 1)) + (2 * bx - k), (2 * ay - l) + (2 * by - m)
                            binom = comb(j - 1, ax_) * comb(k, bx) * comb(l, ay) * comb(m, by)
                            xp = np.conj(dxm0) ** (j - 1 - ax_) * np.conj(dxp0) ** ax_
                            yp = np.conj(dym0) ** (l - ay) * np.conj(dyp0) ** ay
                        else:
                            nux, nuy = (2 * ax_ - j) + (2 * bx - k), (2 * ay - (l - 1)) + (2 * by - m)
                            binom = comb(j, ax_) * comb(k, bx) * comb(l - 1, ay) * comb(m, by)
                            xp = np.conj(dxm0) ** (j - ax_) * np.conj(dxp0) ** ax_
                            yp = np.conj(dym0) ** (l - 1 - ay) * np.conj(dyp0) ** ay
                        if round(float(_alias(nux * qdx + nuy * qdy)), 9) != target:
                            continue
                        mix = binom * xp * yp * dxm0 ** bx * dxp0 ** (k - bx) * dym0 ** by * dyp0 ** (m - by)
                        om = two_pi * (nux * qdx + nuy * qdy)
                        # (n_bpms,), data-turn frame
                        spatial = (same_turn + np.exp(-1j * om) * next_turn) * np.exp(1j * om * k_b)
                        c = coeff / (1 - np.exp(1j * (two_pi * q_r - om))) * spatial[:, None] * mix[None, :]
                        if (ax_, bx, ay, by) == (0, k, 0, m):
                            clean += c
                        else:
                            cross += c
    readout = np.abs(dxm0 if plane == "X" else dym0)[None, :]
    return clean / readout, cross / readout


def _delta_plus_correction(
    meas_input: DotDict,
    plane: str,
    line: LineTuple,
    tunes: TuneDict,
    deltas: dict[str, tuple[str, np.ndarray, np.ndarray]],
    n_kicks: int,
    index: pd.Index,
    nmax: int,
) -> np.ndarray:
    """
    Model-assisted delta_+ cross-term correction A^(cross) for the spectral `line`, per BPM
    (`index`) and per kick file: the cross-term part of :func:`_forced_line_model`, in the units
    of the measured complex line, to be subtracted from it before extracting the RDT. A no-op
    (zeros) when the model lacks the needed data.
    """
    model = _forced_line_model(meas_input, plane, line, tunes, deltas, n_kicks, index, nmax)
    if model is None:
        return np.zeros((len(index), n_kicks), dtype=complex)
    return model[1]


def _location_factor(
    meas_input: DotDict, plane: str, line: LineTuple, rdt: RDTTuple, tunes: TuneDict, index: pd.Index,
) -> np.ndarray | None:
    """
    Model-assisted AC-dipole location factor L(b) per BPM: the plain forced-to-free conversion
    (:func:`_forced_to_free_factor`) assumes every source lies between the AC dipole and the
    observation point (as for a BPM right after the AC dipole). A source between the BPM and the
    AC dipole instead contributes to the forced RDT with an extra exp(2i pi (nu_x Q_x- + nu_y Q_y-))
    relative to that assumption, so f_free(b) = f'(b) * factor / L(b) with

        L(b) = [sum_{w in (b, AC)} N_w(b) + exp(2i pi (nu_x Q_x- + nu_y Q_y-)) sum_{other w} N_w(b)]
               / sum_w N_w(b),

    N_w(b) the source's forward-propagated free-RDT term. Needs the target RDT's own sources in
    the model (K_{n-1}L column); returns `None` otherwise.
    """
    elements = _elements_in_beam_direction(meas_input)
    if elements is None or not {"MUX", "MUY", "BETX", "BETY"} <= set(elements.columns):
        return None
    j, k, l, m = rdt  # noqa: E741
    h_w = _model_h_arr(elements, j, k, l, m, j + k + l + m)
    if h_w is None or not np.any(h_w):
        return None
    headers = getattr(elements, "headers", {})
    try:
        q_full_x, q_full_y = float(headers["Q1"]), float(headers["Q2"])
        mux_b = elements.loc[index, "MUX"].to_numpy()
        muy_b = elements.loc[index, "MUY"].to_numpy()
    except KeyError:
        return None
    ac = _ac_dipole_mu(meas_input, elements, index)
    if ac is None:
        return None
    mask = h_w != 0
    mux_w, muy_w = elements.loc[mask, "MUX"].to_numpy(), elements.loc[mask, "MUY"].to_numpy()
    n_w = h_w[mask][None, :] * np.exp(2j * np.pi * (
        (j - k) * _wrapped_phase_matrix(mux_w, mux_b, q_full_x)
        + (l - m) * _wrapped_phase_matrix(muy_w, muy_b, q_full_y)
    ))  # (n_bpms, n_src)
    nux, nuy = line[0], line[1]
    jump = np.exp(2j * np.pi * (nux * (tunes["X"]["Q"] - tunes["X"]["QF"])
                                + nuy * (tunes["Y"]["Q"] - tunes["Y"]["QF"])))
    pos_w = _wrapped_phase(ac[0], mux_w, q_full_x)
    pos_b = _wrapped_phase(ac[0], mux_b, q_full_x)
    between = pos_w[None, :] > pos_b[:, None]  # source between the BPM and the AC dipole
    total = n_w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(between, n_w, jump * n_w).sum(axis=1)) / total


def _add_tunes_if_in_second_turn(
    df: pd.DataFrame, input_files: InputFiles, line: LineTuple, phase2: np.ndarray, dpp_value: float | None,
) -> np.ndarray:
    # With pandas 3.x phase2 might be passed as a read-only view so we ensure
    # a copy is made here since we intend to mutate before returning
    phase2 = np.array(phase2, copy=True)
    mask = df_diff(df, S, S2) > 0  # Get all the S2 positions that are less than S
    tunes = np.empty((2, len(input_files.dpp_frames("X", dpp_value))))
    for i, plane in enumerate(PLANES):
        tunes[i] = np.array(
            [lin.headers[f"Q{i + 1}"] for lin in input_files.dpp_frames(plane, dpp_value)]
        )
    phase2[mask, :] = phase2[mask, :] + line[0] * tunes[0] + line[1] * tunes[1]
    return phase2


def _calculate_rdt_phases_from_line_phases(
    df: pd.DataFrame, input_files: InputFiles, line: LineTuple, line_phase: np.ndarray, dpp_value: float | None,
) -> np.ndarray:
    phases = np.zeros((2, df.index.size, len(input_files.dpp_frames("X", dpp_value))))
    for i, plane in enumerate(PLANES):
        if line[i] != 0:
            phases[i] = (
                input_files.joined_frame(plane, [f"{PHASE_ADV}{plane}"], dpp_value=dpp_value)
                .loc[df.index, :]
                .to_numpy()
                % 1
            )
    return line_phase - line[0] * phases[0] - line[1] * phases[1] + 0.25


def _fit_rdt_amplitudes(
    invariants: dict[str, pd.DataFrame], line_amp: np.ndarray, plane: str, rdt: RDTTuple,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns RDT amplitudes in units of metres ^ {1 - n/2}, where n is the order of RDT.
    """
    import warnings

    amps, err_amps = np.empty(line_amp.shape[0]), np.empty(line_amp.shape[0])
    kick_data = get_linearized_problem(invariants, plane, rdt)  # corresponding to actions in metres
    guess = np.mean(line_amp / kick_data, axis=1)

    def fitting(x, f):
        return f * x

    for i, bpm_rdt_data in enumerate(line_amp):
        # Catch the potential 'OptimizeWarning: Covariance of the parameters could not be estimated'
        # that can happen during the curve_fit. We relay it as a logged message, which allows us to
        # avoid polluting the stderr and allows the user to not see it depending on log level
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(fitting, kick_data, bpm_rdt_data, p0=guess[i])
            amps[i] = popt[0]
            sqrt_pcov = np.sqrt(pcov).flat[0]
            err_amps[i] = (
                sqrt_pcov if np.isfinite(sqrt_pcov) else 0.0
            )  # if single file is used, the error is reported as Inf, which is then overwritten with 0

            # We log any captured warning at warning level
            for warning in records:
                LOGGER.warning(f"Curve fit warning: {warning.message}")

    return amps, err_amps


def get_linearized_problem(invs: dict[str, np.ndarray], plane: str, rdt: RDTTuple) -> np.ndarray:
    """
    2 * j * f_jklm * (powers of 2Jx and 2Jy) : f_jklm is later a parameter of a fit
    we use sqrt(2J): unit is sqrt(m).
    """
    j, k, l, m = rdt  # noqa: E741
    act_x = invs["X"].T[0]
    act_y = invs["Y"].T[0]
    if plane == "X":
        return 2 * j * act_x ** (j + k - 2) * act_y ** (l + m)
    return 2 * l * act_x ** (j + k) * act_y ** (l + m - 2)


def get_line_sign_and_suffix(
    line: LineTuple, input_files: InputFiles, plane: str
) -> tuple[int, str]:
    suffix = f"{line[0]}{line[1]}".replace("-", "_")  # link to harpy, for consistency? (jgray 2024)
    conj_suffix = f"{-line[0]}{-line[1]}".replace("-", "_")
    if f"{COL_AMP}{suffix}" in input_files[plane][0].columns:
        return 1, suffix
    if f"{COL_AMP}{conj_suffix}" in input_files[plane][0].columns:
        return -1, conj_suffix

    # The specified AMP column hasn't been found in the lin file
    msg = (
        f"The column AMP{suffix} has not been found in the lin{plane.lower()} file. "
        "Consider re-running the frequency analysis with a higher order resonance term"
    )
    raise ValueError(msg)


def complex_secondary_lines(
    phase_adv: ArrayLike,
    err_padv: ArrayLike,
    sig1: ArrayLike,
    sig2: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts amplitude and phase of secondary spectral lines from BPM pairs.

    Args:
        phase_adv: phase advances between two BPMs.
        err_padv: error on the phase advance between two BPMs.
        sig1: Complex coefficients of a secondary lines at the first BPM of the pairs.
        sig2: Complex coefficients of a secondary lines at the second BPM of the pairs.

    Returns:
         `Tuple` with amplitudes, phases, err_amplitudes and err_phases of the complex signal.
    """
    tp = 2.0 * np.pi
    # computing complex secondary line (h-) = x - ip_x. Why do we divide by 2? (jgray 2024)
    sig = (sig1 * (1 + 1j / np.tan(phase_adv * tp)) - sig2 * 1j / np.sin(phase_adv * tp)) / 2
    # computing error secondary line (h-) # TODO is this really the error?
    esig = (
        (
            sig1 * 1j / np.square(np.sin(phase_adv * tp))
            + sig2 * -1j * np.cos(phase_adv * tp) / np.square(np.sin(phase_adv * tp))
        )
        * err_padv
        / 2
    )
    return (np.abs(sig), (np.angle(sig) / tp) % 1, np.abs(esig), (np.angle(esig) / tp) % 1)


def to_complex(amplitudes: ArrayLike, phases: ArrayLike, period: float = 1) -> np.ndarray:
    with suppress(AttributeError):
        amplitudes = amplitudes.to_numpy()

    with suppress(AttributeError):
        phases = phases.to_numpy()

    return amplitudes * np.exp(2j * np.pi * phases / period)
