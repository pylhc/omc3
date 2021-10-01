"""
Coupling
--------

This module contains linear coupling calculations related functionality of ``optics_measurements``.
It provides functions to computes and the coupling resonance driving terms, which are part of the standard
optics outputs.
"""
from collections import OrderedDict
from functools import partial, reduce
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tfs
from numpy import cos, exp, sin, sqrt, tan
from optics_functions.coupling import coupling_via_cmatrix

from omc3.definitions.constants import PI2, PI2I
from omc3.harpy.constants import COL_MU
from omc3.optics_measurements.beta_from_phase import _tilt_slice_matrix
from omc3.optics_measurements.constants import (
    AMPLITUDE,
    F1001,
    F1010,
    NAME,
    PHASE,
    PHASE_ADV,
    SECONDARY_AMPLITUDE_X,
    SECONDARY_AMPLITUDE_Y,
    SECONDARY_FREQUENCY_X,
    SECONDARY_FREQUENCY_Y,
    S,
)
from omc3.utils import logging_tools, stats

LOGGER = logging_tools.get_logger(__name__)

COLS_TO_KEEP_X: List[str] = [NAME, S, f"{AMPLITUDE}01", f"{PHASE}01", f"{PHASE_ADV}X"]
COLS_TO_KEEP_Y: List[str] = [NAME, S, f"{AMPLITUDE}10", f"{PHASE}10", f"{PHASE_ADV}Y"]
CUTOFF: int = 5


def calculate_coupling(
    meas_input: dict,
    input_files: dict,
    phase_dict: Dict[str, Tuple[Dict[str, tfs.TfsDataFrame], Sequence[tfs.TfsDataFrame]]],
    tune_dict: Dict[str, float],
    header_dict: OrderedDict,
) -> None:
    """
    Calculates the coupling RDTs f1001 and f1010, as well as the closest tune approach Cminus (|C-|).
    This represents the "2 BPM method" in https://cds.cern.ch/record/1264111/files/CERN-BE-Note-2010-016.pdf
    (a more up-to-date reference will come in the near future).

    Two formulae are used to calculate the Cminus, taken from the following reference:
    https://cds.cern.ch/record/2135848/files/PhysRevSTAB.17.051004.pdf
    The first one (Eq(1)) is an approximation using only the amplitudes of the RDTs, while the second one
    (Eq(2) in the same paper) is more exact but needs also the phase of the RDT.

    The results are written down in the optics_measurements outputs as **f1001.tfs** and **f1010.tfs** files.

    Args:
        meas_input (dict): `OpticsInput` object containing analysis settings from the command-line.
        input_files (dict): `InputFiles` (dict) object containing frequency spectra files (linx/y) for
            each transverse plane (as keys).
        phase_dict (Dict[str, Tuple[Dict[str, tfs.TfsDataFrame], tfs.TfsDataFrame]]): dictionary containing
            the measured phase advances, with an entry for each transverse plane. In said entry is a
            dictionary with the measured phase advances for 'free' and 'uncompensated' cases, as well as
            the location of the output ``TfsDataFrames`` for the phases.
        tune_dict (Dict[str, float]): `TuneDict` object containing measured tunes. There is an entry
            calculated for the 'Q', 'QF', 'QM', 'QFM' and 'ac2bpm' modes, each value being a float.
        header_dict (OrderedDict): header dictionary of common items for coupling output files,
            will be attached as the header to the **f1001.tfs** and **f1010.tfs** files..
    """
    LOGGER.info("Calculating coupling")
    bd = meas_input.accelerator.beam_direction
    compensation = "uncompensated" if meas_input.compensation == "model" else "free"

    # We need vertical and horizontal spectra, so we have to intersect first all inputs with X and Y phase
    # output furthermore the output has to be rearranged in the order of the model (important for e.g. LHC
    # beam 2) and thus we need to intersect the *model* index with all the above. Since the first index of
    # the .intersect chain dictates the order, we have to start with the model index.
    LOGGER.debug("Intersecting measurements, starting with model")
    joined: tfs.TfsDataFrame = _joined_frames(input_files)  # merge transverse input frames
    joined_index = (
        meas_input.accelerator.model.index.intersection(joined.index)
        .intersection(phase_dict["X"][compensation]["MEAS"].index)
        .intersection(phase_dict["Y"][compensation]["MEAS"].index)
    )
    joined = joined.loc[joined_index].copy()

    phases_x: tfs.TfsDataFrame = phase_dict["X"][compensation]["MEAS"].loc[joined_index].copy()
    phases_y: tfs.TfsDataFrame = phase_dict["Y"][compensation]["MEAS"].loc[joined_index].copy()

    # standard arithmetic mean for amplitude columns, circular mean (`period=1`) for frequency columns
    LOGGER.debug("Averaging (arithmetic mean) amplitude columns")
    for col in [SECONDARY_AMPLITUDE_X, SECONDARY_AMPLITUDE_Y]:
        arithmetically_averaved_columns = [c for c in joined.columns if c.startswith(col)]
        joined[col] = stats.weighted_mean(joined[arithmetically_averaved_columns], axis=1)

    LOGGER.debug("Averaging (circular mean) frequency columns")
    for col in [SECONDARY_FREQUENCY_X, SECONDARY_FREQUENCY_Y]:
        circularly_averaved_columns = [x for x in joined.columns if x.startswith(col)]
        joined[col] = bd * stats.circular_mean(
            joined[circularly_averaved_columns], axis=1
        )  # TODO: check  with andreas for the period=1 in comment but not in code

    LOGGER.debug("Finding BPM pairs for momentum reconstruction")
    bpm_pairs_x, deltas_x = _find_pair(phases_x, 1)
    bpm_pairs_y, deltas_y = _find_pair(phases_y, 1)

    LOGGER.debug("Computing complex lines from spectra")
    A01: np.ndarray = 0.5 * _get_complex_line(
        joined[SECONDARY_AMPLITUDE_X] * exp(joined[SECONDARY_FREQUENCY_X] * PI2I), deltas_x, bpm_pairs_x
    )
    B10: np.ndarray = 0.5 * _get_complex_line(
        joined[SECONDARY_AMPLITUDE_Y] * exp(joined[SECONDARY_FREQUENCY_Y] * PI2I), deltas_y, bpm_pairs_y
    )
    A0_1: np.ndarray = 0.5 * _get_complex_line(  # TODO: check this underscore position with andreas
        joined[SECONDARY_AMPLITUDE_X] * exp(-joined[SECONDARY_FREQUENCY_X] * PI2I), deltas_x, bpm_pairs_x
    )
    B_10: np.ndarray = 0.5 * _get_complex_line(
        joined[SECONDARY_AMPLITUDE_Y] * exp(-joined[SECONDARY_FREQUENCY_Y] * PI2I), deltas_y, bpm_pairs_y
    )

    q1001_from_A = -np.angle(A01) + (bd * joined[f"{COL_MU}Y"].to_numpy() - 0.25) * PI2
    q1001_from_B = np.angle(B10) - (bd * joined[f"{COL_MU}X"].to_numpy() - 0.25) * PI2
    eq_1001 = exp(1.0j * q1001_from_A) + exp(1.0j * q1001_from_B)

    q1010_from_A = -np.angle(A0_1) - (bd * joined[f"{COL_MU}Y"].to_numpy() - 0.25) * PI2
    q1010_from_B = -np.angle(B_10) - (bd * joined[f"{COL_MU}X"].to_numpy() - 0.25) * PI2
    eq_1010 = exp(1.0j * q1010_from_A) + exp(1.0j * q1010_from_B)

    LOGGER.debug("Computing average of coupling RDTs")
    f1001 = -0.5 * sqrt(np.abs(A01 * B10)) * eq_1001 / abs(eq_1001)
    f1010 = 0.5 * sqrt(np.abs(A0_1 * B_10)) * eq_1010 / abs(eq_1010)

    LOGGER.debug("Getting tune separation from measurements")
    tune_separation = np.abs(tune_dict["X"]["QFM"] % 1.0 - tune_dict["Y"]["QFM"] % 1.0)

    LOGGER.debug("Calculating approximated Cminus")
    C_approx = 4.0 * tune_separation * np.mean(np.abs(f1001))
    header_dict["Cminus_approx"] = C_approx
    LOGGER.info(
        f"|C-| (approx) = {C_approx:.5f}, tune_sep = {tune_separation:.3f}, from Eq.1 in PRSTAB 17,051004"
    )

    LOGGER.debug("Calculating exact Cminus")
    C_exact = np.abs(4.0 * tune_separation * np.mean(f1001 * exp(1.0j * (joined[f"{COL_MU}X"] - joined[f"{COL_MU}Y"]))))
    header_dict["Cminus_exact"] = C_exact
    LOGGER.info(f"|C-| (exact)  = {C_exact:.5f}, from Eq.2 w/o i*s*Delta/R in PRSTAB 17,051004")

    if meas_input.compensation == "model":
        LOGGER.debug("Compensating coupling RDT values by model")
        f1001, f1010 = compensate_rdts_by_model(f1001, f1010, tune_dict)

    LOGGER.debug("Combining RDTs into a single dataframe")
    rdt_df = pd.DataFrame(  # TODO: take care of column names (next PR)
        index=joined_index,
        columns=[S, f"{F1001}R", f"{F1010}R", f"{F1001}I", f"{F1010}I", f"{F1001}W", f"{F1010}W", f"{F1001}A", f"{F1010}A"],
        data=np.array(
            [
                meas_input.accelerator.model.loc[joined_index, S],
                np.real(f1001),
                np.real(f1010),
                np.imag(f1001),
                np.imag(f1010),
                np.abs(f1001),
                np.abs(f1010),
                np.angle(f1001) / PI2,
                np.angle(f1010) / PI2,
            ]
        ).transpose(),
    )
    rdt_df = rdt_df.sort_values(by=S)

    LOGGER.debug("Adding model values and deltas")
    model_coupling = coupling_via_cmatrix(meas_input.accelerator.model).loc[rdt_df.index]
    RDTCOLS = [F1001, F1010]
    # TODO: take care of column names (next PR)
    for (domain, func) in [("I", np.imag), ("R", np.real), ("W", np.abs)]:
        for col in RDTCOLS:
            mdlcol = func(model_coupling[col])
            rdt_df[f"{col}{domain}MDL"] = mdlcol
            rdt_df[f"ERR{col}{domain}"] = 0.0
            rdt_df[f"DELTA{col}{domain}"] = rdt_df[f"{col}{domain}"] - mdlcol
            rdt_df[f"ERRDELTA{col}{domain}"] = 0.0

    _write_coupling_files(rdt_df, meas_input.outputdir, header_dict)


def _write_coupling_files(rdt_df: tfs.TfsDataFrame, outdir: Union[str, Path], header_dict: dict) -> None:
    """
    Write out to file both coupling RDTs data (sum and difference resonance terms)

    Args:
        rdt_df (tfs.TfsDataFrame): complete dataframe with both sum and difference resonance RDT values,
            and comparison to model. Relevant columns are selected for output.
        outdir (Union[str, Path]): location of the output directory as queried from the commandline.
        header_dict (dict): headers dictionary to attach to the output dataframes.
    """
    common_cols = [S]
    cols_to_print_f1001 = common_cols + [col for col in rdt_df.columns if "1001" in col]
    cols_to_print_f1010 = common_cols + [col for col in rdt_df.columns if "1010" in col]

    tfs.write(
        Path(outdir) / f"{F1001.lower()}.tfs", rdt_df[cols_to_print_f1001], header_dict, save_index=NAME
    )
    tfs.write(
        Path(outdir) / f"{F1010.lower()}.tfs", rdt_df[cols_to_print_f1010], header_dict, save_index=NAME
    )


def compensate_rdts_by_model(
    f1001: np.ndarray, f1010: np.ndarray, tune_dict: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compensate coupling RDTs by model (equation) only, implies we're providing a driven model (ACD / ATD
    kick). The scaling factors are calculated from the model's free and driven tunes, and the scaled RDTs
    are returned.

    Args:
        f1001 (np.ndarray): the pre-calculated driven coupling RDTs as an array.
        f1010 (np.ndarray): the pre-calculated driven coupling RDTs as an array.
        tune_dict (Dict[str, float]): `TuneDict` object containing measured tunes. There is an entry
            calculated for the 'Q', 'QF', 'QM', 'QFM' and 'ac2bpm' modes, each value being a float.

    Returns:
        The scaled RDTs.
    """
    f1001 = np.array(f1001)  # make sure we don't modify inplace
    f1010 = np.array(f1010)

    LOGGER.debug("Retrieving model's natural tunes")  # QFM is free since model is ACD/ADT- driven model
    Qx = PI2 * tune_dict["X"]["QFM"]
    Qy = PI2 * tune_dict["Y"]["QFM"]

    LOGGER.debug("Retrieving model's driven tunes")  # QM is driven as model is ACD/ADT-driven model
    Qx_driven = PI2 * tune_dict["X"]["QM"]
    Qy_driven = PI2 * tune_dict["Y"]["QM"]

    LOGGER.debug("Computing scaling factor from driven model")
    f1001_scaling_factor = np.sqrt(np.abs(sin(Qy_driven - Qx) * sin(Qx_driven - Qy))) / np.abs(sin(Qx - Qy))
    f1010_scaling_factor = np.abs(np.sqrt(sin(Qx + Qy_driven) * sin(Qy + Qx_driven)) / sin(Qx + Qy))
    f1001 *= f1001_scaling_factor
    f1010 *= f1010_scaling_factor

    LOGGER.info("Compensation by model:")
    LOGGER.info(f"\t f1001 scaling factor: {f1001_scaling_factor:.5f}")
    LOGGER.info(f"\t f1010 scaling factor: {f1010_scaling_factor:.5f}")
    return f1001, f1010


def compensate_rdts_ryoichi():
    pass


# ----- Helpers ----- #

# def _take_next(phases: tfs.TfsDataFrame, shift: int = 1):
#     """
#     Takes the following BPM for momentum reconstruction by a given shift.
#
#     Args:
#         phases (tfs.TfsDataFrame): Dataframe matrix of phase advances, as calculated in phase.py.
#         shift (int): ???
#     """
#     indices = np.roll(np.arange(phases.to_numpy().shape[0]), shift)
#     return indices, phases.to_numpy()[np.arange(phases.to_numpy().shape[0]), indices] - 0.25


# TODO: bd is not used? check with andreas
def _find_pair(phases: tfs.TfsDataFrame, bd: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the best candidate for momentum reconstruction.

    Args:
      phases (tfs.TfsDataFrame): Dataframe matrix of phase advances, as calculated in phase.py.
      bd (int): beam direction, will be negative for beam 2.

    Returns:
        The indices of best candidates, and the corresponding phase advances between these indices.
    """
    slice_ = _tilt_slice_matrix(phases.to_numpy(), 0, 2 * CUTOFF) - 0.25  # do not overwrite built-in 'slice'
    indices = np.argmin(abs(slice_), axis=0)
    deltas = slice_[indices, range(len(indices))]
    indices = (indices + np.arange(len(indices))) % len(indices)
    return np.array(indices), deltas


def _get_complex_line(
    spectral_lines: Union[pd.Series, np.ndarray],
    deltas: Union[pd.Series, np.ndarray],
    pairs: Union[pd.Series, np.ndarray],
) -> np.ndarray:
    """
    Calculates the complex line from the real lines at positions i and j, where j is determined by
    taking the next BPM with a phase advance sufficiently close to pi/2

    Args:
      spectral_lines (Union[pd.Series, np.ndarray]): vector with measured (real) spectral lines.
      deltas (Union[pd.Series, np.ndarray]): vector with phase advances minus 90deg.
      pairs (Union[pd.Series, np.ndarray]): vector with indices for pairing.

    Returns:
        A numpy array with the results at all given positions from the inputs.
    """
    spectral_lines = np.array(spectral_lines)  # make sure we avoid any inplace modification of data
    deltas = np.array(deltas)
    pairs = np.array(pairs)
    return (1.0 - 1.0j * tan(PI2 * deltas)) * spectral_lines - 1.0j / cos(PI2 * deltas) * spectral_lines[
        pairs
    ]


def _joined_frames(input_files: dict) -> tfs.TfsDataFrame:
    """
    Merges spectrum data from the two planes from all the input files.

    Args:
        input_files (dict): `InputFiles` (dict) object containing frequency spectra files (linx/y) for each
            transverse plane (as keys).
    """
    joined_dfs = []
    assert len(input_files["X"]) == len(input_files["Y"])

    for i, (linx, liny) in enumerate(zip(input_files["X"], input_files["Y"])):
        linx = linx[COLS_TO_KEEP_X].copy()
        liny = liny[COLS_TO_KEEP_Y].copy()

        linx = linx.rename(columns=rename_col("X", i))
        liny = liny.rename(columns=rename_col("Y", i))

        merged_transverse_lins_df = pd.merge(
            left=linx,
            right=liny,
            on=[NAME, S],
            how="inner",
            sort=False,
        )
        joined_dfs.append(merged_transverse_lins_df)

    partial_merge = partial(pd.merge, how="inner", on=[NAME, S], sort=False)
    reduced = reduce(partial_merge, joined_dfs).set_index(NAME)
    reduced = reduced.rename(
        columns={f"{PHASE_ADV}X_X_0": f"{PHASE_ADV}X", f"{PHASE_ADV}Y_Y_0": f"{PHASE_ADV}Y"}
    )
    return tfs.TfsDataFrame(reduced)


def rename_col(plane: str, index: int) -> Callable:
    """
    Generate appropriate column name for renaming before merging dataframes from InputFiles.

    Args:
        plane (str): plane for which to apply the renaming.
        index (int): index location of the df which columns are renamed in the input files.

    Returns:
        The renaming function callable to be given to pandas.
    """

    def fn(column):
        if column in [NAME, S]:
            return column
        return f"{column}_{plane}_{index}"

    return fn
