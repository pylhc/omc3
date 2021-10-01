"""
Coupling
--------

This module contains linear coupling calculations related functionality of ``optics_measurements``.
It provides functions to computes and the coupling resonance driving terms, which are part of the standard
optics outputs.
"""
from collections import OrderedDict
from functools import reduce
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    header_dict: OrderedDict
) -> None:
    """
    Calculates the coupling RDTs f1001 and f1010, as well as the closest tune approach Cminus (|C-|).
    This represents the "2 BPM method" in https://cds.cern.ch/record/1264111/files/CERN-BE-Note-2010-016.pdf
    (a more up-to-date reference will come in the near future).

    Two formulae are used to calculate the Cminus, taken from the following reference:
    http://cds.cern.ch/record/2135848/files/PhysRevSTAB.17.051004.pdf
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

    # We need vertical and horizontal spectra, so we have to intersect first all inputs with X and Y phase
    # output furthermore the output has to be rearranged in the order of the model (important for e.g. LHC
    # beam 2) and thus we need to intersect the *model* index with all the above. Since the first index of
    # the .intersect chain dictates the order, we have to start with the model index.
    LOGGER.debug("Intersecting measurements, starting with model")
    compensation = "uncompensated" if meas_input.compensation == "model" else "free"
    joined = _joined_frames(input_files)
    joined_index = (
        meas_input.accelerator.model.index.intersection(joined.index)
        .intersection(phase_dict["X"][compensation]["MEAS"].index)
        .intersection(phase_dict["Y"][compensation]["MEAS"].index)
    )

    joined = joined.loc[joined_index].copy()
    phases_x = phase_dict["X"][compensation]["MEAS"].loc[joined_index].copy()
    phases_y = phase_dict["Y"][compensation]["MEAS"].loc[joined_index].copy()

    bd = meas_input.accelerator.beam_direction

    # standard arithmetic mean for amplitude columns, circular mean (`period=1`) for frequency columns
    LOGGER.debug("Averaging amplitude and frequency columns")
    for col in [SECONDARY_AMPLITUDE_X, SECONDARY_AMPLITUDE_Y]:
        cols = [c for c in joined.columns if c.startswith(col)]
        joined[col] = stats.weighted_mean(joined[cols], axis=1)

    for col in [SECONDARY_FREQUENCY_X, SECONDARY_FREQUENCY_Y]:
        cols = [x for x in joined.columns if x.startswith(col)]
        joined[col] = bd * stats.circular_mean(joined[cols], axis=1)

    pairs_x, deltas_x = _find_pair(phases_x, 1)
    pairs_y, deltas_y = _find_pair(phases_y, 1)

    A01 = 0.5 * _get_complex(
        joined[SECONDARY_AMPLITUDE_X].to_numpy() * exp(joined[SECONDARY_FREQUENCY_X].to_numpy() * PI2I),
        deltas_x,
        pairs_x,
    )
    B10 = 0.5 * _get_complex(
        joined[SECONDARY_AMPLITUDE_Y].to_numpy() * exp(joined[SECONDARY_FREQUENCY_Y].to_numpy() * PI2I),
        deltas_y,
        pairs_y,
    )
    A0_1 = 0.5 * _get_complex(
        joined[SECONDARY_AMPLITUDE_X].to_numpy() * exp(-joined[SECONDARY_FREQUENCY_X].to_numpy() * PI2I),
        deltas_x,
        pairs_x,
    )
    B_10 = 0.5 * _get_complex(
        joined[SECONDARY_AMPLITUDE_Y].to_numpy() * exp(-joined[SECONDARY_FREQUENCY_Y].to_numpy() * PI2I),
        deltas_y,
        pairs_y,
    )

    # columns in `joined` that haven't been swapped before, need to be now
    q1001_from_A = -np.angle(A01) + (bd * joined[f"{COL_MU}Y"].to_numpy() - 0.25) * PI2
    q1001_from_B = np.angle(B10) - (bd * joined[f"{COL_MU}X"].to_numpy() - 0.25) * PI2
    eq1001 = exp(1.0j * q1001_from_A) + exp(1.0j * q1001_from_B)

    q1010_from_A = -np.angle(A0_1) - (bd * joined[f"{COL_MU}Y"].to_numpy() - 0.25) * PI2
    q1010_from_B = -np.angle(B_10) - (bd * joined[f"{COL_MU}X"].to_numpy() - 0.25) * PI2
    eq1010 = exp(1.0j * q1010_from_A) + exp(1.0j * q1010_from_B)

    # `eq / abs(eq)` to get the average
    f1001 = -0.5 * sqrt(np.abs(A01 * B10)) * eq1001 / abs(eq1001)
    f1010 = 0.5 * sqrt(np.abs(A0_1 * B_10)) * eq1010 / abs(eq1010)

    tune_sep = np.abs(tune_dict["X"]["QFM"] % 1.0 - tune_dict["Y"]["QFM"] % 1.0)

    # old Cminus, approximate formula without rdt phases
    C_old = 4.0 * tune_sep * np.mean(np.abs(f1001))
    header_dict["Cminus_approx"] = C_old
    LOGGER.info(f"|C-| (approx) = {C_old}, tune_sep = {tune_sep}, from Eq.1 in PRSTAB 17,051004")

    # new Cminus
    C_new = np.abs(
        4.0 * tune_sep * np.mean(f1001 * exp(1.0j * (joined[f"{COL_MU}X"] - joined[f"{COL_MU}Y"])))
    )
    header_dict["Cminus_exact"] = C_new
    LOGGER.info(f"|C-| (exact)  = {C_new}, from Eq.2 w/o i*s*Delta/R in PRSTAB 17,051004")

    if meas_input.compensation == "model":
        f1001, f1010 = compensate_model(f1001, f1010, tune_dict)

    rdt_df = pd.DataFrame(
        index=joined_index,
        columns=["S", "F1001R", "F1010R", "F1001I", "F1010I", "F1001W", "F1010W", "F1001A", "F1010A"],
        data=np.array(
            [
                meas_input.accelerator.model.loc[joined_index, "S"],
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

    rdt_df.sort_values(by="S", inplace=True)

    # adding model values and deltas
    model_coupling = coupling_via_cmatrix(meas_input.accelerator.model).loc[rdt_df.index]
    RDTCOLS = [F1001, F1010]
    # TODO: yikes, uniify with (C)RDTs
    for (domain, func) in [("I", np.imag), ("R", np.real), ("W", np.abs)]:
        for col in RDTCOLS:
            mdlcol = func(model_coupling[col])
            rdt_df[f"{col}{domain}MDL"] = mdlcol
            rdt_df[f"ERR{col}{domain}"] = 0.0
            rdt_df[f"DELTA{col}{domain}"] = rdt_df[f"{col}{domain}"] - mdlcol
            rdt_df[f"ERRDELTA{col}{domain}"] = 0.0

    _write_coupling_tfs(rdt_df, meas_input.outputdir, header_dict)


# TODO: th
def _write_coupling_tfs(rdt_df, outdir, header_dict):
    common_cols = [S]
    # TODO: unify columns with (C)RDTs
    cols_to_print_f1001 = common_cols + [col for col in rdt_df.columns if "1001" in col]
    cols_to_print_f1010 = common_cols + [col for col in rdt_df.columns if "1010" in col]

    tfs.write(Path(outdir) / f"{F1001}.tfs", rdt_df[cols_to_print_f1001], header_dict, save_index="NAME")
    tfs.write(Path(outdir) / f"{F1001}.tfs", rdt_df[cols_to_print_f1010], header_dict, save_index="NAME")


# TODO: th
def compensate_model(f1001, f1010, tune_dict):
    """
    Compensation by model only.

    Args:
        f1001: the pre-calculated driven coupling RDTs
        f1010: the pre-calculated driven coupling RDTs
        tune_dict (TuneDict): the free and driven tunes
    """
    Qx = PI2 * tune_dict["X"]["QFM"]  # natural tunes
    Qy = PI2 * tune_dict["Y"]["QFM"]

    dQx = PI2 * tune_dict["X"]["QM"]  # driven tunes
    dQy = PI2 * tune_dict["Y"]["QM"]

    factor1001 = np.sqrt(np.abs(sin(dQy - Qx) * sin(dQx - Qy))) / np.abs(sin(Qx - Qy))
    factor1010 = np.abs(np.sqrt(sin(Qx + dQy) * sin(Qy + dQx)) / sin(Qx + Qy))
    # TODO: might act in-place, check and fix
    f1001 *= factor1001
    f1010 *= factor1010

    LOGGER.info("Compensation by model")
    LOGGER.info(f"f1001 factor: {factor1001}")
    LOGGER.info(f"f1010 factor: {factor1010}")

    return f1001, f1010


def compensate_ryoichi():
    pass


# ----- Helpers ----- #

# TODO: th
def _take_next(phases, shift=1):
    """
    Takes the following BPM for momentum reconstruction by a given shift
    """
    indices = np.roll(np.arange(phases.to_numpy().shape[0]), shift)
    return indices, phases.to_numpy()[np.arange(phases.to_numpy().shape[0]), indices] - 0.25


# TODO: th
def _find_pair(phases, bd) -> tuple:
    """finds the best candidate for momentum reconstruction

    Args:
      phases (matrix): phase advance matrix
      bd (int): beam direction, will be negative for beam 2.
    """
    slice_ = _tilt_slice_matrix(phases.to_numpy(), 0, 2 * CUTOFF) - 0.25  # do not overwrite builting 'slice'
    indices = np.argmin(abs(slice_), axis=0)
    deltas = slice_[indices, range(len(indices))]
    indices = (indices + np.arange(len(indices))) % len(indices)

    return np.array(indices), deltas


# TODO: th, rename
def _get_complex(spectral_lines, deltas, pairs):
    """
    calculates the complex line from the real lines at positions i and j, where j is determined by
    taking the next BPM with a phase advance sufficiently close to pi/2

    Args:
      spectral_lines (vector): measured (real) spectral lines
      deltas (vector): phase advances minus 90deg
      pairs (vector): indices for pairing
    """
    return (1.0 - 1.0j * tan(PI2 * deltas)) * spectral_lines - 1.0j / cos(PI2 * deltas) * spectral_lines[
        pairs
    ]


def _joined_frames(input_files: dict):
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

        linx.rename(columns=rename_col("X", i), inplace=True)
        liny.rename(columns=rename_col("Y", i), inplace=True)

        merged_df = pd.merge(
            left=linx,
            right=liny,
            on=["NAME", "S"],
            how="inner",
            sort=False,
        )
        joined_dfs.append(merged_df)

    # TODO: make this call pythonic
    reduced = reduce(
        lambda a, b: pd.merge(a, b, how="inner", on=["NAME", "S"], sort=False), joined_dfs
    ).set_index("NAME")
    reduced.rename(columns={"MUX_X_0": "MUX", "MUY_Y_0": "MUY"}, inplace=True)  # TODO: constants
    return reduced


def rename_col(plane: str, index: int) -> str:
    """
    Generate appropriate column name for renaming before merging dataframes from InputFiles.

    Args:
        plane (str): plane for which to apply the renaming.
        index (int): index location of the df which columns are renamed in the input files.

    Returns:
        Generated name as string.
    """
    def fn(column):
        if column in ["NAME", "S"]:
            return column
        return f"{column}_{plane}_{index}"
    return fn
