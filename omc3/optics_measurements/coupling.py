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
from numpy import cos, exp, ndarray, sin, sqrt, tan
from optics_functions.coupling import coupling_via_cmatrix

from omc3.definitions.constants import PI2, PI2I
from omc3.harpy.constants import COL_MU
from omc3.optics_measurements.beta_from_phase import _tilt_slice_matrix
from omc3.optics_measurements.constants import (
    AMPLITUDE,
    F1001,
    F1010,
    IMAG,
    REAL,
    NAME,
    PHASE,
    PHASE_ADV,
    SECONDARY_AMPLITUDE_X,
    SECONDARY_AMPLITUDE_Y,
    SECONDARY_FREQUENCY_X,
    SECONDARY_FREQUENCY_Y,
    S,
    ERR,
    EXT,
    MDL,
    DELTA
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
    joined_index: pd.Index = (
        meas_input.accelerator.model.index.intersection(joined.index)
        .intersection(phase_dict["X"][compensation]["MEAS"].index)
        .intersection(phase_dict["Y"][compensation]["MEAS"].index)
    )
    joined = joined.loc[joined_index].copy()

    phases_x: tfs.TfsDataFrame = phase_dict["X"][compensation]["MEAS"].loc[joined_index, joined_index].copy()
    phases_y: tfs.TfsDataFrame = phase_dict["Y"][compensation]["MEAS"].loc[joined_index, joined_index].copy()

    LOGGER.debug("Averaging (arithmetic mean) amplitude columns")
    for col in [SECONDARY_AMPLITUDE_X, SECONDARY_AMPLITUDE_Y]:
        arithmetically_averaved_columns = [c for c in joined.columns if c.startswith(col)]
        joined[col] = stats.weighted_mean(joined[arithmetically_averaved_columns], axis=1)

    LOGGER.debug("Averaging (circular mean) frequency columns")  # make sure to use period=1 here
    for col in [SECONDARY_FREQUENCY_X, SECONDARY_FREQUENCY_Y]:
        circularly_averaved_columns = [x for x in joined.columns if x.startswith(col)]
        joined[col] = bd * stats.circular_mean(
            joined[circularly_averaved_columns], axis=1, period=1
        )

    LOGGER.debug("Finding BPM pairs for momentum reconstruction")
    bpm_pairs_x, deltas_x = _find_pair(phases_x, meas_input.coupling_pairing)
    bpm_pairs_y, deltas_y = _find_pair(phases_y, meas_input.coupling_pairing)

    LOGGER.debug("Computing complex lines from spectra")
    A01: np.ndarray = 0.5 * _get_complex_line(
        joined[SECONDARY_AMPLITUDE_X] * exp(joined[SECONDARY_FREQUENCY_X] * PI2I), deltas_x, bpm_pairs_x
    )
    B10: np.ndarray = 0.5 * _get_complex_line(
        joined[SECONDARY_AMPLITUDE_Y] * exp(joined[SECONDARY_FREQUENCY_Y] * PI2I), deltas_y, bpm_pairs_y
    )
    A0_1: np.ndarray = 0.5 * _get_complex_line(
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
    f1001: np.ndarray = -0.5 * sqrt(np.abs(A01 * B10)) * eq_1001 / abs(eq_1001)
    f1010: np.ndarray = 0.5 * sqrt(np.abs(A0_1 * B_10)) * eq_1010 / abs(eq_1010)

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

    LOGGER.debug("Adding model values and deltas")
    model_coupling = coupling_via_cmatrix(meas_input.accelerator.model).loc[joined_index]

    f1001_df = _rdt_to_output_df(f1001, model_coupling[F1001], meas_input.accelerator.model, joined_index)
    f1010_df = _rdt_to_output_df(f1010, model_coupling[F1010], meas_input.accelerator.model, joined_index)

    tfs.write(Path(meas_input.outputdir) / f"{F1001.lower()}{EXT}", f1001_df, header_dict)
    tfs.write(Path(meas_input.outputdir) / f"{F1010.lower()}{EXT}", f1010_df, header_dict)


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

def _find_pair(phases: tfs.TfsDataFrame, mode: int = 1):
    """
    Does the BPM pairing for coupling calculation.

    Args:
        mode (int): Value to determine the BPM pairing. If ``0`` is given,
            tries to find the best candidate. If a value ``n>=1`` is given,
            then takes the n-th following BPM downstream for the pairing.
    """

    if mode == 0:
        return _find_candidate(phases)
    else:
        return _take_next(phases, mode)


def _take_next(phases: tfs.TfsDataFrame, shift: int = 1):
    """
    Takes the following BPM for momentum reconstruction by a given shift.

    Args:
        phases (tfs.TfsDataFrame): Dataframe matrix of phase advances, as calculated in phase.py.
        shift (int): Value to determine the BPM pairing. If ``0`` is given,
           tries to find the best candidate. If a value ``n>=1`` is given,
           then takes the n-th following BPM downstream for the pairing.
   """
    indices = np.roll(np.arange(phases.to_numpy().shape[0]), shift)
    return indices, phases.to_numpy()[np.arange(phases.to_numpy().shape[0]), indices] - 0.25


def _find_candidate(phases: tfs.TfsDataFrame) -> Tuple[np.ndarray, np.ndarray]:
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


def _rdt_to_output_df(
    fterm: Union[pd.Series, np.ndarray],
    fterm_mdl: Union[pd.Series, np.ndarray],
    model_df: tfs.TfsDataFrame,
    index: pd.Index,
) -> pd.DataFrame:
    """
    Creates the output coupling RDT dataframe from the given RDT and its model-calculated counterpart.
    Combines all the needed columns (``S``, ``NAME``, ``AMP``, ``PHASE``, ``REAL``, ``IMAG``, and the
    ``ERR*`` and ``*MDL`` columns).

    .. note::
        At the moment, the dataframe holds a ``DELTAREAL`` and ``DELTAIMAG`` columns, which are calculated
        as `REAL - REALMDL` and `IMAG - IMAGMDL`. As most of the time the model is coupling-free, these are
        usually going to be identical values to ``REAL`` and ``IMAG`` columns. They are still included in
        case one wishes to have a coupled model, as some machines sometimes do (but not LHC afaik).

        Similarly, there are ``ERRDELTAREAL`` and ``ERRDELTAIMAG`` columns, which are at the moment
        the same values as ``ERRREAL`` and ``ERRIMAG`` columns. In the future, we might want to have
        a fancier calculation for these.

    .. important::
        The columns mentionned in the note above are required and expected in the correction calculation.
        It would fail the correction functionality to remove these columns.

    Args:
        fterm (Union[pd.Series, np.ndarray]): the calculated coupling RDT.
        fterm_mdl (Union[pd.Series, np.ndarray]): corresponding RDT values calculated from the model (e.g.
            calculated via cmatrix from the model_df).
        model_df (tfs.TfsDataFrame): the model dataframe attached to the accelerator object, used to get the
            ``S`` position.
        index (pd.Index): the joined intersected index used to align everything.

    Returns:
        pd.DataFrame: dataframe ready to be written to the out file `.tfs`
    """
    df = pd.DataFrame()
    df[S] = model_df.loc[index, S]
    df[NAME] = index

    LOGGER.debug("Computing RDT amplitude values")
    df[AMPLITUDE] = np.abs(fterm)
    df[AMPLITUDE + MDL] = np.abs(fterm_mdl)

    LOGGER.debug("Computing phase values")
    df[PHASE] = np.angle(fterm)
    df[PHASE + MDL] = np.angle(fterm_mdl)

    LOGGER.debug("Computing deviation from model")
    df[DELTA + AMPLITUDE] = df[AMPLITUDE] - df[AMPLITUDE + MDL]
    df[DELTA + PHASE] = df[PHASE] - df[PHASE + MDL]

    LOGGER.debug("Computing error values")
    df[ERR + AMPLITUDE] = 0  # TODO: will need to implement this calculation later
    df[ERR + PHASE] = 0

    LOGGER.debug("Adding real and imaginary parts columns")
    df[REAL] = np.real(fterm)
    df[REAL + MDL] = np.real(fterm_mdl)
    df[ERR + REAL] = 0  # TODO: same
    # These following columns are needed in the correction calculation later on
    # Most of the time model has 0 coupling so the DELTA is just the REAL / IMAG but let's
    # not neglect that we might want to have weird coupled models sometimes
    # For now error on delta is just the error on REAL / IMAG but in the future
    # we might want to change this for a fancier calculation
    df[DELTA + REAL] = df[REAL] - df[REAL + MDL]
    df[ERR + DELTA + REAL] = df[ERR + REAL]

    df[IMAG] = np.imag(fterm)
    df[IMAG + MDL] = np.imag(fterm_mdl)
    df[ERR + IMAG] = 0  # TODO: same
    # See comment above, same thing here for IMAG
    df[DELTA + IMAG] = df[IMAG] - df[IMAG + MDL]
    df[ERR + DELTA + IMAG] = df[ERR + IMAG]

    return df.sort_values(by=S)


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
            on=[NAME],
            how="inner",
            sort=False,
        )
        joined_dfs.append(merged_transverse_lins_df)

    partial_merge = partial(pd.merge, how="inner", on=[NAME], sort=False)
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
        if column == NAME:
            return column
        return f"{column}_{plane}_{index}"

    return fn
