"""
Beta from Phase
--------------------

 Implements beta and alpha function calculation from phase advance measurements

 Two implementations are provided:

 - The **Analytical N-BPM method**, described in:

    Langner, A. et al., "Analytical N beam position monitor method",
    Phys. Rev. Accel. Beams 20, 111002 (2017).
    https://cds.cern.ch/record/2307554

    It uses all valid BPM triplet combinations within a configurable range, propagates both phase measurement uncertainties
    and systematic lattice errors (quadrupole field gradient errors, quadrupole longitudinal misalignments, BPM longitudinal
    misalignments and sextupole horizontal misalignments) through an analytical covariance matrix.

- The **3-BPM method**, originally from:

    Castro, P. "Luminosity and beta-function measurements at the electron-positron collider ring LEP"
    Ph.D Thesis, University of Valencia, 1996.
    https://repository.cern/records/eny2v-4y338

    The initial, simple version using only adjacent BPM triplets, with phase measurement uncertainties only.
"""
from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import tfs
from scipy.linalg import circulant

from omc3 import __version__ as VERSION  # noqa: N812
from omc3.definitions.constants import PI2
from omc3.optics_measurements.constants import (
    ALPHA,
    BETA,
    BETA_NAME,
    DELTA,
    ERR,
    EXT,
    MDL,
    MEASUREMENT,
    MODEL,
    NAME,
    PHASE_ADV,
    S,
)
from omc3.optics_measurements.phase import CompensationMode
from omc3.optics_measurements.toolbox import df_diff, df_ratio, df_rel_diff
from omc3.utils import logging_tools, stats
from omc3.utils.misc import StrEnum

if TYPE_CHECKING:
    from generic_parser import DotDict
    from logging import Logger

    from omc3.model.accelerators.accelerator import Accelerator
    from omc3.optics_measurements.phase import PhaseDict
    from omc3.optics_measurements.tune import TuneDict


LOGGER: Logger = logging_tools.get_logger(__name__)

EPSILON: float = 1.0E-16
ZERO_THRESHOLD: float = 1e-2
COT_THRESHOLD: float = 15.9
RCOND: float = 1.0e-14


class Methods(StrEnum):
    THREE_BPM: str = "3BPM method"
    A_NBPM: str = "Analytical N-BPM method"
    NO_ERR: str = "No Errors"


def calculate(
    meas_input: DotDict,
    tunes: TuneDict,
    phase_dict: PhaseDict,
    header_dict: dict[str, Any],
    plane: str,
) -> tuple[tfs.TfsDataFrame, dict[str, Any]]:
    """
    Calculates betas and alphas from phase advances.

    Args:
        meas_input: `OpticsInput` object with optics CLI options.
        tunes: `TuneDict` contains model and measured tunes (incl. free motion).
        phase_dict: `PhaseDict` containing phase advances between BPMs from model,
            measurements (incl. errors).
        header_dict:  dictionary of header items common for all output files.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        `BetaDict` object containing specific `TfsDataFrame` with results.
    """
    meas_and_model_tunes = (
        (tunes[plane]["Q"], tunes[plane]["QM"] % 1)
        if meas_input.compensation == CompensationMode.NONE else
        (tunes[plane]["QF"], tunes[plane]["QFM"] % 1)
    )

    if meas_input.three_bpm_method:
        beta_df = three_bpm_method(meas_input, phase_dict, plane, meas_and_model_tunes)
        error_method = Methods.THREE_BPM
    else:
        beta_df, error_method = n_bpm_method(meas_input, phase_dict, plane, meas_and_model_tunes)

    LOGGER.info(f"Errors from {error_method}")
    rmsbb = stats.weighted_rms(
        beta_df.loc[:, f"{DELTA}{BETA}{plane}"].to_numpy(),
        errors=beta_df.loc[:, f"{ERR}{DELTA}{BETA}{plane}"].to_numpy()) * 100
    LOGGER.info(f" - RMS beta beat: {rmsbb:.3f}%")
    header = _get_header(header_dict, error_method, meas_input.range_of_bpms, rmsbb)
    return beta_df, header


def write(beta_df: pd.DataFrame, header: dict[str, Any], outputdir: str|Path, plane: str) -> None:
    output_path = Path(outputdir) / f"{BETA_NAME}{plane.lower()}{EXT}"
    tfs.write(output_path, beta_df, header, save_index=NAME)


def n_bpm_method(
    meas_input: DotDict,
    phase: pd.DataFrame,
    plane: str,
    meas_and_mdl_tunes: tuple[float, float]
) -> tuple[tfs.TfsDataFrame, str]:
    """
    Calculates betas and alphas using all BPM triplet combinations within a sliding window
    of `range_of_bpms` BPMs centred on each probed BPM. Please refer to theory at
    https://cds.cern.ch/record/2307554.

    For each probed BPM i, a total of m = range_of_boms // 2 neighbors are selected on
    each side. Every valid pair (j, k) within those neighbors contributes to form a BPM
    triplet (i, j, k) from which βi and ⍺i are estimated individually via the function
    ``calculate_beta_alpha_from_single_combination``. The estimates of all combinations
    are combined with covariant weighting using the analytical covariance matrix:

        V_β = T · Σ · Tᵀ

    where T is the Jacobian of (βi, ⍺i) with respect to all error sources, and Σ is the
    diagonal matrix of errors sources variances (phase noise + systematic lattice errors).
    See ``_covariant_weighting`` for the combination step.

    The best-knowledge model is used for the beta/alpha/phase-advance values fed into
    the error propagation (Jacobian), while the normal model provides the output reference
    columns ("MDL" suffix).  If no best-knowledge model is available, both fall back to the
    same model.

    Args:
        meas_input: `OpticsInput` object with optics CLI options.
        phase: phase matrices of measurement with errors and model tfs (bpm x bpm).
        plane: marking the horizontal or vertical plane, **X** or **Y**.
        meas_and_mdl_tunes: measured and model tunes.

    Returns:
        `TfsDataFrame` containing betas and alfas from phase, as well as their errors.
    """
    # Determine the range of BPMs to draw combinations from (BPM triplets)
    n_bpms: int = meas_input.range_of_bpms
    n_bpms_phases: int = len(phase[MEASUREMENT].index)
    if n_bpms_phases < n_bpms:
        LOGGER.warning(
            f"Found {n_bpms_phases} BPMs, but {n_bpms} "
            "were requested in N-BPM method. Using all available BPMs instead,"
            "the results will still be correct."
        )
        n_bpms = n_bpms_phases

    # Ensure at least 3 since we need BPM triplets
    if n_bpms < 3:
        raise ValueError(
            "At least 3 BPMs are required for N-BPM method!"
            f"Instead a range of {n_bpms} was requested."
        )

    # Get the accelerator elements including systematic errors, then
    # (best knowledge) model values and tunes
    elements, error_method = get_elements_with_errors(meas_input, plane)
    beta_df = _get_filtered_model_df(meas_input, phase, plane)
    bk_model = _get_filtered_model_df(meas_input, phase, plane, best=True)
    tune, mdltune = meas_and_mdl_tunes

    nbpms: int = len(bk_model.index)
    n_comb: int = np.zeros(nbpms, dtype=int)

    # Prepare array for results - this one will contain the
    # [beta, betaerr, alfa, alfaerr] for each probed BPM
    betas_alfas = np.zeros((len(phase[MEASUREMENT].index), 4))

    m = int(n_bpms / 2)  # half window: probed BPM has m neighbors on each side
    loc_range = np.arange(-m, m + 1)  # relative indices [-m, ..., 0, ..., m] 0 is the probed BPM

    # Phase advances are stored in units of 2π, convert to radiants for cotangent calculations
    phases_meas = phase[MEASUREMENT] * PI2
    phases_err = phase[f"{ERR}{MEASUREMENT}"] * PI2
    phases_err.where(phases_err.notnull(), 1, inplace=True)  # replace NaN errors with 1

    # We go find combinations within the given range for all BPMs in the model
    for indx, probed_bpm_name in enumerate(bk_model.index):
        # Find the first and last BPMs in the range
        indx_el_first: int = elements.index.get_loc(bk_model.index[(indx - m) % nbpms])
        indx_el_last: int = elements.index.get_loc(bk_model.index[(indx + m) % nbpms])
        mu_column: str = f"{PHASE_ADV}{plane}"

        # We build a sliding window of (2m+1) BPMs centered on the probed one.
        # For BPMs near the start/end of the ring the window wraps around. The
        # tune shift (Q·2π) is subtracted / added to keep phases monotonically
        # increasing across the wrap boundary
        # fmt: off
        if indx < m:
            outer_meas_phase_adv = pd.concat((phases_meas.iloc[indx, nbpms + indx - m :] - tune * PI2, phases_meas.iloc[indx, : indx + m + 1]))
            outer_meas_err = pd.concat((phases_err.iloc[indx, nbpms + indx - m :], phases_err.iloc[indx, : indx + m + 1]))
            outer_mdl_ph = np.concatenate((bk_model.iloc[nbpms + indx - m :][mu_column] - mdltune, bk_model.iloc[: indx + m + 1][mu_column])) * PI2
            outer_elmts = pd.concat((elements.iloc[indx_el_first:], elements.iloc[: indx_el_last + 1]))
            outer_elmts_ph = np.concatenate((elements.iloc[indx_el_first:][mu_column] - mdltune, elements.iloc[: indx_el_last + 1][mu_column])) * PI2
        elif indx + m >= nbpms:
            outer_meas_phase_adv = pd.concat((phases_meas.iloc[indx, indx - m :], phases_meas.iloc[indx, : indx + m + 1 - nbpms] + tune * PI2))
            outer_meas_err = pd.concat((phases_err.iloc[indx, indx - m :], phases_err.iloc[indx, : indx + m + 1 - nbpms]))
            outer_mdl_ph = np.concatenate((bk_model.iloc[indx - m :][mu_column], bk_model.iloc[: indx + m + 1 - nbpms][mu_column] + mdltune)) * PI2
            outer_elmts = pd.concat((elements.iloc[indx_el_first:], elements.iloc[: indx_el_last + 1]))
            outer_elmts_ph = np.concatenate((elements.iloc[indx_el_first:][mu_column], elements.iloc[: indx_el_last + 1][mu_column] + mdltune)) * PI2
        else:
            outer_meas_phase_adv = phases_meas.iloc[indx, indx + loc_range]
            outer_meas_err = phases_err.iloc[indx, indx + loc_range]
            outer_mdl_ph = bk_model.iloc[indx + loc_range][mu_column].to_numpy() * PI2
            outer_elmts = elements.iloc[indx_el_first : indx_el_last + 1]
            outer_elmts_ph = elements.iloc[indx_el_first : indx_el_last + 1][mu_column].to_numpy() * PI2
        # fmt: on

        # sin²(φ_l - φ_mdl_j): (n_elements × n_bpms_window) matrix of squared sines of the
        # model phase advance from every element l to every window BPM j.
        sin_squared_elements = np.square(
            np.sin(outer_elmts_ph[:, np.newaxis] - outer_mdl_ph[np.newaxis, :])
        )

        with np.errstate(divide="ignore"):
            # cot(φ₁ⱼ): measured cotangents for each window BPM j, phase from probed BPM
            cot_meas = 1.0 / np.tan(outer_meas_phase_adv.to_numpy())
            # cot(φ_mdl_1j): model cotangents relative to the probed BPM's model phase (index m)
            cot_model = 1.0 / np.tan(outer_mdl_ph - outer_mdl_ph[m])

        # Numerical stability filter: discard BPMs whose phase advance is too close to
        # a multiple of π (|cot| > COT_THRESHOLD ≈ π/ZERO_THRESHOLD), where the cotangent
        # diverges and derivatives become unreliable (paper Sec. III, stability criterion)
        patter = (np.abs(cot_meas) <= COT_THRESHOLD) & (np.abs(cot_model) <= COT_THRESHOLD)

        # Diagonal of the covariance matrix Σ for all error sources. Note systematic errors
        # have been loaded as squares from error_deffs.txt and quadrupole dK1 was multiplied
        # by length in order to obtain the value for the field gradients (lng = length(outer_elmts))
        #   [0 : n_bpms]                → σ²_φⱼ  (phase measurement variance for each window BPM j)
        #   [off1 : off1+lng]           → σ²_ΔK1L = (dK1 · K1L)² variance (quad gradient field errors)
        #   [off2 : off2+lng]           → σ²_ΔX = dX²  (sextupole transverse misalignment)
        #   [off3 : off3+lng]           → σ²_KdS = (dS · K1L)²  (quad longitudinal misalignment)
        #   [off4 : off4+lng]           → σ²_mKdS  (upstream drift of misaligned quad, thin-lens)
        diag = np.concatenate(
            (
                np.square(outer_meas_err.to_numpy()),
                outer_elmts.loc[:]["dK1"],
                outer_elmts.loc[:]["dX"],
                outer_elmts.loc[:]["KdS"],
                outer_elmts.loc[:]["mKdS"],
            )
        )

        outer_elmts = outer_elmts.rename(columns={f"{BETA}{plane}": "BETA"})

        # Gather all valid BPM pairs (x, y) forming a triplet with the probed BPM (x < y enforces no duplicates)
        # In the conditions from this comprehension, a pair is rejected if:
        #   - either BPM failed the cotangent stability filter above,
        #   - |cot_mdl_x - cot_mdl_y| ≤ ZERO_THRESHOLD (near-degenerate model denominator), or
        #   - cot_mdl and cot_meas differences have opposite signs (unphysical: would invert β sign)
        index_tuples = [
            (x, y)
            for x in loc_range[patter] + m
            for y in loc_range[patter] + m
            if (x < y)
            and (abs(cot_model[x] - cot_model[y]) > ZERO_THRESHOLD)
            and (np.sign(cot_model[x] - cot_model[y]) * np.sign(cot_meas[x] - cot_meas[y]) > 0)
        ]

        # We allocate arrays for the covariance matrices & results for beta and alpha.
        # As per docstring these are V_β = T_β · Σ · T_βᵀ  and  V_α = T_α · Σ · T_αᵀ
        # Each row of is the gradient ∇β (or ∇α) for one triplet combination
        # V_β[i,j] = covariance between the β estimates from triplets i and j
        mat_t_beta = np.zeros((len(index_tuples), len(diag)))
        mat_t_alpha = np.zeros((len(index_tuples), len(diag)))
        betas = np.empty(len(index_tuples))
        alphas = np.empty(len(index_tuples))

        # For each given combination we apply the calculations to determine
        # beta, alpha and the covariance matrices terms. See that function.
        for i, c in enumerate(index_tuples):
            betas[i], alphas[i], mat_t_beta[i], mat_t_alpha[i] = (
                calculate_beta_alpha_from_single_combination(
                    c,
                    sin_squared_elements,
                    outer_elmts,
                    cot_model,
                    cot_meas,
                    outer_meas_phase_adv,
                    probed_bpm_name,
                    bk_model.at[probed_bpm_name, f"{BETA}{plane}"],
                    bk_model.at[probed_bpm_name, f"{ALPHA}{plane}"],
                    meas_input.range_of_bpms,
                )
            )

        # Now we have the values, we move on to the error calculations from the covariance terms
        # Zero-variance entries (error sources not defined for this window) are excluded before
        # building the covariance matrix to avoid degenerate rows/columns
        mask = diag != 0
        mat_diag = np.diag(diag[mask])  # make that a diagonal matrix (we only had diagonal terms)

        # Below we compute the covariance matrices for β and ⍺ - explained for β below
        # V_β = T_β[:,active] · Σ_active · T_β[:,active]ᵀ  — covariance matrix of all β estimates
        # Off-diagonal entries V_β[i,j] represent correlation between estimates i and j arising
        # from shared error sources (e.g. two BPM triplets using the same quadrupole)
        mat_v_beta = np.dot(mat_t_beta[:, mask], np.dot(mat_diag, np.transpose(mat_t_beta[:, mask])))
        mat_v_alpha = np.dot(mat_t_alpha[:, mask], np.dot(mat_diag, np.transpose(mat_t_alpha[:, mask])))

        if np.any(mat_v_beta) and np.any(mat_v_alpha):
            # Combine all triplet estimates into a single BLUE-weighted β and α with
            # propagated error. See _covariant_weighting for the derivation
            beti, beterr = _covariant_weighting(mat_v_beta, betas)
            alfi, alferr = _covariant_weighting(mat_v_alpha, alphas)
            n_comb[indx] = len(betas)  # how many combinations get weighted for this BPM's error bar
        else:
            # All-zero covariance arises when: no valid BPM triplets passed the patter/index_tuples
            # filters, or all error sources were zero (no phase noise AND no systematic errors).
            # In either case there is no valid estimate for this BPM; skip and leave betas_alfas at 0
            LOGGER.debug(
                f"ValueError or no combinations left at {probed_bpm_name}.\n"
                f"betas:\n{betas}\n"
                f"alphas:\n{alphas}"
            )
            continue

        betas_alfas[indx, :] = np.array([beti, beterr, alfi, alferr])

    # ----- By now we have finished the big loop, got results for all BPMs ----- #

    # Unpack the per-BPM results from the loop into the output DataFrame.
    # BPMs that hit the 'continue' above remain at zero in betas_alfas (as
    # initialised) and are filtered below.
    beta_df[f"{BETA}{plane}"] = betas_alfas[:, 0]
    beta_df[f"{ERR}{BETA}{plane}"] = betas_alfas[:, 1]
    beta_df[f"{ALPHA}{plane}"] = betas_alfas[:, 2]
    beta_df[f"{ERR}{ALPHA}{plane}"] = betas_alfas[:, 3]
    beta_df["NCOMB"] = n_comb

    # Now we do some quality filtering

    # Drop BPMs for which the loop produced no valid combinations
    invalid_mask = beta_df["NCOMB"] == 0
    if np.any(invalid_mask):
        LOGGER.debug(f"No valid combinations for BPMs: {list(beta_df.index[invalid_mask])}.")
    beta_df = beta_df.loc[beta_df["NCOMB"] > 0]

    # Drop BPMs with a non-physical negative beta (occurs with very noisy phase advances)
    beta_df = beta_df.loc[beta_df[f"{BETA}{plane}"] > 0]

    # Drop BPMs where the estimated error exceeds both the measured and model beta.
    # Keeping at least one comparison to β_mdl guards against cases where β_meas
    # itself is already severely distorted somehow.
    too_high_error_mask = np.logical_or(
        beta_df[f"{BETA}{plane}"] > beta_df[f"{ERR}{BETA}{plane}"],
        beta_df[f"{BETA}{plane}{MDL}"] > beta_df[f"{ERR}{BETA}{plane}"],
    )
    beta_df = beta_df.loc[too_high_error_mask]

    # We add the "DELTA{...}" columns as deviations from the model
    # values and then we return
    beta_df = _calc_and_add_delta_columns(beta_df, plane)
    return beta_df, error_method


def get_elements_with_errors(meas_input: DotDict, plane: str) -> tuple[pd.DataFrame, str]:
    """
    Loads the accelerator lattice elements with their systematic error variances,
    ready to be windowed per probed BPM in ``n_bpm_method``.

    Reads the machine-specific error definition file (``error_defs_file``) and
    populates four variance columns on the element table via ``_assign_uncertainties``:
    ``dK1``, ``dX``, ``KdS``, ``mKdS``.

    Only elements with at least one non-zero error source are retained (see
    ``_assign_uncertainties`` for the filtering logic). The returned ``error_method``
    string is used downstream to label the output file header and the RMS beta-beat
    log message.

    Args:
        meas_input: `OpticsInput` object with optics CLI options. Must have
        ``accelerator.error_defs_file`` set and ``accelerator.elements`` populated.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        Tuple of:
        - elements DataFrame indexed by element name with columns:
            S, K1L, K2L, MU{plane}, BET{plane}, dK1, dX, KdS, mKdS, BPMdS.
        - error_method string: ``Methods.A_NBPM`` if at least one systematic error
        was assigned, ``Methods.NO_ERR`` otherwise.

    Raises:
        OSError: if ``error_defs_file`` is None (no error definition file configured).
    """
    if meas_input.accelerator.error_defs_file is None:
        raise OSError("Error definition file could not be found")

    # Extract only the columns needed for the Jacobian in 'calculate_beta_alpha_from_single_combination':
    # K1L → scaled into dK1 variance;
    # K2L → multiplied into the dX Jacobian entry;
    # MU{plane} and BET{plane} → model phase advances and betas used in the sin² / bet_sin terms.
    # Then _assign_uncertainties attributes the systematic errors from file
    elements = meas_input.accelerator.elements.loc[:, [S, "K1L", "K2L", f"{PHASE_ADV}{plane}", f"{BETA}{plane}"]]
    LOGGER.debug("Accelerator Error Definition")
    elements = _assign_uncertainties(elements, meas_input.accelerator.error_defs_file)

    # Check whether any systematic error was actually assigned to at least one element.
    # dK1, dX and KdS are checked but mKdS is derived from KdS so it needs no separate check.
    # BPMdS is stored but not currently included in the diag vector so not checked.
    errors_assigned = (
        len(elements["dK1"].to_numpy().nonzero()[0])
        + len(elements["dX"].to_numpy().nonzero()[0])
        + len(elements["KdS"].to_numpy().nonzero()[0])
    ) > 0
    if not errors_assigned:
        LOGGER.warning(
            "No systematic errors were given or no element was found for the given "
            "error definitions. The systematic lattice errors are not used."
        )

    # Assign the error method (remember we're only called for analytical N-BPM):
    #   A_NBPM = full analytical method with systematic errors
    #   NO_ERR = only phase measurement uncertainties propagated (reduced accuracy)
    error_method = Methods.A_NBPM if errors_assigned else Methods.NO_ERR
    return elements, error_method


def calculate_beta_alpha_from_single_combination(
    c: tuple[int, int],
    sin_squared_elements: np.ndarray,
    outer_elmts: pd.DataFrame,
    cot_model: np.ndarray,
    cot_meas: np.ndarray,
    outer_meas_phase_adv: pd.Series,
    probed_bpm_name: str,
    betmdl1: float,
    alfmdl1: float,
    range_of_bpms: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Computes β₁ and α₁ for one BPM triplet (probed BPM 1, reference BPMs x and y)
    and builds the corresponding rows of the Jacobian matrices T_β and T_α used for
    covariance propagation in ``n_bpm_method``.

    Note about the Jacobian structure (Eq. 22): T = (T^φ  T^K  T^s)
    The full Jacobian is split into three blocks: phase uncertainty T^φ,
    quadrupole field errors T^K and BPM longitudinal misalignments T^s.
    Sextupole horizontal misalignments enter into T^K via Eq (13-14) as
    effective quadrupole errors due to feed-down. Here K2L comes into play.
    Longitudinal quadrupole misalignments are decomposed into two thin-lens
    elements at the quadrupole edges (section II.B, Fig. 3): forward (KdS)
    and backward (mKdS).

    The returned `betaline` and `alfaline` are 1D vectors of length (2m+1, 4*n_elements),
    each one being a row of T_β / T_α. Their layout matches the `diag` vector in
    ``n_bpm_method`` (see below for the blocks).

    Args:
        c: tuple, pair of relative indices (ix, iy) for the two reference BPMs of this triplet.
            Indices run 0 ... 2m with the probed BPM at position m.
        sin_squared_elements: (n_elements * n_bpms_window) array of squared sines of the
            model phase advance from every element l to every window BPM j. In there,
            entry [l, j] = sin²(φ_l - φ_mdl_j). Pre-computed in ``n_bpm_method``.
        outer_elmts: DataFrame of lattice elements in the window, which containes the BETA,
            K1L, K2L, dK1, dX, KdS and mKdS column (after loading systematic errors).
        cot_model: 1-D array of model cotangents cot(φ_mdl_1j - φ_mdl_11), length 2m+1;
            where entry m is - by construction (probed BPM to itself).
        cot_meas: 1D array of model cotangents cot(φ_meas_1j), of length 2m+1.
        outer_meas_phase_adv: measured phase advances (radians) from the probed BPM
            to each BPM in the window (index carries BPM names used for element lookup).
        probed_bpm_name: The name of BPM 1 whose β and α are being estimated.
        betmdl1 (float): model β at the probed BPM.
        alfmdl1 (float): model α at the probed BPM.
        range_of_bpms: total window width.

    Returns:
        A tuple (β₁, α₁, betaline, alfaline) where betaline and alfaline are the
        Jacobian rows for this combination, to be stacked into the complete Jacobian
        matrices T_β and T_α inside of ``n_bpm_method``.
    """
    m = int(range_of_bpms / 2)  # half window: probed BPM has m neighbors on each side
    ix = c[0]  # index of BPM i
    iy = c[1]  # index of BPM y
    dif_cot_model = cot_model[ix] - cot_model[iy]

    # Beta estimate at probed BPM for this triplet (Eq. 2)
    dif_cot_meas = cot_meas[ix] - cot_meas[iy]  # numerator term: cot(φ_meas_bpmx) − cot(φ_meas_bpmy)
    denom = dif_cot_model / betmdl1  # denominator term: (cot(φ_mdl_bpmx) − cot(φ_mdl_bpmy)) / β_mdl
    beta_i = dif_cot_meas / denom    # β₁ = β_mdl · Δcot_meas / Δcot_mdl

    # Alpha estimate at probed BPM for this triplet - derived from transport matrix ratio
    avg_cot_model = (cot_model[ix] + cot_model[iy]) / 2
    denomalf = 2 * (avg_cot_model + alfmdl1)  # denominator term: cot(φ_mdl_bpmx) + cot(φ_mdl_bpmy) + 2α_mdl
    avg_cot_meas = (cot_meas[ix] + cot_meas[iy]) / 2
    alfa_i = 0.5 * (denomalf * dif_cot_meas / dif_cot_model - 2 * avg_cot_meas)

    lng: int = len(outer_elmts)

    # ----- Now the Jacobian rows for beta and alpha

    # Sign factors for the Jacobian entries, corresponding to A_{ij}(λ) in paper Eq. (24)
    #   fac1 = +1 if BPM j (ix) is to the left  of probed BPM (ix < m)
    #   fac1 = -1 if BPM j (ix) is to the right of probed BPM (ix > m)
    # (and vice versa for fac2 / BPM k)
    fac1: int = -np.sign(c[0] - m)
    fac2: int = np.sign(c[1] - m)

    # Total length = (2m + 1) phase entries + 4 * n_elements systematic error entries
    line_length: int = 4 * len(outer_elmts.index) + 2 * m + 1
    outer_elmts_bet = outer_elmts.loc[:, "BETA"].to_numpy()
    outer_el_k2 = outer_elmts.loc[:, "K2L"].to_numpy()
    betaline = np.zeros(line_length)
    alfaline = np.zeros(line_length)

    # # Locate BPMs j, k and probed BPM i in the element list, then derive the
    # contiguous slices of elements that lie between each reference BPM and the
    # probed BPM.  These slices are where systematic errors contribute to the
    # phase advance (and then β and α) via the Jacobian.
    mloc: int = outer_elmts.index.get_loc(probed_bpm_name)  # at m is the probed BPM
    xloc_r: int = outer_elmts.index.get_loc(outer_meas_phase_adv.index[ix])  # at x is first BPM in the triplet
    yloc_r: int = outer_elmts.index.get_loc(outer_meas_phase_adv.index[iy])  # at y is last BPM in the triplet

    # Compute sin²(φ_mdl_bpmx − φ_mdl_1) and sin²(φ_mdl_bpmy − φ_mdl_1)
    # which show up in T^φ (Eq. 8) and T^K (Eq. 23) respectively
    denom_sinx = sin_squared_elements[xloc_r, m]
    denom_siny = sin_squared_elements[yloc_r, m]
    # Element index ranges between each BPM in the triplet and the probed BPM
    xmloc1, xmloc2 = min(xloc_r, mloc), max(xloc_r, mloc)
    ymloc1, ymloc2 = min(yloc_r, mloc), max(yloc_r, mloc)

    # Get betas and sin for the elements in the slices: slice the
    # pre-computed sin² matrix and model arrays to the elements
    # in each [reference BPM <-> probed BPM] interval
    elem_ph_xa = sin_squared_elements[xmloc1:xmloc2, ix]
    elem_ph_ya = sin_squared_elements[ymloc1:ymloc2, iy]
    elem_beta_xa = outer_elmts_bet[xmloc1:xmloc2]
    elem_beta_ya = outer_elmts_bet[ymloc1:ymloc2]
    elem_k2_xa = outer_el_k2[xmloc1:xmloc2]
    elem_k2_ya = outer_el_k2[ymloc1:ymloc2]

    # Common factor for all systematic error Jacobian entries in T^K
    bet_sin_ix = elem_ph_xa * elem_beta_xa / (denom_sinx * denom)
    bet_sin_iy = elem_ph_ya * elem_beta_ya / (denom_siny * denom)

    # Index offsets into betaline/alfaline for each error source block:
    # T = (T^φ | T^K_dK1 | T^K_dX | T^K_KdS | T^K_mKdS)
    # Phase entries occupy [0 : range_of_bpms], systematic errors follow
    off1: int = range_of_bpms            # T^K: dK1 block — quadrupole field errors (Eq. 23)
    off2: int = range_of_bpms + lng      # T^K·K₂L: dX block — sextupole transverse misalignment (Eqs. 13-14)
    off3: int = range_of_bpms + 2 * lng  # T^K: KdS  — quad longitudinal misalignment, forward element (Sec. II.B)
    off4: int = range_of_bpms + 3 * lng  # T^K: mKdS — quad longitudinal misalignment, backward/upstream element (Sec. II.B)

    # Now below is the big summation over the contributing terms for the errors,
    # incl. phase uncertainty and systematic errors.

    # apply phase uncertainty - T^φ entries: ∂β/∂φⱼ and ∂β/∂φₖ  (Eq. 8)
    betaline[ix] = -1 / (denom_sinx * denom)
    betaline[iy] = 1 / (denom_siny * denom)
    # apply quadrupolar field uncertainty
    betaline[xmloc1 + off1 :xmloc2 + off1] += fac1 * bet_sin_ix
    betaline[ymloc1 + off1 :ymloc2 + off1] += fac2 * bet_sin_iy
    # apply sextupole transverse misalignment
    betaline[xmloc1 + off2 : xmloc2 + off2] += fac1 * elem_k2_xa * bet_sin_ix  # the K2L shows up here
    betaline[ymloc1 + off2 : ymloc2 + off2] += fac2 * elem_k2_ya * bet_sin_iy  # the K2L shows up here
    # apply quadrupole longitudinal misalignments
    betaline[xmloc1 + off3 : xmloc2 + off3] += fac1 * bet_sin_ix
    betaline[ymloc1 + off3 : ymloc2 + off3] += fac2 * bet_sin_iy
    betaline[xmloc1 + off4 : xmloc2 + off4] -= fac1 * bet_sin_ix
    betaline[ymloc1 + off4 : ymloc2 + off4] -= fac2 * bet_sin_iy

    # The same thing is done for the alfaline

    # apply phase uncertainty
    alfaline[ix] = -1 / (denom_sinx * denom * betmdl1) * denomalf + 1 / denom_sinx
    alfaline[iy] = 1 / (denom_siny * denom * betmdl1) * denomalf + 1 / denom_siny
    # apply quadrupolar field uncertainty
    alfaline[xmloc1 + off1 :xmloc2 + off1] += fac1 * (.5 * (bet_sin_ix * denomalf + bet_sin_ix / betmdl1 * dif_cot_meas))
    alfaline[ymloc1 + off1 :ymloc2 + off1] += fac2 * (.5 * (bet_sin_iy * denomalf + bet_sin_iy / betmdl1 * dif_cot_meas))
    # apply sextupole transverse misalignment
    alfaline[xmloc1 + off2 : xmloc2 + off2] += fac1 * elem_k2_xa * bet_sin_ix
    alfaline[ymloc1 + off2 : ymloc2 + off2] += fac2 * elem_k2_ya * bet_sin_iy
    # apply quadrupole longitudinal misalignments
    alfaline[xmloc1 + off3 : xmloc2 + off3] += fac1 * (.5 * elem_k2_xa * (bet_sin_ix * denomalf + bet_sin_ix / betmdl1 * dif_cot_meas))
    alfaline[ymloc1 + off3 : ymloc2 + off3] += fac2 * (.5 * elem_k2_ya * (bet_sin_iy * denomalf + bet_sin_iy / betmdl1 * dif_cot_meas))
    alfaline[xmloc1 + off4 : xmloc2 + off4] -= fac1 * (.5 * (bet_sin_ix * denomalf + bet_sin_ix / betmdl1 * dif_cot_meas))
    alfaline[ymloc1 + off4 : ymloc2 + off4] -= fac2 * (.5 * (bet_sin_iy * denomalf + bet_sin_iy / betmdl1 * dif_cot_meas))

    return beta_i, alfa_i, betaline, alfaline


def _covariant_weighting(mat: np.ndarray, col: np.ndarray) -> typle[float, float]:
    """
    Combines multiple estimates into a single BLUE (Best Linear Unbiased Estimator)
    weighted value with propagated uncertainty (Analytical N-BPM paper Appendix A,
    Eqs. A21-A22).

    Given n_comb individual β (or α) estimates and their full covariance matrix V,
    the optimal weights are:

        g_i = (Σ_k V⁻¹_{ik}) / (Σ_{j,k} V⁻¹_{jk})      (Eq. A21)

    and the BLUE estimate and its variance are:

        β̂ = Σ_i g_i β_i                                (Eq. A22)
        σ²_β̂ = gᵀ V g

    The pseudo-inverse (pinv) is used instead of a regular inverse to handle
    near-singular covariance matrices (e.g. highly correlated triplets), with
    conditioning threshold RCOND.

    Args:
        mat: (n_comb * n_comb) covariance matrix V of the individual estimates.
        col: 1D array of length n_comb holding the individual β or α estimates.

    Returns:
        Tuple (estimate, uncertainty) where estimate is the BLUE-weighted scalar
        and uncertainty is its propagated 1σ error.

    Raises:
        ValueError: if the sum of V⁻¹ is zero (degenerate / all-zero covariance matrix).
    """
    mat_inv = np.linalg.pinv(mat, rcond=RCOND)  # V⁻¹, pseudo-inverse for numerical stability
    wb = np.sum(mat_inv, axis=1)                # g̃_i = Σ_k V⁻¹_{ik}: unnormalized BLUE weights (row sums)
    mat_inv_sum = np.sum(wb)                    # Σ_{j,k} V⁻¹_{jk}: normalization constant

    if mat_inv_sum == 0:
        raise ValueError("Covariance weighting failed: the covariance matrix is degenerate")

    # BLUE estimate: β̂ = g̃ᵀ col / Σ V⁻¹  (Eq. A22)
    # Propagated variance: σ²_β̂ = g̃ᵀ V g̃ / (Σ V⁻¹)²
    estimate: float = float(np.dot(wb.T, col) / mat_inv_sum)
    variance: float = np.sqrt(np.dot(wb.T, np.dot(mat, wb)) / mat_inv_sum ** 2)
    return estimate, variance


def _assign_uncertainties(twiss_full: pd.DataFrame, errordefspath: Path | str) -> pd.DataFrame:
    """
    Reads the definition file for systematic uncertainties and populates variance
    columns on ``twiss_full``, then filters out elements with no uncertainty assigned

    All created output columns store **variances** (σ²), not standard deviations (σ).
    The error definition file provides rms σ values; and squaring is done in here.

    Here are the output columns (variances) and how they map to error sources:

    +----------+------------------------------------------------+-----------------------------------+
    | Column   | Physical meaning                               | Formula                           |
    +----------+------------------------------------------------+-----------------------------------+
    | dK1      | Quadrupole relative field gradient error       | σ²_ΔK1L = (dK1 · K1L)²            |
    | dX       | Sextupole transverse (horizontal) misalignment | σ²_ΔX = dX²                       |
    | KdS      | Quadrupole longitudinal misalignment (forward) | σ²_KdS = (dS · K1L)²              |
    | mKdS     | Upstream drift of longitudinally misaligned    | σ²_mKdS = KdS of previous elem    |
    |          | quad (thin-lens decomposition, Sec. II.B)      |                                   |
    | BPMdS    | BPM longitudinal misalignment                  | σ²_BPMdS = dS²  (stored, unused?) |
    +----------+------------------------------------------------+-----------------------------------+

    The PATTERN column in the error file is a regex matched against element names,
    or a `key:<name>` literal for a single named element. The MAINFIELD column
    distinguishes elements types.

    Only elements with `UNC=True` are returned. This UNC flag is also set on the
    element immediately preceding a quadrupole with dK1 > 0 (to capture the upstream
    drift needed for the mKdS thin-lens term in Sec. II.B, Fig. 3).

    Args:
        twiss_full: lattice element DataFrame (from ``accelerator.elements``),
            indexed by element name, containing at least K1L and K2L columns.
        errordefspath: path to the TFS systematic uncertainties definition file.

    Returns:
        A filtered copy of ``twiss_full`` containing only elements with at least one
        non-zero variance, with added columns dK1, dX, KdS, mKdS, BPMdS (all σ²).
    """
    LOGGER.debug("Loading systematic uncertainties")
    errdefs = tfs.read(errordefspath)
    twiss_full = twiss_full.assign(UNC=False, dK1=0.0, KdS=0.0, mKdS=0.0, dX=0.0, BPMdS=0.0)

    # Loop over uncertainty definitions, each row matching a set of elements via
    # PATTERN, and assign the corresponding variance to the appropriate column.
    # Also set UNC to true on relevant elements.
    for indx in errdefs.index:
        patt = errdefs.loc[indx, "PATTERN"]
        if patt.startswith("key:"):
            LOGGER.debug(f"creating uncertainty information for {patt}")
            mask = patt.split(":")[1]
        else:
            reg = re.compile(patt)
            LOGGER.debug(f"creating uncertainty information for RegEx {patt}")
            mask = twiss_full.index.str.contains(reg)

        # Quadrupole field error variance: σ²_ΔK1L = (relative_error · K1L)²
        twiss_full.loc[mask, "dK1"] = (errdefs.loc[indx, "dK1"] * twiss_full.loc[mask, "K1L"]) ** 2
        # Sextupole transverse misalignment variance: σ²_ΔX = dX² (used later with K2L for feeddown)
        twiss_full.loc[mask, "dX"] = errdefs.loc[indx, "dX"] ** 2
        if errdefs.loc[indx, "MAINFIELD"] == "BPM":
            # BPM longitudinal misalignment
            twiss_full.loc[mask, "BPMdS"] = errdefs.loc[indx, "dS"]**2
        else:
            # Quadrupole longitudinal misalignment variance: σ²_KdS = (dS · K1L)²
            twiss_full.loc[mask, "KdS"] = (errdefs.loc[indx, "dS"] * twiss_full.loc[mask, "K1L"]) ** 2
        twiss_full.loc[mask, "UNC"] = True  # flag uncertainty assigned

    # in case of quadrupole longitudinal misalignments, the element (DRIFT) in front of the
    # misaligned quadrupole will be used for the thin lens approximation of the misalignment:
    # mKdS is the KdS of the preceding element: in the thin-lens decomposition of a longitudinally
    # misaligned quadrupole (Sec. II.B, Fig. 3), the upstream drift element carries the backward
    # thin-lens kick, so its variance equals that of the quad it precedes
    twiss_full["mKdS"] = np.roll(twiss_full.loc[:]["KdS"], 1)
    # Also flag the element immediately before any quad with field errors as UNC=True,
    # so the mKdS entry of that element is included in the window slicing
    twiss_full.loc[:, "UNC"] = np.logical_or(
        abs(np.roll(twiss_full.loc[:, "dK1"], -1)) > 1.0e-12, twiss_full.loc[:, "UNC"]
    )
    LOGGER.debug("Finished assigning variances from systematic uncertainties")
    return twiss_full.loc[twiss_full["UNC"]]


def _get_header(header_dict: dict[str, Any], error_method: str, range_of_bpms: int, rmsbb: float) -> dict[str, Any]:
    header = header_dict.copy()
    header['BetaAlgorithmVersion'] = VERSION
    header['RCond'] = RCOND
    header['RangeOfBPMs'] = "Adjacent" if error_method == Methods.THREE_BPM else range_of_bpms
    header['ErrorsFrom:'] = error_method
    header["RMS_BETABEAT"] = f"{rmsbb:.3f} %"
    return header


def three_bpm_method(
    meas_input: DotDict,
    phase: pd.DataFrame,
    plane: str,
    meas_and_mdl_tunes: tuple[float, float],
) -> pd.DataFrame:
    """
    Calculates betas and alphas using three adjacent-BPM triplet combinations
    (Castro, Ph.D. Thesis, University of Valencia, 1996). Unlike in the
    ``n_bpm_method`` this uses only the immediately neighbouring BPMs
    and propagates phase measurement uncertainties only (no systematic
    uncertainties for lattice errors).

    For each probed BPM i, three triplets are formed and their β/β_mdl estimates
    are averaged arithmetically (see ``bet_frac`` below):
        - xxxABBx → probed BPM followed by two forward neighbours
        - xBBAxxx → two backward neighbours followed by probed BPM
        - xxBABxx → probed BPM flanked by skip-one neighbours

    The input phase advance data from ``phase["MEAS"]``, ``phase["MODEL"]``,
    ``phase["ERRMEAS"]`` (from ``get_phases``) are of the form:

    +----------+----------+----------+----------+----------+
    |          |   BPM1   |   BPM2   |   BPM3   |   BPM4   |
    +----------+----------+----------+----------+----------+
    |   BPM1   |    0     |  phi_21  |  phi_31  |  phi_41  |
    +----------+----------+----------+----------+----------+
    |   BPM2   |  phi_12  |     0    |  phi_32  |  phi_42  |
    +----------+----------+----------+----------+----------+
    |   BPM3   |  phi_13  |  phi_23  |    0     |  phi_43  |
    +----------+----------+----------+----------+----------+

    and ``tilt_slice_matrix(matrix, shift, slice, tune)`` brings it into the form:

    +-----------+--------+--------+--------+--------+
    |           |  BPM1  |  BPM2  |  BPM3  |  BPM4  |
    +-----------+--------+--------+--------+--------+
    | BPM_(i-1) | phi_1n | phi_21 | phi_32 | phi_43 |
    +-----------+--------+--------+--------+--------+
    | BPM_i     |    0   |    0   |    0   |    0   |
    +-----------+--------+--------+--------+--------+
    | BPM_(i+1) | phi_12 | phi_23 | phi_34 | phi_45 |
    +-----------+--------+--------+--------+--------+

    The computed ``cot_phase_*_shift1`` gives:

    +-----------------------------+-----------------------------+-----------------------------+
    | cot(phi_1n) - cot(phi_1n-1) |  cot(phi_21) - cot(phi_2n)  |   cot(phi_32) - cot(phi_31) |
    +-----------------------------+-----------------------------+-----------------------------+
    |         NaN                 |         NaN                 |         NaN                 |
    +-----------------------------+-----------------------------+-----------------------------+
    |         NaN                 |         NaN                 |         NaN                 |
    +-----------------------------+-----------------------------+-----------------------------+
    |  cot(phi_13) - cot(phi_12)  |  cot(phi_24) - cot(phi_23)  |   cot(phi_35) - cot(phi_34) |
    +-----------------------------+-----------------------------+-----------------------------+

    -> for the combination xxxABBx: first row,
    -> for the combination xBBAxxx: fourth row,
    -> for the combination xxBABxx: second row of ``cot_phase_*_shift2``.

    Args:
        meas_input: `OpticsInput` object with optics CLI options.
        phase: phase matrices of measurement with errors and model tfs (bpm x bpm).
        plane: marking the horizontal or vertical plane, **X** or **Y**.
        meas_and_mdl_tunes: measured and model tunes.

    Returns:
        `TfsDataFrame` containing betas and alfas from phase.
    """
    # Ensure at least 3 BPMs so we can form one triplet
    if len(phase[MEASUREMENT].index) < 3:
        raise ValueError(
            "At least 3 BPMs are required for 3-BPM method. "
            f"Instead only {len(phase[MEASUREMENT].index)} were found in input."
        )

    tune, mdltune = meas_and_mdl_tunes
    beta_df = _get_filtered_model_df(meas_input, phase, plane)

    # Tilt and slice so that for each probed BPM i the 5 rows hold (see docstring):
    #   row 0: φ(i-2→j)   two BPMs back
    #   row 1: φ(i-1→j)   one BPM back
    #   row 2: 0,         probed BPM i
    #   row 3: φ(i+1→j),  one BPM forward
    #   row 4: φ(i+2→j),  two BPMs forward
    # (see _tilt_slice_matrix for the rearrangement). Convert to radians
    tilted_meas = _tilt_slice_matrix(phase[MEASUREMENT].to_numpy(copy=True), 2, 5, tune) * PI2
    tilted_model = _tilt_slice_matrix(phase[MODEL].to_numpy(copy=True), 2, 5, mdltune) * PI2
    tilted_errmeas = _tilt_slice_matrix(phase[f"{ERR}{MEASUREMENT}"].to_numpy(copy=True), 2, 5, 0) * PI2

    betmdl = beta_df.loc[:, f"{BETA}{plane}{MDL}"].to_numpy()
    alfmdl = beta_df.loc[:, f"{ALPHA}{plane}{MDL}"].to_numpy()

    with np.errstate(divide="ignore"):
        cot_phase_meas = 1 / np.tan(tilted_meas)
        cot_phase_model = 1 / np.tan(tilted_model)

    # Calculate numerators and denominators for far more cases than needed
    #   shift1 are the cases BBA, ABB, AxBB, AxxBB etc. (the used BPMs are adjacent)
    #   shift2 are the cases where the used BPMs are separated by one. only BAB is used for 3-BPM
    # Here EPSILON guards against a zero Δcot_mdl denominator (nearly degenerate triplet)
    cot_phase_meas_shift1 = cot_phase_meas - np.roll(cot_phase_meas, -1, axis=0)
    cot_phase_model_shift1 = cot_phase_model - np.roll(cot_phase_model, -1, axis=0) + EPSILON
    cot_phase_meas_shift2 = cot_phase_meas - np.roll(cot_phase_meas, -2, axis=0)
    cot_phase_model_shift2 = cot_phase_model - np.roll(cot_phase_model, -2, axis=0) + EPSILON

    # Calculate β/β_mdl for each of the 3 triplets, which is Δcot_meas / Δcot_mdl
    # Here bet_frac is their arithmetic mean, so beti = β_mdl · mean(β/β_mdl)
    bet_frac = (
        cot_phase_meas_shift1[0] / cot_phase_model_shift1[0]
        + cot_phase_meas_shift1[3] / cot_phase_model_shift1[3]
        + cot_phase_meas_shift2[1] / cot_phase_model_shift2[1]
    ) / 3

    # Alpha model term: average of (cot φ_mdl,j + cot φ_mdl,k + 2α_mdl) / 2
    # across the 3 combinations
    alf_mdl_term = (
        ((cot_phase_model + np.roll(cot_phase_model, -1, axis=0))[0] + 2.0 * alfmdl)
        + ((cot_phase_model + np.roll(cot_phase_model, -1, axis=0))[3] + 2.0 * alfmdl)
        + ((cot_phase_model + np.roll(cot_phase_model, -2, axis=0))[1] + 2.0 * alfmdl)
    ) / 6.0
    alf_meas_term = (
        ((cot_phase_meas + np.roll(cot_phase_meas, -1, axis=0))[0])
        + ((cot_phase_meas + np.roll(cot_phase_meas, -1, axis=0))[3])
        + ((cot_phase_meas + np.roll(cot_phase_meas, -2, axis=0))[1])
    ) / 6.0

    # multiply the fractions by betmdl and calculate the arithmetic mean
    beti = bet_frac * betmdl
    alfi = bet_frac * alf_mdl_term + alf_meas_term

    # Error propagation, from phase uncertainties
    # calculate errphi_ij^2 / sin^2 phimdl_ij * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_squared_model = tilted_errmeas**2 / np.square(np.sin(tilted_model)) * betmdl
        sin_quadrup_model = tilted_errmeas**2 / np.power(np.sin(tilted_model), 4) * betmdl

    # Square to get variance terms (summed in quadrature across the two BPMs of each triplet)
    sin_squared_model = np.square(sin_squared_model)
    sin_squ_model_shift1 = sin_squared_model + np.roll(sin_squared_model, -1, axis=0) / np.square(cot_phase_model_shift1)
    sin_squ_model_shift2 = sin_squared_model + np.roll(sin_squared_model, -2, axis=0) / np.square(cot_phase_model_shift2)
    sin_quad_model_shift1 = sin_quadrup_model + np.roll(sin_quadrup_model, -1, axis=0) / np.square(cot_phase_model_shift1)
    sin_quad_model_shift2 = sin_quadrup_model + np.roll(sin_quadrup_model, -2, axis=0) / np.square(cot_phase_model_shift2)

    # beterr = (1/3) · sqrt(Σ variance over 3 combinations): propagated σ_β
    beterr = np.sqrt(sin_squ_model_shift1[0] + sin_squ_model_shift1[3] + sin_squ_model_shift2[1]) / 3
    beta_df[f"{BETA}{plane}"] = beti
    beta_df[f"{ERR}{BETA}{plane}"] = beterr
    beta_df[f"{ALPHA}{plane}"] = alfi
    beta_df[f"{ERR}{ALPHA}{plane}"] = alf_mdl_term * beterr / betmdl + 0.5 * np.sqrt(
        sin_quad_model_shift1[0] + sin_quad_model_shift1[3] + sin_quad_model_shift2[1])
    return _calc_and_add_delta_columns(beta_df, plane)


def _calc_and_add_delta_columns(beta_df: pd.DataFrame, plane: str) -> pd.DataFrame:
    """Append beta-beat and alpha-beat columns to dataframe, for both values and errors."""
    beta_df[f"{DELTA}{BETA}{plane}"] = df_rel_diff(beta_df, f"{BETA}{plane}", f"{BETA}{plane}{MDL}")
    beta_df[f"{ERR}{DELTA}{BETA}{plane}"] = df_ratio(beta_df, f"{ERR}{BETA}{plane}", f"{BETA}{plane}{MDL}")
    beta_df[f"{DELTA}{ALPHA}{plane}"] = df_diff(beta_df, f"{ALPHA}{plane}", f"{ALPHA}{plane}{MDL}")
    beta_df[f"{ERR}{DELTA}{ALPHA}{plane}"] = beta_df.loc[:, f"{ERR}{ALPHA}{plane}"].to_numpy()
    return beta_df


def _tilt_slice_matrix(
    matrix: np.ndarray, slice_shift: int, slice_width: int, tune: float = 0.0
) -> np.ndarray:
    """
    Tilts and slices the ``matrix``.
    Tilting means shifting each column upwards one step more than the previous columnns, i.e.

    a a a a a       a b c d
    b b b b b       b c d e
    c c c c c  -->  c d e f
    ...             ...
    y y y y y       y z a b
    z z z z z       z a b c

    After tilting, the matrix is rolled by ``slice_shift`` rows so that the diagonal
    entry (where row index == column index, i.e. the probed BPM itself) lands at row
    ``slice_shift``. Only the first ``slice_width`` rows are returned, giving a local
    neighbourhood of BPMs centred on the probed one.

    The ``tune`` argument corrects for the phase wrap at the ring boundary: entries
    that cross the 0->2π discontinuity are shifted by ±tune*2π to keep phases
    monotonically increasing (applied before tilting).

    .. note::
        Careful: this mutates the provided data. Be sure to provide a mutable
        array, ideally a copy and not a view. In pandas 3.x passing a view
        will error.

    Args:
        matrix: (n_bpms * n_bpms) phase advance matrix to tilt.
        slice_shift: row index of the probed BPM in the output (= half-window size).
        slice_width: number of rows to return (= window size, typically 2*slice_shift + 1).
        tune: fractional tune, used to correct phases across the ring boundary.

    Returns:
        (slice_width * n_bpms) array with phase advances in a rolling neighbourhood.
    """
    invrange = matrix.shape[0] - 1 - np.arange(matrix.shape[0])
    matrix[matrix.shape[0] - slice_shift :, :slice_shift] += tune
    matrix[:slice_shift, matrix.shape[1] - slice_shift :] -= tune
    return np.roll(
        matrix[np.arange(matrix.shape[0]), circulant(invrange)[invrange]], slice_shift, axis=0
    )[:slice_width]


def _get_filtered_model_df(meas_input: DotDict, phase: pd.DataFrame, plane: str, best: bool = False) -> pd.DataFrame:
    """
    Returns the model DataFrame filtered to the measured BPMs, with S, beta, alpha
    and mu columns for the given `plane`.

    When `best=False` (normal model), columns are renamed with the `MDL` suffix for the output.
    When `best=True` (best-knowledge model) columns keep their original names for use in the
    N-BPM method's Jacobian calculations.
    """
    model = _get_accel_model(meas_input, best=best)
    df = model.loc[phase[MEASUREMENT].index, [S, f"{BETA}{plane}", f"{ALPHA}{plane}", f"{PHASE_ADV}{plane}"]]
    if not best:
        df = df.rename(
            columns={
                f"{BETA}{plane}": f"{BETA}{plane}{MDL}",
                f"{ALPHA}{plane}": f"{ALPHA}{plane}{MDL}",
                f"{PHASE_ADV}{plane}": f"{PHASE_ADV}{plane}{MDL}",
            }
        )
    return df


def _get_accel_model(meas_input: DotDict, best: bool) -> pd.DataFrame:
    """
    Returns the best-knowledge model if available and `best=True`, otherwise
    returns the normal model from the input's defined accelerator.
    """
    accel: Accelerator = meas_input.accelerator
    if not best:
        return accel.model

    if accel.model_best_knowledge is None:
        LOGGER.warning("No best knowledge model - using the normal one.")
        return accel.model

    return accel.model_best_knowledge
