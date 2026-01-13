"""
Response MAD-NG
===============

Provides a function to create the responses of beta, phase, dispersion, tune and coupling via MAD-NG derivatives.

The variables under investigation need to be provided as a list (which can be obtained from the accelerator
class).

For now, the response matrix is stored in a hdf5 file.

Differences to the MAD-X implementation
---------------------------------------

Differential Algebra (DA) is used to compute the derivatives of the optics parameters with respect to the
variables (knobs). For most optics parameters (beta, phase, dispersion), a single twiss calculation is performed
with DA enabled, allowing the extraction of derivatives directly from the twiss output.

Coupling derivatives are computed in a separate twiss calculation after perturbing the variables slightly.
This is necessary because the coupling calculation is unstable when there is no coupling present.
By applying a small perturbation to the variables, we ensure that the coupling derivatives can be computed reliably.
This perturbation is a constant value added to all variables during the coupling twiss calculation, and has been tested
for the LHC to provide the most stable results.

By only performing two twiss calculations (one for standard optics and one for coupling),
this approach is significantly more efficient than performing separate calculations for each variable.

:author: Joshua Gray
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from cpymad.madx import Madx
from pymadng import MAD

from omc3.model.accelerators.accelerator import AccElementTypes
from omc3.model.model_creators.manager import CreatorType, get_model_creator_class
from omc3.optics_measurements.constants import (
    BETA,
    DISPERSION,
    F1001,
    F1010,
    NORM_DISPERSION,
    PHASE_ADV,
    TUNE,
)
from omc3.utils.logging_tools import MADX, get_logger

LOG = get_logger(__name__)

if TYPE_CHECKING:
    import logging
    from collections.abc import Sequence
    from pathlib import Path

    from omc3.model.accelerators.accelerator import Accelerator

MADNG_VARMAP = {
    "mu1": f"{PHASE_ADV}X",
    "mu2": f"{PHASE_ADV}Y",
    "betx": f"{BETA}X",
    "bety": f"{BETA}Y",
    "disp1": f"{DISPERSION}X",
    "disp3": f"{DISPERSION}Y",  # check if d2 = dpx?
}

COUPLING_VARMAP = {
    "f1001": F1001,
    "f1010": F1010,
}

MADNG_OPTICS = {
    "disp1": "dx",
    "disp3": "dy",
}

# From testing, this value gives enough stability for the coupling derivatives
# We have to have a delta k as the calculation of the coupling rdts are unstable
# when we don't have coupling.
# But since all knobs are changed at the same time, it is ideal to perturb them
# by as small a value as possible to avoid large changes in the optics.
NG_DELTA_K = 1e-6


class LoggingStream:
    """A stream that logs written data to the logger at MADX level."""

    def __init__(self, logger: logging.Logger, level: int = MADX):
        self.logger = logger
        self.level = level
        self.buffer = []

    def write(self, text: str):
        if text.strip():  # Only log non-empty lines
            self.logger.log(self.level, text.rstrip())

    def flush(self):
        pass


def _initialise_madng_with_logging(
    madx_seq_path: Path, seq_name: str, accel_inst: Accelerator
) -> tuple[MAD, Path]:
    """Initialize MAD-NG with stdout logging to a file in the model directory.

    Args:
        madx_seq_path: Path to the MAD-X sequence file
        seq_name: Name of the sequence
        accel_inst: Accelerator instance

    Returns:
        Tuple of (MAD instance, log file path)
    """
    # Create log file in the model directory
    log_path = accel_inst.model_dir / "madng_response.log"
    LOG.info(f"MAD-NG output will be logged to: {log_path}")

    # Initialize MAD-NG with stdout redirected to log file
    mad = MAD(stdout=str(log_path), redirect_stderr=True, debug=True)
    _load_sequence(mad, str(madx_seq_path.absolute()), seq_name, accel_inst.beam_direction)
    _setup_beam(mad, accel_inst.energy)

    return mad, log_path


def _log_madng_output(log_path: Path):
    """Log MAD-NG output from the log file.

    Args:
        log_path: Path to the log file
    """
    if log_path.exists():
        with log_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    LOG.log(MADX, line)


def _load_sequence(mad: MAD, sequence_file: str, seq_name: str, beam_direction: int):
    """Load a sequence file into MAD-NG (copied from BaseMadInterface)."""
    mad.send(f'MADX:load("{sequence_file}")')
    if mad.MADX[seq_name] == 0:
        raise ValueError(f"Sequence '{seq_name}' not found in MAD file '{sequence_file}'")
    mad.send(f"loaded_sequence = MADX.{seq_name}")
    mad["SEQ_NAME"] = seq_name
    mad.loaded_sequence.dir = beam_direction


def _setup_beam(mad: MAD, beam_energy: float, particle: str = "proton"):
    """Set up beam parameters (copied from BaseMadInterface)."""
    mad.send(
        f'loaded_sequence.beam = beam {{ particle = "{particle}", energy = {beam_energy:.15e} }}'
    )


def create_fullresponse(
    accel_inst: Accelerator,
    variable_categories: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Generate a dictionary containing response matrices for
    beta, phase, dispersion, tune and coupling using MAD-NG derivatives.

    Args:
        accel_inst : Accelerator Instance.
        variable_categories (list): Categories of the variables/knobs to use.

    Returns:
        dict: Dictionary of response DataFrames keyed by optics type (e.g., 'BETAX', 'PHASEX', etc.)
    """
    LOG.info("Creating fullresponse via MAD-NG")

    LOG.debug("Creating MAD-X Sequence to be used in MAD-NG")
    creator_class = get_model_creator_class(accel_inst, CreatorType.NOMINAL)
    creator = creator_class(accel_inst)
    creator.prepare_run()

    # Used by MAD-NG to load the sequence
    madx_seq_path = accel_inst.model_dir / creator.save_sequence_filename
    seq_name = creator.sequence_name

    if madx_seq_path.exists():
        raise FileExistsError(f"Saved sequence file {madx_seq_path} already exists!")

    with Madx(stdout=LoggingStream(LOG)) as madx:
        madx.chdir(str(accel_inst.model_dir.absolute()))
        madx.input(creator.get_base_madx_script())
        madx.input(creator.get_save_sequence_script())

    if not madx_seq_path.exists():
        raise FileNotFoundError(f"Saved sequence file {madx_seq_path} was not created!")

    # Initialize MAD-NG with logging
    mad, log_path = _initialise_madng_with_logging(madx_seq_path, seq_name, accel_inst)

    # Get variables
    variables = accel_inst.get_variables(classes=variable_categories)
    if len(variables) == 0:
        raise ValueError("No variables found! Make sure your categories are valid!")

    LOG.info(f"Computing response for {len(variables)} variables")

    try:
        # Compute twiss with derivatives
        response_dict = _compute_response_with_derivatives(mad, variables, accel_inst)
    except Exception as e:
        LOG.error("Error while computing response with MAD-NG derivatives.")
        _log_madng_output(log_path)
        raise e
    _log_madng_output(log_path)

    # Delete the saved sequence file to clean up
    madx_seq_path.unlink(missing_ok=True)

    return response_dict


def _compute_response_with_derivatives(
    mad: MAD, variables: list[str], accel_inst: Accelerator
) -> dict[str, pd.DataFrame]:
    """
    Compute response matrices using MAD-NG derivatives.

    This function sets up differential algebra in MAD-NG to compute derivatives of optics
    (beta, phase, dispersion, tunes, coupling) with respect to the given variables (knobs).
    It performs two twiss calculations: one for standard optics and one for coupling,
    then extracts and processes the derivative data into response matrices.

    Args:
        mad: MAD-NG instance
        variables: List of variable names (knobs) for which to compute responses
        accel_inst: Accelerator instance providing BPM patterns and other info

    Returns:
        Dictionary of response DataFrames keyed by optics type (e.g., 'BETAX', 'PHASEX', etc.)
    """
    n_vars = len(variables)

    # Define optics parameters for which we want derivatives
    optics_list = list(MADNG_VARMAP.keys())
    coupling_params = list(COUPLING_VARMAP.keys())

    kopt_dict = _create_kopt_dict(optics_list, n_vars)
    flat_opt_list = [item for sublist in kopt_dict.values() for item in sublist]

    coupling_kopt_dict = _create_kopt_dict(coupling_params, n_vars)
    coupling_optics_list = [item for sublist in coupling_kopt_dict.values() for item in sublist]

    dk = dict.fromkeys(variables, NG_DELTA_K)

    # Set up Differential Algebra for derivatives
    mad.send("""
--start-mad
local knob_list = py:recv()
local opt_list = py:recv()
local bpm_pattern = py:recv()
local num_k = #knob_list
local k_ord = 2
local x0 = MAD.damap { nv = 6, np = num_k, no = {k_ord, k_ord, k_ord, k_ord, 1, 1}, po=1, pn=knob_list}
for i, knob in ipairs(knob_list) do
    ! dk[knob] = MADX[knob] + 0 (Converts the knob to a DA variable)
    MADX[knob] = MADX[knob] + x0[knob]
end

-- Observe all elements
local observed in MAD.element.flags
local drift_element in MAD.element
loaded_sequence:deselect(observed)
loaded_sequence:select(observed, {pattern=bpm_pattern})  ! Observe all BPMs
loaded_sequence:select(observed, {pattern="$end"}) ! Also observe END for tunes

-- Compute twiss with derivatives
tws, _ = twiss {
    sequence=loaded_sequence,
    observe=1,
    X0=x0,
    trkopt=opt_list,
    coupling=true, ! Set to true for betx and bety calculation
}

local dk = py:recv()
local coupling_list = py:recv()
-- Apply delta k to knobs so that the coupling derivatives can be computed too
for i, knob in ipairs(knob_list) do
    MADX[knob] = MADX[knob] + dk[knob]
end

tws_coupling, _ = twiss {
    sequence=loaded_sequence,
    observe=1,
    X0=x0,
    trkopt=coupling_list,
    coupling=true,
}
--end-mad
""")

    mad.send(variables)
    mad.send(flat_opt_list)
    mad.send(accel_inst.RE_DICT[AccElementTypes.BPMS])
    mad.send(dk)
    mad.send(coupling_optics_list)

    optics_columns = [MADNG_OPTICS.get(optic, optic) for optic in optics_list]
    all_cols = ["name"] + optics_columns + flat_opt_list

    # Get twiss data with all columns
    LOG.debug(f"Extracting {all_cols} from MAD-NG twiss output")
    twiss_df = mad.tws.to_df(columns=all_cols)
    twiss_df.set_index("name", inplace=True)
    LOG.debug(f"Twiss extracted with shape {twiss_df.shape}")

    LOG.debug(f"Extracting {coupling_optics_list} from MAD-NG twiss output")
    twiss_df_coupling = mad.tws_coupling.to_df(columns=["name"] + coupling_optics_list)
    twiss_df_coupling.set_index("name", inplace=True)
    LOG.debug(f"Coupling twiss extracted with shape {twiss_df_coupling.shape}")

    del mad  # close the MAD-NG instance now we're done with it

    # Extract response matrices
    response_dict = {}

    # Get element names
    bpms = twiss_df.index

    # Tunes (at the END element)
    end = bpms[-1]  # Get the 'END' element for tunes
    tune_data = np.array(
        [
            twiss_df[kopt_dict["mu1"]].loc[end].to_numpy(),
            twiss_df[kopt_dict["mu2"]].loc[end].to_numpy(),
        ]
    )
    response_dict[TUNE] = pd.DataFrame(
        data=tune_data, index=[f"{TUNE}1", f"{TUNE}2"], columns=variables
    )

    # Remove END from bpms for other optics
    bpms = bpms[:-1]
    twiss_df = twiss_df.loc[bpms]
    twiss_df_coupling = twiss_df_coupling.loc[bpms]

    # Build response dictionary for other optics
    for optic_param, response_key in MADNG_VARMAP.items():
        deriv_cols = kopt_dict[optic_param]
        response_dict[response_key] = pd.DataFrame(
            data=twiss_df[deriv_cols].to_numpy(),  # shape (bpms, n_vars)
            index=bpms,
            columns=variables,
        )

    # Compute normalised dispersion response
    # d/dx (D/sqrt(beta)) = (2 beta * dD/dx - D * dbeta/dx) / (2 beta^(3/2))
    # Normalised Y dispersion can't even be used, see global correction OPTICS_PARAMS_CHOICES, so we skip it
    for plane in ["X"]:
        disp_col = f"d{plane.lower()}"
        beta_col = f"bet{plane.lower()}"

        # Convert to numpy arrays for computation and avoid NaNs (pandas may introduce NaNs)
        disp = twiss_df[disp_col].values
        beta = twiss_df[beta_col].values
        disp_resp = response_dict[f"{DISPERSION}{plane}"].values
        beta_resp = response_dict[f"{BETA}{plane}"].values

        numerator = 2 * beta[:, np.newaxis] * disp_resp - disp[:, np.newaxis] * beta_resp
        denominator = 2 * beta[:, np.newaxis] ** 1.5
        norm_disp_resp = numerator / denominator

        response_dict[f"{NORM_DISPERSION}{plane}"] = pd.DataFrame(
            data=norm_disp_resp, index=bpms, columns=variables
        )

    # Normalise beta response
    for plane in ["X", "Y"]:
        beta_col = f"bet{plane.lower()}"
        beta = twiss_df[beta_col].values
        response_dict[f"{BETA}{plane}"] = response_dict[f"{BETA}{plane}"] / beta[:, np.newaxis]

    for coupling_param, response_key in COUPLING_VARMAP.items():
        deriv_cols = coupling_kopt_dict[coupling_param]
        coupling_data = twiss_df_coupling[deriv_cols]
        response_dict[f"{response_key}R"] = pd.DataFrame(
            data=coupling_data.map(lambda x: x.real).to_numpy(),
            index=bpms,
            columns=variables,
        )
        response_dict[f"{response_key}I"] = pd.DataFrame(
            data=coupling_data.map(lambda x: x.imag).to_numpy(),
            index=bpms,
            columns=variables,
        )

    if accel_inst.beam_direction == -1:
        # Change the sign of the coupling responses for reverse beam direction
        # In all honesty, I don't know why this has to be done, but it works... (jgray 2025)
        for coupling_param in COUPLING_VARMAP.values():
            response_dict[f"{coupling_param}R"] *= -1
            response_dict[f"{coupling_param}I"] *= -1

    # Check for any NaNs in the response matrices
    for key, df in response_dict.items():
        if df.isna().any().any():
            LOG.warning(f"NaNs found in response matrix {key}: {df.isna().sum().sum()} NaNs")

    return response_dict


def _create_kopt_dict(params: list[str], n_vars: int) -> dict[str, list[str]]:
    """
    Create dictionary of derivative column names for given parameters.

    For each parameter (e.g., 'betx'), generates a list of column names for the derivatives
    with respect to each variable, using binary masks to identify which variable the derivative
    is for. This is used to extract the correct columns from the MAD-NG twiss output.

    Args:
        params: List of parameter names (e.g., ['mu1', 'betx', ...])
        n_vars: Number of variables (knobs) for which derivatives are computed

    Returns:
        Dictionary mapping parameter names to lists of derivative column names
    """
    return {
        param: [f"{param}_{_make_binary_mask(j, n_vars)}" for j in range(n_vars)]
        for param in params
    }


def _make_binary_mask(j: int, n_vars: int) -> str:
    """
    Create binary mask string for derivative w.r.t. variable j.

    Generates a string of '1's and '0's where the j-th position is '1' and others '0',
    used by MAD-NG to specify which variable's derivative is being computed.
    For example, with 3 variables and j=1, returns '010'.

    Args:
        j: Index of the variable for which the derivative is taken
        n_vars: Total number of variables

    Returns:
        Binary mask string
    """
    return "".join("1" if i == j else "0" for i in range(n_vars))
