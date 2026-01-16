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

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pymadng import MAD

from omc3.madx_wrapper import run_string
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

LOGGER = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from omc3.model.accelerators.accelerator import Accelerator

MADNG_VARMAP: dict[str, str] = {
    "mu1": f"{PHASE_ADV}X",
    "mu2": f"{PHASE_ADV}Y",
    "betx": f"{BETA}X",
    "bety": f"{BETA}Y",
    "disp1": f"{DISPERSION}X",
    "disp3": f"{DISPERSION}Y",  # disp3 corresponds to vertical dispersion (dy) in MAD-NG
}

COUPLING_VARMAP: dict[str, str] = {
    "f1001": F1001,
    "f1010": F1010,
}

MADNG_OPTICS: dict[str, str] = {
    "disp1": "dx",
    "disp3": "dy",
}

# From testing, this value gives enough stability for the coupling derivatives
# We have to have a delta k as the calculation of the coupling rdts are unstable
# when we don't have coupling.
# But since all knobs are changed at the same time, it is ideal to perturb them
# by as small a value as possible to avoid large changes in the optics.
NG_DELTA_K = 1e-6

@contextmanager
def _initialise_madng_with_logging(accel_inst: Accelerator):
    """Initialise MAD-NG, sending the stdout of the MAD-NG instance to a log file.

    This function does the very basics;
    - Creates a MAD-NG instance with logging to a file
    - Loads the sequence
    - Adds the beam to the sequence
    This is a minimal setup that allows the MAD-NG instance to be used for this sequence.

    Args:
        accel_inst: Accelerator instance

    Returns:
        Tuple of (MAD instance, log file path)
    """
    # Create log file in the model directory
    log_path = accel_inst.model_dir / "madng_response.log"
    LOGGER.info(f"MAD-NG output will be logged to: {log_path}")

    # Initialize MAD-NG with stdout redirected to log file
    with MAD(stdout=str(log_path), redirect_stderr=True, debug=True) as mad:
        _load_sequence(mad, accel_inst)
        _setup_beam(mad, accel_inst.energy)

        yield mad, log_path


def _log_madng_output(log_path: Path):
    """Once MAD-NG has written to a log file, read and log its contents.

    Args:
        log_path: Path to the log file
    """
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if line := line.strip():
                LOGGER.log(MADX, line)
    else:
        LOGGER.warning(f"MAD-NG log file {log_path} does not exist!")


def _load_sequence(mad: MAD, accel_inst: Accelerator):
    """Load a sequence into MAD-NG, handling all path creation, file generation, and cleanup internally."""
    # Create the model creator and prepare
    creator_class = get_model_creator_class(accel_inst, CreatorType.NOMINAL)
    creator = creator_class(accel_inst)
    creator.prepare_run()

    # Construct paths
    madx_seq_path = accel_inst.model_dir / creator.save_sequence_filename
    seq_name = creator.sequence_name
    log_file = accel_inst.model_dir / "madng_sequence_creation.log"

    if madx_seq_path.exists():
        raise FileExistsError(f"Saved sequence file {madx_seq_path} already exists!")

    # Generate the MAD-X sequence file
    madx_string = creator.get_base_madx_script() + "\n" + creator.get_save_sequence_script()
    run_string(madx_string, log_file=log_file)

    if not madx_seq_path.exists():
        raise FileNotFoundError(
            f"Saved sequence file {madx_seq_path} was not created! Check {log_file}"
        )

    # Load into MAD-NG
    mad.send(f'MADX:load("{str(madx_seq_path.absolute())}")')
    if mad.MADX[seq_name] == 0:
        raise ValueError(
            f"Sequence '{seq_name}' not found in MAD file '{str(madx_seq_path.absolute())}'"
        )
    mad.send(f"loaded_sequence = MADX.{seq_name}")
    mad["SEQ_NAME"] = seq_name
    mad.loaded_sequence.dir = accel_inst.beam_direction

    # Clean up the temporary file
    madx_seq_path.unlink(missing_ok=True)


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
    LOGGER.info("Creating fullresponse via MAD-NG")

    # Initialise MAD-NG with logging
    with _initialise_madng_with_logging(accel_inst) as (mad, log_path):
        # Get variables
        variables = accel_inst.get_variables(classes=variable_categories)
        if len(variables) == 0:
            raise ValueError("No variables found! Make sure your categories are valid!")

        LOGGER.info(f"Computing response for {len(variables)} variables")

        try:
            # Compute derivatives with twiss
            response_dict = _compute_response_with_derivatives(mad, variables, accel_inst)
        except RuntimeError as e:
            LOGGER.error("Error while computing response with MAD-NG derivatives.")
            _log_madng_output(log_path)
            raise e
        _log_madng_output(log_path)

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

    # Create mapping from optics parameters to derivative column names, e.g., 'betx' -> ['betx_100', 'betx_010', ...]
    kopt_dict = _create_kopt_dict(optics_list, n_vars)
    flat_opt_list = [item for sublist in kopt_dict.values() for item in sublist]

    # Create mapping for coupling parameters e.g., 'f1001' -> ['f1001_100', 'f1001_010', ...]
    coupling_kopt_dict = _create_kopt_dict(coupling_params, n_vars)
    coupling_optics_list = [item for sublist in coupling_kopt_dict.values() for item in sublist]

    dk = dict.fromkeys(variables, NG_DELTA_K)

    # Set up Differential Algebra for derivatives
    # This MAD-NG script performs the following workflow:
    # 1. Receive lists of knobs, optics parameters, and BPM pattern from Python.
    # 2. Create a differential algebra map (damap) for containing the variables and initial conditions.
    # 3. Convert knobs to DA variables by adding their damap components.
    # 4. Select BPMs and END element for observation.
    # 5. Run twiss calculation with derivatives enabled for standard optics.
    # 6. Receive delta k values and coupling optics list.
    # 7. Perturb knobs with delta k for stable coupling derivative computation.
    # 8. Run second twiss calculation for coupling derivatives.
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
loaded_sequence:select(observed, {pattern="$end"}) ! Also observe END marker for tunes

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

    # Send the required data to MAD-NG script
    mad.send(variables)  # List of knob names
    mad.send(flat_opt_list)  # Flattened list of derivative column names for standard optics
    mad.send(accel_inst.RE_DICT[AccElementTypes.BPMS])  # BPM pattern for observation
    mad.send(dk)  # Dictionary of delta k values for perturbation
    mad.send(coupling_optics_list)  # Flattened list of derivative column names for coupling

    # Extract and process twiss data from MAD-NG
    optics_columns = [MADNG_OPTICS.get(optic, optic) for optic in optics_list]
    all_cols = ["name"] + optics_columns + flat_opt_list

    # Get twiss data with all columns
    LOGGER.debug(f"Extracting {all_cols} from MAD-NG twiss output")
    twiss_df = mad.tws.to_df(columns=all_cols)
    twiss_df.set_index("name", inplace=True)
    LOGGER.debug(f"Twiss extracted with shape {twiss_df.shape}")

    LOGGER.debug(f"Extracting {coupling_optics_list} from MAD-NG twiss output")
    twiss_df_coupling = mad.tws_coupling.to_df(columns=["name"] + coupling_optics_list)
    twiss_df_coupling.set_index("name", inplace=True)
    LOGGER.debug(f"Coupling twiss extracted with shape {twiss_df_coupling.shape}")

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
        disp_col = f"{DISPERSION}{plane}"
        beta_col = f"{BETA}{plane}"
        # Convert to numpy arrays for computation and avoid NaNs (pandas may introduce NaNs)
        disp = twiss_df[disp_col.lower()].to_numpy()
        beta = twiss_df[beta_col.lower()].to_numpy()
        disp_resp = response_dict[disp_col].to_numpy()
        beta_resp = response_dict[beta_col].to_numpy()

        numerator = 2 * beta[:, np.newaxis] * disp_resp - disp[:, np.newaxis] * beta_resp
        denominator = 2 * beta[:, np.newaxis] ** 1.5
        norm_disp_resp = numerator / denominator

        response_dict[f"{NORM_DISPERSION}{plane}"] = pd.DataFrame(
            data=norm_disp_resp, index=bpms, columns=variables
        )

    # Normalise beta response
    for plane in ["X", "Y"]:
        beta_col = f"{BETA}{plane}"
        beta = twiss_df[beta_col.lower()].to_numpy()
        response_dict[beta_col] = response_dict[beta_col] / beta[:, np.newaxis]

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
        # In all honesty, I don't know why this has to be done, but it works... (jgray 2026)
        for coupling_param in COUPLING_VARMAP.values():
            response_dict[f"{coupling_param}R"] *= -1
            response_dict[f"{coupling_param}I"] *= -1

    # Check for any NaNs in the response matrices
    for key, df in response_dict.items():
        if df.isna().any().any():
            LOGGER.warning(f"NaNs found in response matrix for {key}: {df.isna().sum().sum()} NaNs")

    return response_dict


def _create_kopt_dict(params: list[str], n_vars: int) -> dict[str, list[str]]:
    """
    Create a dictionary mapping each optics parameter to its list of derivative column names.

    For each parameter (e.g., 'betx'), this generates a list of column names corresponding to
    the derivatives of that parameter with respect to each variable (knob). Each column name
    uses a binary mask string to indicate which variable the derivative is taken with respect to.

    The binary mask is a string of length n_vars, consisting of '0's and '1's, where a '1' at
    position j indicates that the derivative is with respect to the j-th variable in the list
    provided to MAD-NG. All other positions are '0'.

    For example, with 3 variables ['var1', 'var2', 'var3'] and parameter 'betx', the output would be:
    {
        'betx': ['betx_100', 'betx_010', 'betx_001']
    }
    Here, 'betx_100' is the derivative of betx w.r.t. var1, 'betx_010' w.r.t. var2, etc.

    Args:
        params: List of parameter names (e.g., ['mu1', 'betx', ...])
        n_vars: Number of variables (knobs) for which derivatives are computed

    Returns:
        Dictionary mapping each parameter name to its list of derivative column names
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
