"""
Response MAD-NG
---------------

Provides a function to create the responses of beta, phase, dispersion, tune and coupling via MAD-NG derivatives.

The variables under investigation need to be provided as a list (which can be obtained from the accelerator
class).

For now, the response matrix is stored in a hdf5 file.

:author: [Your Name]
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
    # NAME,
    # NORM_DISPERSION,
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
    "beta11": f"{BETA}X",
    "beta22": f"{BETA}Y",
    "disp1": f"{DISPERSION}X",
    "disp3": f"{DISPERSION}Y",  # check if d2 = dpx?
    # "q1": f"{TUNE}1",
    # "q2": f"{TUNE}2",
}

MADNG_OPTICS = {
    "disp1": "dx",
    "disp3": "dy",
}


class LoggingStream:
    """A stream that logs written data to the logger at MADX level."""

    def __init__(self, logger: logging.Logger, level: int = MADX):
        self.logger = logger
        self.level = level
        self.buffer = []

    def write(self, text: str) -> None:
        if text.strip():  # Only log non-empty lines
            self.logger.log(self.level, text.rstrip())

    def flush(self) -> None:
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
    _load_sequence(mad, str(madx_seq_path.absolute()), seq_name)
    _setup_beam(mad, accel_inst.energy)

    return mad, log_path


def _log_madng_output(log_path: Path) -> None:
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


def _load_sequence(mad: MAD, sequence_file: str, seq_name: str) -> None:
    """Load a sequence file into MAD-NG (copied from BaseMadInterface)."""
    mad.send(f'MADX:load("{sequence_file}")')
    if mad.MADX[seq_name] == 0:
        raise ValueError(f"Sequence '{seq_name}' not found in MAD file '{sequence_file}'")
    mad.send(f"loaded_sequence = MADX.{seq_name}")
    mad["SEQ_NAME"] = seq_name


def _setup_beam(mad: MAD, beam_energy: float, particle: str = "proton") -> None:
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

    Args:
        mad: MAD-NG instance
        variables: List of variable names
        accel_inst: Accelerator instance

    Returns:
        Dictionary of response DataFrames
    """
    n_vars = len(variables)

    # Define optics parameters for which we want derivatives
    optics_list = list(MADNG_VARMAP.keys())

    kopt_dict = {
        optic_param: [f"{optic_param}_{_make_binary_mask(j, n_vars)}" for j in range(n_vars)]
        for optic_param in optics_list
    }
    flat_opt_list = [item for sublist in kopt_dict.values() for item in sublist]

    # Set up Differential Algebra for derivatives
    mad.send("""
--start-mad
local knob_list = py:recv()
local opt_list = py:recv()
local bpm_pattern = py:recv()
local num_k = #knob_list
local k_ord = 2
local x0 = MAD.damap { nv = 6, np = num_k, no = {k_ord, k_ord, k_ord, k_ord, 1, 1}, po=1, pn=knob_list}
local dk = {}
for i, knob in ipairs(knob_list) do
    ! dk[knob] = MADX[knob] == 0 and 1e-6 or 0.0
    MADX[knob] = MADX[knob] + x0[knob] !+ dk[knob]
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
!    trkrdt=rdt_list,
    cmap=false,
    coupling=true,
!    debug=7,
}

-- Reset the knobs (not necessary if we end MAD session here)
for i, knob in ipairs(knob_list) do
    MADX[knob] = MADX[knob]:get0()! - dk[knob]
end
--end-mad
""")

    mad.send(variables)
    mad.send(flat_opt_list)
    mad.send(accel_inst.RE_DICT[AccElementTypes.BPMS])

    optics_columns = [MADNG_OPTICS.get(optic, optic) for optic in optics_list]
    all_cols = ["name"] + optics_columns + flat_opt_list

    # Get twiss data with all columns
    twiss_df = mad.tws.to_df(columns=all_cols)
    twiss_df.set_index("name", inplace=True)

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

    # Normalise beta response
    for beta_key in ["beta11", "beta22"]:
        deriv_cols = kopt_dict[beta_key]
        twiss_df[deriv_cols] = twiss_df[deriv_cols] / twiss_df[beta_key].values[:, np.newaxis]

    # Build response dictionary for other optics
    for optic_param, response_key in MADNG_VARMAP.items():
        deriv_cols = kopt_dict[optic_param]
        response_dict[response_key] = pd.DataFrame(
            data=twiss_df[deriv_cols].loc[bpms].to_numpy(),  # shape (bpms, n_vars)
            index=bpms,
            columns=variables,
        )

    return response_dict


def _make_binary_mask(j: int, n_vars: int) -> str:
    """Create binary mask string for derivative w.r.t. variable j."""
    return "".join("1" if i == j else "0" for i in range(n_vars))
