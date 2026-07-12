"""
Response Xsuite
---------------

Provides a function to create the responses of beta, phase, dispersion, tune and coupling by
computing the twiss of the sequence with `xsuite <https://xsuite.readthedocs.io>`_ (``xtrack``)
for each knob variation, in parallel.

The sequence itself is still built and matched by ``MAD-X`` (as for the model creation, see
:mod:`omc3.model.model_creators.lhc_xsuite_model_creator`), then loaded into ``xtrack``.

The variables under investigation need to be provided as a list (which can be obtained from the
accelerator class).

For now, the response matrix is stored in a hdf5 file.

:author: OMC Team
"""

from __future__ import annotations

import copy
import multiprocessing
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs
import xtrack as xt
from numpy.exceptions import ComplexWarning
from optics_functions.coupling import coupling_via_cmatrix

from omc3.correction.constants import INCR, ORBIT_DPP
from omc3.model.model_creators.manager import CreatorType, get_model_creator_class
from omc3.model.xsuite_bridge import _PROTON_MASS_EV, XSUITE_JSON, create_xsuite_json
from omc3.optics_measurements.constants import (
    ALPHA,
    BETA,
    DISPERSION,
    F1001,
    F1010,
    NAME,
    NORM_DISPERSION,
    PHASE_ADV,
    TUNE,
)
from omc3.utils import logging_tools
from omc3.utils.contexts import suppress_warnings, timeit

LOG = logging_tools.get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from omc3.model.accelerators.accelerator import Accelerator
    from omc3.model.model_creators.abstract_model_creator import ModelCreator


# Full Response xsuite ##########################################################

def create_fullresponse(
    accel_inst: Accelerator,
    variable_categories: Sequence[str],
    delta_k: float = 2e-5,
    sequence_file: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate a dictionary containing response matrices for
    beta, phase, dispersion, tune and coupling and saves it to a file.

    Args:
        accel_inst : Accelerator Instance.
        variable_categories (list): Categories of the variables/knobs to use. (from .json)
        delta_k (float): delta K1L to be applied to quads for sensitivity matrix

    Returns:
        dict: Dictionary of response DataFrames keyed by optics type (e.g., 'BETAX', 'PHASEX', etc.)
    """
    if getattr(accel_inst, "beam", 1) != 1:
        # Beam 2 needs the reversed line direction (beam_direction == -1) that is not yet
        # applied here, so the twiss (and hence the response) would be wrong. Refuse rather
        # than silently produce an incorrect response. Matches the xsuite model creator.
        raise NotImplementedError(
            "The xsuite response creator currently only supports LHC beam 1. "
            "Use the 'madx' or 'twiss' creator for beam 2. "
            "Beam-2 support is planned for a later migration milestone."
        )

    LOG.debug("Generating Fullresponse via xsuite.")
    with timeit(lambda t: LOG.debug(f"  Total time generating fullresponse: {t} s")):
        variables = accel_inst.get_variables(classes=variable_categories)
        if len(variables) == 0:
            raise ValueError("No variables found! Make sure your categories are valid!")
        var_to_twiss = _run_xsuite(accel_inst, variables, delta_k, sequence_file)
        return _create_fullresponse_from_dict(var_to_twiss)


COLUMNS_TO_KEEP = [
    "NAME",
    f"{PHASE_ADV}X",
    f"{PHASE_ADV}Y",
    f"{BETA}X",
    f"{BETA}Y",
    f"{ALPHA}X",
    f"{ALPHA}Y",
    f"{DISPERSION}X",
    f"{DISPERSION}Y",
    "R11_EDW_TENG",
    "R12_EDW_TENG",
    "R21_EDW_TENG",
    "R22_EDW_TENG",
]


def _convert_xsuite_tbl_to_tfs(tbl: xt.Table, increment: float) -> pd.DataFrame:
    """Convert an xsuite table to a madx-like pandas dataframe."""
    df = tfs.TfsDataFrame(tbl.rows["bpm.*"].to_pandas())
    # Convert all columns to uppercase to match madx convention
    df.columns = [col.upper() for col in df.columns]
    # Only keep the columns we need for the response matrix, to avoid confusion and save memory
    df = df.loc[:, COLUMNS_TO_KEEP]
    df = df.set_index(NAME)
    # Convert the index from lowercase to uppercase to match madx convention
    df.index = df.index.str.upper()
    df[f"{TUNE}1"] = tbl.qx
    df[f"{TUNE}2"] = tbl.qy
    df[INCR] = increment

    # Rename the edw_teng columns to match madx convention
    return df.rename(
        columns={
            "R11_EDW_TENG": "R11",
            "R12_EDW_TENG": "R12",
            "R21_EDW_TENG": "R21",
            "R22_EDW_TENG": "R22",
        }
    )


_WORKER_ENV = None
_WORKER_SEQ = None


def _init_xsuite_worker(seq_file: Path, sequence_name: str, p0c: float) -> None:
    """Initialise xsuite environment for a worker process."""
    global _WORKER_ENV, _WORKER_SEQ
    # Check if it's a madx sequence file or an xsuite json file
    if seq_file.suffix == ".json":
        _WORKER_ENV = xt.Environment.from_json(str(seq_file))
    else:
        _WORKER_ENV = xt.load(file=str(seq_file), format="madx")
    _WORKER_SEQ = _WORKER_ENV[sequence_name.lower()]
    _WORKER_SEQ.particle_ref = xt.Particles(p0c=p0c, mass0=_PROTON_MASS_EV)


def _compute_twiss_for_var(args: tuple[str, float]) -> tuple[str, pd.DataFrame]:
    """Compute twiss for a single variable and convert to TFS."""
    var, delta_k = args
    if var != "0":
        _WORKER_ENV[var] += delta_k
    try:
        tbl = _WORKER_SEQ.twiss4d(coupling_edw_teng=True)
    finally:
        if var != "0":
            _WORKER_ENV[var] -= delta_k
    increment = 0.0 if var == "0" else delta_k
    return var, _convert_xsuite_tbl_to_tfs(tbl, increment)


def _run_xsuite(
    accel_inst: Accelerator,
    variables: Sequence[str],
    delta_k: float,
    sequence_file: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Generates madx job-files"""
    LOG.debug("Generating Xsuite response.")
    var_to_twiss = {}

    no_dpp_vars = [var for var in variables if var != ORBIT_DPP]

    creator = _get_nominal_model_creator(accel_inst)
    if sequence_file is None:
        creator.prepare_run()
        # Build (once) the xtrack lattice json via the sanctioned MAD-X bridge; the workers
        # load it directly with xt.Environment.from_json (see _init_xsuite_worker).
        seq_file = create_xsuite_json(creator, accel_inst.model_dir / XSUITE_JSON)
    else:
        seq_file = sequence_file

    tasks = [("0", 0)] + [(var, delta_k) for var in no_dpp_vars]
    num_processes = min(multiprocessing.cpu_count(), len(tasks))
    LOG.warning(f"Starting parallel twiss+conversion using {num_processes} processes.")

    with multiprocessing.Pool(
        processes=num_processes,
        initializer=_init_xsuite_worker,
        initargs=(seq_file, creator.sequence_name, accel_inst.energy * 1e9),
    ) as pool:
        for var, df in pool.imap_unordered(_compute_twiss_for_var, tasks):
            var_to_twiss[var] = df

    LOG.warning("Parallel twiss+conversion completed.")

    return var_to_twiss


def _get_nominal_model_creator(accel_inst: Accelerator) -> ModelCreator:
    """Get the nominal model creator, to which we can add the change of parameters.

    This is always done on the nominal model, not the best knowledge model, to ensure
    that the response matrix is in the most linear regime and therefore most accurate
    (for most scenarios).
    """
    creator_class = get_model_creator_class(accel_inst, CreatorType.NOMINAL)
    return creator_class(accel_inst)


def _create_fullresponse_from_dict(
    var_to_twiss: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Convert var-tfs dictionary to fullresponse dictionary."""
    var_to_twiss = _add_coupling(var_to_twiss)
    keys = list(var_to_twiss.keys())

    columns = [
        f"{PHASE_ADV}X",
        f"{PHASE_ADV}Y",
        f"{BETA}X",
        f"{BETA}Y",
        f"{DISPERSION}X",
        f"{DISPERSION}Y",
        f"{F1001}R",
        f"{F1001}I",
        f"{F1010}R",
        f"{F1010}I",
        f"{TUNE}1",
        f"{TUNE}2",
        INCR,
    ]

    bpms = var_to_twiss["0"].index
    resp = np.empty((len(keys), bpms.size, len(columns)))

    for i, key in enumerate(keys):
        resp[i] = var_to_twiss[key].loc[:, columns].to_numpy()

    resp = resp.transpose(2, 1, 0)
    model_index = list(keys).index("0")

    # Create normalised dispersion and dividing BET by nominal model
    normalised_dispersion_x = np.divide(
        resp[columns.index(f"{DISPERSION}X")], np.sqrt(resp[columns.index(f"{BETA}X")])
    )
    normalised_dispersion_y = np.divide(
        resp[columns.index(f"{DISPERSION}Y")], np.sqrt(resp[columns.index(f"{BETA}Y")])
    )
    resp[columns.index(f"{BETA}X")] = np.divide(
        resp[columns.index(f"{BETA}X")],
        resp[columns.index(f"{BETA}X"), :, model_index][:, np.newaxis],
    )
    resp[columns.index(f"{BETA}Y")] = np.divide(
        resp[columns.index(f"{BETA}Y")],
        resp[columns.index(f"{BETA}Y"), :, model_index][:, np.newaxis],
    )

    # Subtracting nominal model from data
    resp = np.subtract(resp, resp[:, :, model_index][:, :, np.newaxis])
    normalised_dispersion_x = np.subtract(
        normalised_dispersion_x, normalised_dispersion_x[:, model_index][:, np.newaxis]
    )
    normalised_dispersion_y = np.subtract(
        normalised_dispersion_y, normalised_dispersion_y[:, model_index][:, np.newaxis]
    )

    # Remove difference of nominal model with itself (bunch of zeros) and divide by increment
    resp = np.delete(resp, model_index, axis=2)
    normalised_dispersion_x = np.delete(normalised_dispersion_x, model_index, axis=1)
    normalised_dispersion_y = np.delete(normalised_dispersion_y, model_index, axis=1)
    keys.remove("0")

    # Divide by increment
    normalised_dispersion_x = np.divide(normalised_dispersion_x, resp[columns.index(f"{INCR}")])
    normalised_dispersion_y = np.divide(normalised_dispersion_y, resp[columns.index(f"{INCR}")])
    resp = np.divide(resp, resp[columns.index(f"{INCR}")])
    tune_arr = np.column_stack(
        (
            resp[columns.index(f"{TUNE}1"), 0, :],
            resp[columns.index(f"{TUNE}2"), 0, :],
        )
    ).T

    # fmt: off
    with suppress_warnings(ComplexWarning):  # raised as everything is complex-type now
        return {
            f"{PHASE_ADV}X": pd.DataFrame(data=resp[columns.index(f"{PHASE_ADV}X")], index=bpms, columns=keys).astype(np.float64),
            f"{PHASE_ADV}Y": pd.DataFrame(data=resp[columns.index(f"{PHASE_ADV}Y")], index=bpms, columns=keys).astype(np.float64),
            f"{BETA}X": pd.DataFrame(data=resp[columns.index(f"{BETA}X")], index=bpms, columns=keys).astype(np.float64),
            f"{BETA}Y": pd.DataFrame(data=resp[columns.index(f"{BETA}Y")], index=bpms, columns=keys).astype(np.float64),
            f"{DISPERSION}X": pd.DataFrame(data=resp[columns.index(f"{DISPERSION}X")], index=bpms, columns=keys).astype(np.float64),
            f"{DISPERSION}Y": pd.DataFrame(data=resp[columns.index(f"{DISPERSION}Y")], index=bpms, columns=keys).astype(np.float64),
            f"{NORM_DISPERSION}X": pd.DataFrame(data=normalised_dispersion_x, index=bpms, columns=keys).astype(np.float64),
            f"{NORM_DISPERSION}Y": pd.DataFrame(data=normalised_dispersion_y, index=bpms, columns=keys).astype(np.float64),
            f"{F1001}R": pd.DataFrame(data=resp[columns.index(f"{F1001}R")], index=bpms, columns=keys).astype(np.float64),
            f"{F1001}I": pd.DataFrame(data=resp[columns.index(f"{F1001}I")], index=bpms, columns=keys).astype(np.float64),
            f"{F1010}R": pd.DataFrame(data=resp[columns.index(f"{F1010}R")], index=bpms, columns=keys).astype(np.float64),
            f"{F1010}I": pd.DataFrame(data=resp[columns.index(f"{F1010}I")], index=bpms, columns=keys).astype(np.float64),
            f"{TUNE}": pd.DataFrame(data=tune_arr, index=[f"{TUNE}1", f"{TUNE}2"], columns=keys).astype(np.float64),
        }
    # fmt: on


def _add_coupling(
    dict_of_tfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    For each TfsDataFrame in the input dictionary, computes the coupling RDTs and adds a column for
    the real and imaginary parts of the computed coupling RDTs. Returns a copy of the input dictionary with
    the aforementioned computed columns added for each TfsDataFrame.

    Args:
        dict_of_tfs (dict[str, tfs.TfsDataFrame]): dictionary of Twiss dataframes.

    Returns:
        An identical dictionary of Twiss dataframes, with the computed columns added.
    """
    result_dict_of_tfs = copy.deepcopy(dict_of_tfs)
    with timeit(lambda elapsed: LOG.debug(f"  Time adding coupling: {elapsed} s")):
        for tfs_dframe in result_dict_of_tfs.values():
            coupling_rdts_df = coupling_via_cmatrix(tfs_dframe)
            tfs_dframe[f"{F1001}R"] = np.real(coupling_rdts_df[f"{F1001}"]).astype(np.float64)
            tfs_dframe[f"{F1001}I"] = np.imag(coupling_rdts_df[f"{F1001}"]).astype(np.float64)
            tfs_dframe[f"{F1010}R"] = np.real(coupling_rdts_df[f"{F1010}"]).astype(np.float64)
            tfs_dframe[f"{F1010}I"] = np.imag(coupling_rdts_df[f"{F1010}"]).astype(np.float64)
        return result_dict_of_tfs
