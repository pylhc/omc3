"""
Response MAD-X
--------------

Provides a function to create the responses of beta, phase, dispersion, tune and coupling via iterative
madx calls.

The variables under investigation need to be provided as a list (which can be obtained from the accelerator
class).

For now, the response matrix is stored in a hdf5 file.

:author: Lukas Malina, Joschua Dilly, Jaime (...) Coello de Portugal
"""
from __future__ import annotations

import copy
import multiprocessing
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs
from numpy.exceptions import ComplexWarning
from optics_functions.coupling import coupling_via_cmatrix

from omc3.model.model_creators.manager import CreatorType, get_model_creator_class 
import omc3.madx_wrapper as madx_wrapper
from omc3.correction.constants import INCR, ORBIT_DPP
from omc3.model.accelerators.accelerator import AccElementTypes, Accelerator
from omc3.optics_measurements.constants import (
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
    from omc3.model.model_creators.abstract_model_creator import ModelCreator


# Full Response Mad-X ##########################################################


def create_fullresponse(
    accel_inst: Accelerator,
    variable_categories: Sequence[str],
    delta_k: float = 2e-5,
    num_proc: int = multiprocessing.cpu_count(),
    temp_dir: Path | None = None
) -> dict[str, pd.DataFrame]:
    """ Generate a dictionary containing response matrices for
        beta, phase, dispersion, tune and coupling and saves it to a file.

        Args:
            accel_inst : Accelerator Instance.
            variable_categories (list): Categories of the variables/knobs to use. (from .json)
            delta_k (float): delta K1L to be applied to quads for sensitivity matrix
            num_proc (int): Number of processes to use in parallel.
            temp_dir (str): temporary directory. If ``None``, uses folder of original_jobfile.
    """
    LOG.debug("Generating Fullresponse via Mad-X.")
    with timeit(lambda t: LOG.debug(f"  Total time generating fullresponse: {t} s")):
        if not temp_dir:
            temp_dir = Path(accel_inst.model_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        variables = accel_inst.get_variables(classes=variable_categories)
        if len(variables) == 0:
            raise ValueError("No variables found! Make sure your categories are valid!")
        num_proc = num_proc if len(variables) > num_proc else len(variables)
        process_pool = multiprocessing.Pool(processes=num_proc)

        incr_dict = _generate_madx_jobs(accel_inst, variables, delta_k, num_proc, temp_dir)
        _call_madx(process_pool, temp_dir, num_proc)
        _clean_up(temp_dir, num_proc)

        var_to_twiss = _load_madx_results(variables, process_pool, incr_dict, temp_dir)
        fullresponse = _create_fullresponse_from_dict(var_to_twiss)

    return fullresponse


def _generate_madx_jobs(
    accel_inst: Accelerator,
    variables: Sequence[str],
    delta_k: float,
    num_proc: int,
    temp_dir: Path
) -> dict[str, float]:
    """ Generates madx job-files """
    LOG.debug("Generating MAD-X jobfiles.")
    incr_dict = {'0': 0.0}
    compute_deltap: bool = ORBIT_DPP in variables
    no_dpp_vars = [var for var in variables if var != ORBIT_DPP]
    vars_per_proc = int(np.ceil(len(no_dpp_vars) / num_proc))
    creator = _get_nominal_model_creator(accel_inst)

    madx_job = _get_madx_job(accel_inst)
    deltap_twiss = ""
    if compute_deltap:
        # This is here only for multiple iteration of the global correction
        # By including dpp here, it means that if deltap is in variables and dpp is not 0, the orbit and tune magnets change
        # We have to be very careful that DELTAP_NAME is not used ANYWHERE else in MAD-X
        madx_job += f"{ORBIT_DPP} = {accel_inst.dpp};\n" # Set deltap to 0
        
        # get update deltap setup from model creator
        madx_job += creator.get_update_deltap_script(deltap=ORBIT_DPP)
        deltap_twiss = f", deltap={ORBIT_DPP}"

    for proc_idx in range(num_proc):
        jobfile_path = _get_jobfiles(temp_dir, proc_idx)

        current_job = madx_job
        for i in range(vars_per_proc):
            var_idx = proc_idx * vars_per_proc + i
            if var_idx >= len(no_dpp_vars):
                break
            var = no_dpp_vars[var_idx]
            incr_dict[var] = delta_k
            current_job += f"{var} = {var}{delta_k:+.15e};\n"
            current_job += f"twiss, file='{str(temp_dir / f'twiss.{var}')}'{deltap_twiss};\n"
            current_job += f"{var} = {var}{-delta_k:+.15e};\n\n"

        if proc_idx == num_proc - 1:
            current_job += f"twiss, file='{str(temp_dir / 'twiss.0')}'{deltap_twiss};\n"
            
            if compute_deltap: # If ORBIT_DPP is in variables, we run this in the last iteration
                # Due to the match and correction of the orbit, this needs to be run at the end of the process
                incr_dict[ORBIT_DPP] = delta_k
                current_job += f"{ORBIT_DPP} = {ORBIT_DPP}{delta_k:+.15e};\n"
                current_job += creator.get_update_deltap_script(deltap=ORBIT_DPP) # Do twiss, correct, match
                current_job += f"twiss, deltap={ORBIT_DPP}, file='{str(temp_dir/f'twiss.{ORBIT_DPP}')}';\n"
        jobfile_path.write_text(current_job)
    return incr_dict


def _get_madx_job(accel_inst: Accelerator) -> str:
    # use relative paths as we use model_dir as cwd
    model_dir_backup = accel_inst.model_dir  
    accel_inst.model_dir = Path()

    # get nominal setup from creator
    creator = _get_nominal_model_creator(accel_inst)
    job_content = creator.get_base_madx_script()
    job_content += (
        "select, flag=twiss, clear;\n"
        f"select, flag=twiss, pattern='{accel_inst.RE_DICT[AccElementTypes.BPMS]}', "
        "column=NAME,S,BETX,ALFX,BETY,ALFY,DX,DY,DPX,DPY,X,Y,K1L,MUX,MUY,R11,R12,R21,R22;\n\n")

    # restore model_dir
    accel_inst.model_dir = model_dir_backup
    return job_content


def _get_nominal_model_creator(accel_inst: Accelerator) -> ModelCreator:
    """ Get the nominal model creator, to which we can add the change of parameters.
    
    This is always done on the nominal model, not the best knowledge model, to ensure
    that the response matrix is in the most linear regime and therefore most accurate 
    (for most scenarios).
    """
    creator_class = get_model_creator_class(accel_inst, CreatorType.NOMINAL)
    return creator_class(accel_inst)


def _call_madx(process_pool: multiprocessing.Pool, temp_dir: str, num_proc: int) -> None: # type: ignore
    """ Call madx in parallel """
    LOG.debug(f"Starting {num_proc:d} MAD-X jobs...")
    madx_jobs = [_get_jobfiles(temp_dir, index) for index in range(num_proc)]
    process_pool.map(_launch_single_job, madx_jobs)
    LOG.debug("MAD-X jobs done.")


def _clean_up(temp_dir: Path, num_proc: int) -> None:
    """ Merge Logfiles and clean temporary outputfiles """
    LOG.debug("Cleaning output and building log...")
    full_log = ""
    for index in range(num_proc):
        job_path = _get_jobfiles(temp_dir, index)
        log_path = job_path.with_name(f"{job_path.name}.log")
        full_log += log_path.read_text()
        log_path.unlink()
        if index:  # keep 0th for reference
            job_path.unlink()
    
    # write compressed full log file
    full_log_name = "response_madx_full.log"
    zipfile.ZipFile(
        temp_dir / f"{full_log_name}.zip", 
        mode="w", 
        compression=zipfile.ZIP_DEFLATED
    ).writestr(full_log_name, full_log)


def _load_madx_results(
    variables: list[str],
    process_pool,
    incr_dict: dict,
    temp_dir: Path
) -> dict[str, tfs.TfsDataFrame]:
    """ Load the madx results in parallel and return var-tfs dictionary """
    LOG.debug("Loading Madx Results.")
    vars_and_paths = []
    for value in variables + ['0']:
        vars_and_paths.append((value, temp_dir))
    var_to_twiss = {}
    for var, tfs_data in process_pool.map(_load_and_remove_twiss, vars_and_paths):
        tfs_data[INCR] = incr_dict[var]
        var_to_twiss[var] = tfs_data
    return var_to_twiss


def _create_fullresponse_from_dict(var_to_twiss: dict[str, tfs.TfsDataFrame]) -> dict[str, pd.DataFrame]:
    """ Convert var-tfs dictionary to fullresponse dictionary. """
    var_to_twiss = _add_coupling(var_to_twiss)
    keys = list(var_to_twiss.keys())

    columns = [f"{PHASE_ADV}X", f"{PHASE_ADV}Y", f"{BETA}X", f"{BETA}Y", f"{DISPERSION}X", f"{DISPERSION}Y",
               f"{F1001}R", f"{F1001}I", f"{F1010}R", f"{F1010}I", f"{TUNE}1", f"{TUNE}2", INCR]

    bpms = var_to_twiss["0"].index
    resp = np.empty((len(keys), bpms.size, len(columns)))
    
    for i, key in enumerate(keys):
        resp[i] = var_to_twiss[key].loc[:, columns].to_numpy()

    resp = resp.transpose(2, 1, 0)
    model_index = list(keys).index("0")

    # create normalized dispersion and dividing BET by nominal model
    NDX_arr = np.divide(resp[columns.index(f"{DISPERSION}X")], np.sqrt(resp[columns.index(f"{BETA}X")]))
    NDY_arr = np.divide(resp[columns.index(f"{DISPERSION}Y")], np.sqrt(resp[columns.index(f"{BETA}Y")]))
    resp[columns.index(f"{BETA}X")] = np.divide(
        resp[columns.index(f"{BETA}X")], resp[columns.index(f"{BETA}X"), :, model_index][:, np.newaxis]
    )
    resp[columns.index(f"{BETA}Y")] = np.divide(
        resp[columns.index(f"{BETA}Y")], resp[columns.index(f"{BETA}Y"), :, model_index][:, np.newaxis]
    )

    # subtracting nominal model from data
    resp = np.subtract(resp, resp[:, :, model_index][:, :, np.newaxis])
    NDX_arr = np.subtract(NDX_arr, NDX_arr[:, model_index][:, np.newaxis])
    NDY_arr = np.subtract(NDY_arr, NDY_arr[:, model_index][:, np.newaxis])
    
    # Remove difference of nominal model with itself (bunch of zeros) and divide by increment
    resp = np.delete(resp, model_index, axis=2)
    NDX_arr = np.delete(NDX_arr, model_index, axis=1)
    NDY_arr = np.delete(NDY_arr, model_index, axis=1)
    keys.remove("0")

    NDX_arr = np.divide(NDX_arr, resp[columns.index(f"{INCR}")])
    NDY_arr = np.divide(NDY_arr, resp[columns.index(f"{INCR}")])
    resp = np.divide(resp,resp[columns.index(f"{INCR}")])
    Q_arr = np.column_stack((resp[columns.index(f"{TUNE}1"), 0, :], resp[columns.index(f"{TUNE}2"), 0, :])).T
 
    with suppress_warnings(ComplexWarning):  # raised as everything is complex-type now
        return {
            f"{PHASE_ADV}X": pd.DataFrame(data=resp[columns.index(f"{PHASE_ADV}X")], index=bpms, columns=keys).astype(np.float64),
            f"{PHASE_ADV}Y": pd.DataFrame(data=resp[columns.index(f"{PHASE_ADV}Y")], index=bpms, columns=keys).astype(np.float64),
            f"{BETA}X": pd.DataFrame(data=resp[columns.index(f"{BETA}X")], index=bpms, columns=keys).astype(np.float64),
            f"{BETA}Y": pd.DataFrame(data=resp[columns.index(f"{BETA}Y")], index=bpms, columns=keys).astype(np.float64),
            f"{DISPERSION}X": pd.DataFrame(data=resp[columns.index(f"{DISPERSION}X")], index=bpms, columns=keys).astype(np.float64),
            f"{DISPERSION}Y": pd.DataFrame(data=resp[columns.index(f"{DISPERSION}Y")], index=bpms, columns=keys).astype(np.float64),
            f"{NORM_DISPERSION}X": pd.DataFrame(data=NDX_arr, index=bpms, columns=keys).astype(np.float64),
            f"{NORM_DISPERSION}Y": pd.DataFrame(data=NDY_arr, index=bpms, columns=keys).astype(np.float64),
            f"{F1001}R": pd.DataFrame(data=resp[columns.index(f"{F1001}R")], index=bpms, columns=keys).astype(np.float64),
            f"{F1001}I": pd.DataFrame(data=resp[columns.index(f"{F1001}I")], index=bpms, columns=keys).astype(np.float64),
            f"{F1010}R": pd.DataFrame(data=resp[columns.index(f"{F1010}R")], index=bpms, columns=keys).astype(np.float64),
            f"{F1010}I": pd.DataFrame(data=resp[columns.index(f"{F1010}I")], index=bpms, columns=keys).astype(np.float64),
            f"{TUNE}": pd.DataFrame(data=Q_arr, index=[f"{TUNE}1", f"{TUNE}2"], columns=keys).astype(np.float64),
        }


def _get_jobfiles(temp_dir: Path, index: int) -> Path:
    """ Return names for jobfile and iterfile according to index """
    return temp_dir / f"job.iterate.{index:d}.madx"


def _launch_single_job(inputfile_path: Path) -> None:
    """ Function for pool to start a single madx job """
    log_file = inputfile_path.with_name(inputfile_path.name + ".log")
    madx_wrapper.run_file(inputfile_path, log_file=log_file, cwd=inputfile_path.parent)


def _load_and_remove_twiss(var_and_path: tuple[str, Path]) -> tuple[str, tfs.TfsDataFrame]:
    """ Function for pool to retrieve results """
    (var, path) = var_and_path
    twissfile = path / f"twiss.{var}"
    tfs_data = tfs.read(twissfile, index=NAME)
    tfs_data[f"{TUNE}1"] = tfs_data.Q1
    tfs_data[f"{TUNE}2"] = tfs_data.Q2
    twissfile.unlink()
    return var, tfs_data


def _add_coupling(dict_of_tfs: dict[str, tfs.TfsDataFrame]) -> dict[str, tfs.TfsDataFrame]:
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
        for var, tfs_dframe in result_dict_of_tfs.items():  # already copies, so it's safe to act on them
            coupling_rdts_df = coupling_via_cmatrix(tfs_dframe)
            tfs_dframe[f"{F1001}R"] = np.real(coupling_rdts_df[f"{F1001}"]).astype(np.float64)
            tfs_dframe[f"{F1001}I"] = np.imag(coupling_rdts_df[f"{F1001}"]).astype(np.float64)
            tfs_dframe[f"{F1010}R"] = np.real(coupling_rdts_df[f"{F1010}"]).astype(np.float64)
            tfs_dframe[f"{F1010}I"] = np.imag(coupling_rdts_df[f"{F1010}"]).astype(np.float64)
        return result_dict_of_tfs
