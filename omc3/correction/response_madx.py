"""
Provides a function to create the responses of beta, phase, dispersion, tune and coupling
via iterative madx calls.

The variables under investigation need to be provided as a list (which can be gotten from
accelerator class).

For now, the response matrix is stored in a 'pickled' file.

:author: Lukas Malina, Joschua Dilly, Jaime (...) Coello de Portugal
"""
import copy
import multiprocessing
import os
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import tfs
from optics_functions.coupling import coupling_via_cmatrix

import omc3.madx_wrapper as madx_wrapper
from omc3.correction.constants import (BETA, DISP, F1001, F1010, INCR,
                                       NORM_DISP, PHASE_ADV, TUNE)
from omc3.model.accelerators.accelerator import Accelerator
from omc3.utils import iotools, logging_tools
from omc3.utils.contexts import suppress_warnings, timeit

LOG = logging_tools.get_logger(__name__)


# Full Response Mad-X ##########################################################


def generate_fullresponse(
    accel_inst: Accelerator,
    variable_categories: Sequence[str],
    delta_k: float = 2e-5,
    num_proc: int = multiprocessing.cpu_count(),
    temp_dir: str = None
) -> Dict[str, pd.DataFrame]:
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
            temp_dir = accel_inst.model_dir
        iotools.create_dirs(temp_dir)

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
    temp_dir: str
) -> Dict[str, float]:
    """ Generates madx job-files """
    LOG.debug("Generating MADX jobfiles.")
    incr_dict = {'0': 0.0}
    vars_per_proc = int(np.ceil(len(variables) / num_proc))

    madx_job = _get_madx_job(accel_inst)

    for proc_idx in range(num_proc):
        jobfile_path = _get_jobfiles(temp_dir, proc_idx)

        current_job = madx_job
        for i in range(vars_per_proc):
            var_idx = proc_idx * vars_per_proc + i
            if var_idx >= len(variables):
                break
            var = variables[var_idx]
            incr_dict[var] = delta_k
            current_job += f"{var}={var}{delta_k:+f};\n"
            current_job += f"twiss, file='{os.path.join(temp_dir, f'twiss.{var}')}';\n"
            current_job += f"{var}={var}{-delta_k:+f};\n\n"

        if proc_idx == num_proc - 1:
            current_job += f"twiss, file='{os.path.join(temp_dir, 'twiss.0')}';\n"

        with open(jobfile_path, "w") as jobfile:
            jobfile.write(current_job)
    return incr_dict


def _get_madx_job(accel_inst: Accelerator) -> str:
    job_content = accel_inst.get_base_madx_script(accel_inst.model_dir)
    job_content += (
        "select, flag=twiss, clear;\n"
        f"select, flag=twiss, pattern='^BPM.*\.B{accel_inst.beam:d}$', "
        "column=NAME,S,BETX,ALFX,BETY,ALFY,DX,DY,DPX,DPY,X,Y,K1L,MUX,MUY,R11,R12,R21,R22;\n\n")
    return job_content


def _call_madx(process_pool: multiprocessing.Pool, temp_dir: str, num_proc: int) -> None:
    """ Call madx in parallel """
    LOG.debug(f"Starting {num_proc:d} MAD-X jobs...")
    madx_jobs = [_get_jobfiles(temp_dir, index) for index in range(num_proc)]
    process_pool.map(_launch_single_job, madx_jobs)
    LOG.debug("MAD-X jobs done.")


def _clean_up(temp_dir: str, num_proc: int) -> None:
    """ Merge Logfiles and clean temporary outputfiles """
    LOG.debug("Cleaning output and building log...")
    full_log = ""
    for index in range(num_proc):
        job_path = _get_jobfiles(temp_dir, index)
        log_path = job_path + ".log"
        with open(log_path, "r") as log_file:
            full_log += log_file.read()
        os.remove(log_path)
        os.remove(job_path)
    full_log_path = os.path.join(temp_dir, "response_madx_full.log")
    with open(full_log_path, "w") as full_log_file:
        full_log_file.write(full_log)


def _load_madx_results(
    variables: Sequence[str],
    process_pool: multiprocessing.Pool,
    incr_dict: dict,
    temp_dir: str
) -> Dict[str, tfs.TfsDataFrame]:
    """ Load the madx results in parallel and return var-tfs dictionary """
    LOG.debug("Loading Madx Results.")
    vars_and_paths = []
    for value in variables + ['0']:
        vars_and_paths.append((value, temp_dir))
    var_to_twiss = {}
    for var, tfs_data in process_pool.map(_load_and_remove_twiss, vars_and_paths):
        tfs_data[f"{INCR}"] = incr_dict[var]
        var_to_twiss[var] = tfs_data
    return var_to_twiss


def _create_fullresponse_from_dict(var_to_twiss: Dict[str, tfs.TfsDataFrame]) -> Dict[str, pd.DataFrame]:
    """ Convert var-tfs dictionary to fullresponse dictionary """
    var_to_twiss = _add_coupling(var_to_twiss)
    keys = list(var_to_twiss.keys())
    # columns = ["MUX","MUY","BETX","BETY","DX", "DY", "F1001R","F1001I","F1010R","F1010I","Q1","Q2","incr"]
    columns = [f"{PHASE_ADV}X", f"{PHASE_ADV}Y", f"{BETA}X", f"{BETA}Y", f"{DISP}X", f"{DISP}Y",
               f"{F1001}R", f"{F1001}I", f"{F1010}R", f"{F1010}I", f"{TUNE}1", f"{TUNE}2", f"{INCR}"]

    bpms = var_to_twiss["0"].index
    resp = np.empty((len(keys), bpms.size, len(columns)))
    
    for i, key in enumerate(keys):
        resp[i] = var_to_twiss[key].loc[:, columns].to_numpy()

    resp = resp.transpose(2, 1, 0)
    model_index = list(keys).index("0")

    # create normalized dispersion and dividing BET by nominal model
    NDX_arr = np.divide(resp[columns.index(f"{DISP}X")], np.sqrt(resp[columns.index(f"{BETA}X")]))
    NDY_arr = np.divide(resp[columns.index(f"{DISP}Y")], np.sqrt(resp[columns.index(f"{BETA}Y")]))
    resp[columns.index(f"{BETA}X")] = np.divide(resp[columns.index(f"{BETA}X")], resp[columns.index(f"{BETA}X"), : , model_index][:, np.newaxis])
    resp[columns.index(f"{BETA}Y")] = np.divide(resp[columns.index(f"{BETA}Y")], resp[columns.index(f"{BETA}Y"), : , model_index][:, np.newaxis])

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
 
    with suppress_warnings(np.ComplexWarning):  # raised as everything is complex-type now
        return {
            f"{PHASE_ADV}X": pd.DataFrame(data=resp[columns.index(f"{PHASE_ADV}X")], index=bpms, columns=keys).astype(np.float64),
            f"{PHASE_ADV}Y": pd.DataFrame(data=resp[columns.index(f"{PHASE_ADV}Y")], index=bpms, columns=keys).astype(np.float64),
            f"{BETA}X": pd.DataFrame(data=resp[columns.index(f"{BETA}X")], index=bpms, columns=keys).astype(np.float64),
            f"{BETA}Y": pd.DataFrame(data=resp[columns.index(f"{BETA}Y")], index=bpms, columns=keys).astype(np.float64),
            f"{DISP}X": pd.DataFrame(data=resp[columns.index(f"{DISP}X")], index=bpms, columns=keys).astype(np.float64),
            f"{DISP}Y": pd.DataFrame(data=resp[columns.index(f"{DISP}Y")], index=bpms, columns=keys).astype(np.float64),
            f"{NORM_DISP}X": pd.DataFrame(data=NDX_arr, index=bpms, columns=keys).astype(np.float64),
            f"{NORM_DISP}Y": pd.DataFrame(data=NDY_arr, index=bpms, columns=keys).astype(np.float64),
            f"{F1001}R": pd.DataFrame(data=resp[columns.index(f"{F1001}R")], index=bpms, columns=keys).astype(np.float64),
            f"{F1001}I": pd.DataFrame(data=resp[columns.index(f"{F1001}I")], index=bpms, columns=keys).astype(np.float64),
            f"{F1010}R": pd.DataFrame(data=resp[columns.index(f"{F1010}R")], index=bpms, columns=keys).astype(np.float64),
            f"{F1010}I": pd.DataFrame(data=resp[columns.index(f"{F1010}I")], index=bpms, columns=keys).astype(np.float64),
            f"{TUNE}": pd.DataFrame(data=Q_arr, index=[f"{TUNE}1", f"{TUNE}2"], columns=keys).astype(np.float64),
        }


def _get_jobfiles(temp_dir: str, index: int) -> str:
    """ Return names for jobfile and iterfile according to index """
    jobfile_path = os.path.join(temp_dir, f"job.iterate.{index:d}.madx")
    return jobfile_path


def _launch_single_job(inputfile_path: str) -> None:
    """ Function for pool to start a single madx job """
    log_file = inputfile_path + ".log"
    madx_wrapper.run_file(inputfile_path, log_file=log_file)


def _load_and_remove_twiss(var_and_path: Tuple[str, str]) -> Tuple[str, tfs.TfsDataFrame]:
    """ Function for pool to retrieve results """
    (var, path) = var_and_path
    twissfile = os.path.join(path, "twiss." + var)
    tfs_data = tfs.read(twissfile, index="NAME")
    tfs_data[f"{TUNE}1"] = tfs_data.Q1
    tfs_data[f"{TUNE}2"] = tfs_data.Q2
    os.remove(twissfile)
    return var, tfs_data


def _add_coupling(dict_of_tfs: Dict[str, tfs.TfsDataFrame]) -> Dict[str, tfs.TfsDataFrame]:
    """
    For each TfsDataFrame in the input dictionary, computes the coupling RDTs and adds a column for
    the real and imaginary parts of the computed coupling RDTs. Returns a copy of the input dictionary with
    the aforementioned computed columns added for each TfsDataFrame.

    Args:
        dict_of_tfs (Dict[str, tfs.TfsDataFrame]): dictionary of Twiss dataframes.

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