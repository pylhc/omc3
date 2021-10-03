"""
Sequence Evaluation
-------------------

Evaluates the variable responses from a sequence in MAD-X.

First: Set all variables to 0
Then: Set one variable at a time to 1

Compare results with case all==0.
"""
import multiprocessing
from contextlib import suppress
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import tfs

import omc3.madx_wrapper as madx_wrapper
from omc3.correction.response_io import write_fullresponse, write_varmap
from omc3.model.accelerators.accelerator import Accelerator, AccElementTypes
from omc3.utils import logging_tools
from omc3.utils.contexts import timeit

LOG = logging_tools.get_logger(__name__)

EXT = "h5"  # Extension Standard


# Read Sequence ##############################################################


def evaluate_for_variables(
    accel_inst: Accelerator,
    variable_categories,
    order: int = 4,
    num_proc: int = multiprocessing.cpu_count(),
    temp_dir: Path = None
) -> dict:
    """ Generate a dictionary containing response matrices for
        beta, phase, dispersion, tune and coupling and saves it to a file.

        Args:
            accel_inst (Accelerator): Accelerator Instance.
            variable_categories (list): Categories of the variables/knobs to use. (from .json)
            order (int or tuple): Max or [min, max] of K-value order to use.
            num_proc (int): Number of processes to use in parallel.
            temp_dir (Path): temporary directory. If ``None``, uses model_dir.
    """
    LOG.debug("Generating Fullresponse via Mad-X.")
    with timeit(lambda elapsed: LOG.debug(f"  Total time generating fullresponse: {elapsed}s")):
        if not temp_dir:
            temp_dir = Path(accel_inst.model_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        variables = accel_inst.get_variables(classes=variable_categories)
        if len(variables) == 0:
            raise ValueError("No variables found! Make sure your categories are valid!")

        num_proc = num_proc if len(variables) > num_proc else len(variables)
        process_pool = multiprocessing.Pool(processes=num_proc)

        k_values = _get_orders(order)

        try:
            _generate_madx_jobs(accel_inst, variables, k_values, num_proc, temp_dir)
            _call_madx(process_pool, temp_dir, num_proc)
            mapping = _load_madx_results(variables, k_values, process_pool, temp_dir)
        finally:
            _clean_up(variables, temp_dir, num_proc)
    return mapping


def _generate_madx_jobs(
    accel_inst: Accelerator,
    variables: Sequence[str],
    k_values: List[float],
    num_proc: int,
    temp_dir: Path,
) -> None:
    """ Generates madx job-files """
    def _assign(var, value):
        return f"{var:s} = {value:d};\n"

    def _do_macro(var):
        return (
            f"exec, create_table(table.{var:s});\n"
            f"write, table=table.{var:s}, file='{str(_get_tablefile(temp_dir, var)):s}';\n"
        )

    LOG.debug("Generating MADX jobfiles.")
    vars_per_proc = int(np.ceil(len(variables) / num_proc))

    # load template
    madx_script: str = _create_basic_job(accel_inst, k_values, variables)

    # build content for testing each variable
    for proc_index in range(num_proc):
        job_content = madx_script % {"TEMPFILE": str(_get_surveyfile(temp_dir, proc_index))}

        for i in range(vars_per_proc):
            try:
                # var to be tested
                current_var = variables[proc_index * vars_per_proc + i]
            except IndexError:
                break
            else:
                job_content += _assign(current_var, 1)
                job_content += _do_macro(current_var)
                job_content += _assign(current_var, 0)
                job_content += "\n"

        # last thing to do: get baseline
        if proc_index+1 == num_proc:
            job_content += _do_macro("0")

        _get_jobfile(temp_dir, proc_index).write_text(job_content)


def _call_madx(process_pool: multiprocessing.Pool, temp_dir: str, num_proc: int) -> None:
    """ Call madx in parallel """
    LOG.debug(f"Starting {num_proc:d} MAD-X jobs...")
    madx_jobs = [_get_jobfile(temp_dir, index) for index in range(num_proc)]
    failed = [LOG.error(fail) for fail in process_pool.map(_launch_single_job, madx_jobs) if fail]
    if len(failed):
        raise RuntimeError(f"{len(failed):d} of {num_proc:d} Madx jobs failed!")
    LOG.debug("MAD-X jobs done.")


def _clean_up(variables: List[str], temp_dir: Path, num_proc: int) -> None:
    """ Merge Logfiles and clean temporary outputfiles """
    LOG.debug("Cleaning output and printing log...")
    for var in (variables + ["0"]):
        table_file = _get_tablefile(temp_dir, var)
        with suppress(FileNotFoundError):  # py3.8 missing_ok=True
            table_file.unlink()
    full_log = ""
    for index in range(num_proc):
        survey_path = _get_surveyfile(temp_dir, index)
        job_path = _get_jobfile(temp_dir, index)
        log_path = job_path.with_name(f"{job_path.name}.log")
        if log_path.exists():
            full_log += log_path.read_text()
        with suppress(FileNotFoundError):  # py3.8 missing_ok=True
            log_path.unlink()

        with suppress(FileNotFoundError):  # py3.8 missing_ok=True
            job_path.unlink()

        with suppress(FileNotFoundError):  # py3.8 missing_ok=True
            survey_path.unlink()
    LOG.debug(full_log)
    with suppress(OSError):
        temp_dir.rmdir()


def _load_madx_results(
    variables: Sequence[str],
    k_values: List[float],
    process_pool: multiprocessing.Pool,
    temp_dir: str,
) -> dict:
    """ Load the madx results in parallel and return var-tfs dictionary """
    LOG.debug("Loading Madx Results.")
    path_and_vars = []
    for value in variables:
        path_and_vars.append((temp_dir, value))

    _, base_tfs = _load_and_remove_twiss((temp_dir, "0"))
    mapping = dict([(order, {}) for order in k_values] + [(f"{order}L", {}) for order in k_values])
    for var, tfs_data in process_pool.map(_load_and_remove_twiss, path_and_vars):
        for order in k_values:
            diff = (tfs_data[order] - base_tfs[order])
            mask = diff != 0  # drop zeros, maybe abs(diff) < eps ?
            k_list = diff.loc[mask]
            mapping[order][var] = k_list
            mapping[f"{order}L"][var] = k_list.mul(base_tfs.loc[mask, "L"])
    return mapping


# Helper #####################################################################


def _get_orders(order: int) -> Sequence[str]:
    """ Returns a list of strings with K-values to be used """
    try:
        return [f"K{i:d}{s:s}" for i in range(3) for s in ["", "S"]]
    except TypeError:
        return [f"K{i:d}{s:s}" for i in range(*order) for s in ["", "S"]]


def _get_jobfile(folder: Path, index: int) -> Path:
    """ Return names for jobfile and iterfile according to index """
    return folder / f"job.varmap.{index:d}.madx"


def _get_tablefile(folder: Path, var: str) -> Path:
    """ Return name of the variable-specific table file """
    return folder / f"table.{var}"


def _get_surveyfile(folder: Path, index: int) -> Path:
    """ Returns the name of the temporary survey file """
    return folder / f"survey.{index:d}.tmp"


def _launch_single_job(inputfile_path: Path):
    """ Function for pool to start a single madx job """
    log_file = inputfile_path.with_name(f"{inputfile_path.name}.log")
    try:
        madx_wrapper.run_file(inputfile_path, log_file=log_file, cwd=inputfile_path.parent)
    except madx_wrapper.MadxError as e:
        return str(e)
    else:
        return None


def _load_and_remove_twiss(path_and_var: Tuple[Path, str]) -> Tuple[str, tfs.TfsDataFrame]:
    """ Function for pool to retrieve results """
    path, var = path_and_var
    twissfile = _get_tablefile(path, var)
    tfs_data = tfs.read(twissfile, index="NAME")
    return var, tfs_data


def _create_basic_job(accel_inst: Accelerator, k_values: List[str], variables: Sequence[str]) -> str:
    """ Create the madx-job basics needed
        TEMPFILE needs to be replaced in the returned string.
    """
    # basic sequence creation
    job_content: str = accel_inst.get_base_madx_script()

    # create a survey and save it to a temporary file
    job_content += (
        "select, flag=survey, clear;\n"
        f"select, flag=survey, pattern='^M.*\.B{accel_inst.beam:d}$', COLUMN=NAME, L;\n"
        "survey, file='%(TEMPFILE)s';\n"
        "readmytable, file='%(TEMPFILE)s', table=mytable;\n"
        "n_elem = table(mytable, tablelength);\n"
        "\n"
    )

    # create macro for assigning values to the k_values per element
    job_content += "assign_k_values(element) : macro = {\n"
    # job_content += "    value, element; show, element;\n"
    for k_val in k_values:
        # (see user guide 10.3 Bending Magnet)
        if k_val == "K0":
            job_content += f"    {k_val:s} = element->angle / element->L;\n"
        elif k_val == "K0S":
            job_content += f"    {k_val:s} = element->tilt / element->L;\n"
        else:
            job_content += f"    {k_val:s} = element->{k_val:s};\n"
    job_content += "};\n\n"

    # create macro for using the row index as variable (see madx userguide)
    job_content += (
        "create_row(tblname, rowidx) : macro = {\n"
        "    exec, assign_k_values(tabstring(tblname, name, rowidx));\n"
        "};\n\n"
    )

    # create macro to create the full table with loop over elements
    job_content += (
        "create_table(table_id) : macro = {\n"
        f"    create, table=table_id, column=_name, L, {','.join(k_values):s};\n"
        "    i_elem = 0;\n"
        "    while (i_elem < n_elem) {\n"
        "        i_elem = i_elem + 1;\n"
        "        setvars, table=mytable, row=i_elem;\n"  # mytable from above!
        "        exec, create_row(mytable, $i_elem);\n"  # mytable from above!
        "        fill,  table=table_id;\n"
        "    };\n"
        "};\n\n"
    )

    # set all variables to zero
    for var in variables:
        job_content += f"{var:s} = 0;\n"

    job_content += "\n"
    return job_content


# Wrapper ##################################################################


def check_varmap_file(accel_inst: Accelerator, vars_categories):
    """ Checks on varmap file and creates it if not in model folder.
    """
    if accel_inst.modifiers is None:
        raise ValueError("Optics not defined. Please provide modifiers.madx. "
                         "Otherwise MADX evaluation might be unstable.")

    varmap_path = Path(accel_inst.model_dir) / f"varmap_{'_'.join(sorted(set(vars_categories)))}.{EXT}"
    if not varmap_path.exists():
        LOG.info(f"Variable mapping '{str(varmap_path):s}' not found. "
                 "Evaluating it via madx.")
        mapping = evaluate_for_variables(accel_inst, vars_categories)
        write_varmap(varmap_path, mapping)

    return varmap_path
