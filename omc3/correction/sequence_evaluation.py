"""
Similar to Sequence Parser but with MADX

First: Set all variables to 0
Then: Set one variable at a time to 1

Compare results with case all==0

"""
import multiprocessing
import os
import pickle
import numpy as np
import madx_wrapper
from utils import logging_tools, iotools
import tfs
from utils.contexts import timeit, suppress_exception

LOG = logging_tools.get_logger(__name__)

EXT = "varmap"  # Extension Standard


# Read Sequence ##############################################################


def evaluate_for_variables(accel_inst, variable_categories, order=4,
                           num_proc=multiprocessing.cpu_count(),
                           temp_dir=None):
    """ Generate a dictionary containing response matrices for
        beta, phase, dispersion, tune and coupling and saves it to a file.

        Args:
            accel_inst : Accelerator Instance.
            variable_categories (list): Categories of the variables/knobs to use. (from .json)
            order (int or tuple): Max or [min, max] of K-value order to use.
            num_proc (int): Number of processes to use in parallel.
            temp_dir (str): temporary directory. If ``None``, uses model_dir.
    """
    LOG.debug("Generating Fullresponse via Mad-X.")
    with timeit(lambda t: LOG.debug("  Total time generating fullresponse: {:f}s".format(t))):
        if not temp_dir:
            temp_dir = accel_inst.model_dir
        iotools.create_dirs(temp_dir)

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


def _generate_madx_jobs(accel_inst, variables, k_values, num_proc, temp_dir):
    """ Generates madx job-files """
    def _assign(var, value):
        return "{var:s} = {value:d};\n".format(var=var, value=value)

    def _do_macro(var):
        return (
            "exec, create_table({table:s});\n"
            "write, table={table:s}, file='{f_out:s}';\n"
        ).format(
            table="table." + var,
            f_out=_get_tablefile(temp_dir, var),
        )

    LOG.debug("Generating MADX jobfiles.")
    vars_per_proc = int(np.ceil(len(variables) / num_proc))

    # load template
    madx_script = _create_basic_job(accel_inst, k_values, variables)

    # build content for testing each variable
    for proc_idx in range(num_proc):
        job_content = madx_script % {"TEMPFILE": _get_surveyfile(temp_dir, proc_idx)}

        for i in range(vars_per_proc):
            try:
                # var to be tested
                current_var = variables[proc_idx * vars_per_proc + i]
            except IndexError:
                break
            else:
                job_content += _assign(current_var, 1)
                job_content += _do_macro(current_var)
                job_content += _assign(current_var, 0)
                job_content += "\n"

        # last thing to do: get baseline
        if proc_idx+1 == num_proc:
            job_content += _do_macro("0")

        with open(_get_jobfile(temp_dir, proc_idx), "w") as job_file:
            job_file.write(job_content)


def _call_madx(process_pool, temp_dir, num_proc):
    """ Call madx in parallel """
    LOG.debug("Starting {:d} MAD-X jobs...".format(num_proc))
    madx_jobs = [_get_jobfile(temp_dir, index) for index in range(num_proc)]
    failed = [LOG.error(fail) for fail in process_pool.map(_launch_single_job, madx_jobs) if fail]
    if len(failed):
        raise RuntimeError("{:d} of {:d} Madx jobs failed!".format(len(failed), num_proc))
    LOG.debug("MAD-X jobs done.")


def _clean_up(variables, temp_dir, num_proc):
    """ Merge Logfiles and clean temporary outputfiles """
    LOG.debug("Cleaning output and printing log...")
    for var in (variables + ["0"]):
        with suppress_exception(OSError):
            os.remove(_get_tablefile(temp_dir, var))
    full_log = ""
    for index in range(num_proc):
        survey_path = _get_surveyfile(temp_dir, index)
        job_path = _get_jobfile(temp_dir, index)
        log_path = job_path + ".log"
        with open(log_path, "r") as log_file:
            full_log += log_file.read()
        with suppress_exception(OSError):
            os.remove(log_path)
        with suppress_exception(OSError):
            os.remove(job_path)
        with suppress_exception(OSError):
            os.remove(survey_path)
    LOG.debug(full_log)

    with suppress_exception(OSError):
        os.rmdir(temp_dir)


def _load_madx_results(variables, k_values, process_pool, temp_dir):
    """ Load the madx results in parallel and return var-tfs dictionary """
    LOG.debug("Loading Madx Results.")
    path_and_vars = []
    for value in variables:
        path_and_vars.append((temp_dir, value))

    _, base_tfs = _load_and_remove_twiss((temp_dir, "0"))
    mapping = dict([(o, {}) for o in k_values] +
                   [(o + "L", {}) for o in k_values])
    for var, tfs_data in process_pool.map(_load_and_remove_twiss, path_and_vars):
        for o in k_values:
            diff = (tfs_data[o] - base_tfs[o])
            mask = diff != 0  # drop zeros, maybe abs(diff) < eps ?
            k_list = diff.loc[mask]
            mapping[o][var] = k_list
            mapping[o + "L"][var] = k_list.mul(base_tfs.loc[mask, "L"])
    return mapping


# Helper #####################################################################


def _get_orders(order):
    """ Returns a list of strings with K-values to be used """
    try:
        return ["K{:d}{:s}".format(i, s) for i in range(order) for s in ["", "S"]]
    except TypeError:
        return ["K{:d}{:s}".format(i, s) for i in range(*order) for s in ["", "S"]]


def _get_jobfile(folder, index):
    """ Return names for jobfile and iterfile according to index """
    return os.path.join(folder, "job.varmap.{:d}.madx".format(index))


def _get_tablefile(folder, var):
    """ Return name of the variable-specific table file """
    return os.path.join(folder, "table." + var)


def _get_surveyfile(folder, index):
    """ Returns the name of the macro """
    return os.path.join(folder, "survey.{:d}.tmp".format(index))


def _launch_single_job(inputfile_path):
    """ Function for pool to start a single madx job """
    log_file = inputfile_path + ".log"
    try:
        madx_wrapper.run_file(inputfile_path, log_file=log_file)
    except madx_wrapper.MadxError as e:
        return str(e)
    else:
        return None


def _load_and_remove_twiss(path_and_var):
    """ Function for pool to retrieve results """
    path, var = path_and_var
    twissfile = _get_tablefile(path, var)
    tfs_data = tfs.read(twissfile, index="NAME")
    return var, tfs_data


def _create_basic_job(accel_inst, k_values, variables):
    """ Create the madx-job basics needed
        TEMPFILE needs to be replaced in the returned string.
    """
    # basic sequence creation
    job_content = accel_inst.get_base_madx_script(accel_inst.model_dir)

    # create a survey and save it to a temporary file
    job_content += (
        "select, flag=survey, clear;\n"
        "select, flag=survey, pattern='^M.*\.B{beam:d}$', COLUMN=NAME, L;\n"
        "survey, file='%(TEMPFILE)s';\n"
        "readmytable, file='%(TEMPFILE)s', table=mytable;\n"
        "n_elem = table(mytable, tablelength);\n"
        "\n"
    ).format(beam=accel_inst.beam)

    # create macro for assigning values to the k_values per element
    job_content += "assign_k_values(element) : macro = {\n"
    # job_content += "    value, element; show, element;\n"
    for k_val in k_values:
        #TODO: Ask someone who knows about MADX if K0-handling is correct
        # (see user guide 10.3 Bending Magnet)
        if k_val == "K0":
            job_content += "    {k:s} = element->angle / element->L;\n".format(k=k_val)
        elif k_val == "K0S":
            job_content += "    {k:s} = element->tilt / element->L;\n".format(k=k_val)
        else:
            job_content += "    {k:s} = element->{k:s};\n".format(k=k_val)
    job_content += "};\n\n"

    # create macro for using the row index as variable (see madx userguide)
    job_content += (
        "create_row(tblname, rowidx) : macro = {\n"
        "    exec, assign_k_values(tabstring(tblname, name, rowidx));\n"
        "};\n\n"
    )

    # create macro to create the full table with loop over elements
    job_content += (
        "create_table(table_id) : macro = {{\n"
        "    create, table=table_id, column=_name, L, {col:s};\n"
        "    i_elem = 0;\n"
        "    while (i_elem < n_elem) {{\n"
        "        i_elem = i_elem + 1;\n"
        "        setvars, table=mytable, row=i_elem;\n"  # mytable from above!
        "        exec, create_row(mytable, $i_elem);\n"  # mytable from above!
        "        fill,  table=table_id;\n"
        "    }};\n"
        "}};\n\n"
    ).format(col=",".join(k_values))

    # set all variables to zero
    for var in variables:
        job_content += "{var:s} = 0;\n".format(var=var)

    job_content += "\n"
    return job_content


# Wrapper ##################################################################


def check_varmap_file(accel_inst, vars_categories):
    """ Checks on varmap file and creates it if not in model folder.
    THIS SHOULD BE REPLACED WITH A CALL TO JAIMES DATABASE, IF IT BECOMES AVAILABLE """
    if accel_inst.modifiers is None:
        raise ValueError("Optics not defined. Please provide modifiers.madx. "
                         "Otherwise MADX evaluation might be unstable.")

    varmapfile_name = f"{accel_inst.NAME.lower()}b{accel_inst.beam:d}_{'_'.join(sorted(set(vars_categories)))}"

    varmap_path = os.path.join(accel_inst.model_dir, varmapfile_name + "." + EXT)
    if not os.path.isfile(varmap_path):
        LOG.info("Variable mapping '{:s}' not found. Evaluating it via madx.".format(varmap_path))
        mapping = evaluate_for_variables(accel_inst, vars_categories)
        with open(varmap_path, 'wb') as dump_file:
            pickle.dump(mapping, dump_file, protocol=-1)

    return varmap_path
