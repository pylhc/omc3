from os import listdir
from shutil import rmtree
from os.path import isfile, join, dirname, abspath, isdir
import pytest

from .twiss_to_lin import optics_measurement_test_files
from . import context

import tfs
from utils import stats
from optics_measurements import measure_optics
from utils.contexts import timeit
from model import manager
from hole_in_one import _optics_entrypoint

LIMITS = {'P': 1e-4, 'B': 3e-3, 'D': 1e-2, 'A': 5e-3}
DEFAULT_LIMIT = 5e-3
BASE_PATH = abspath(join(dirname(__file__), "..", "results"))


def _create_input():
    dpps = [0, 0, 0, -4e-4, -4e-4, 4e-4, 4e-4, 5e-5, -3e-5, -2e-5]
    print(f"\nInput creation: {dpps}")
    opt_dict = dict(accel="lhc", lhc_mode="lhc_runII_2018", beam=1, files=[""],
                    model_dir=join(dirname(__file__), "..", "inputs", "models", "25cm_beam1"),
                    outputdir=BASE_PATH)
    optics_opt, rest = _optics_entrypoint(opt_dict)
    optics_opt.accelerator = manager.get_accel_instance(rest)
    lins = optics_measurement_test_files(opt_dict["model_dir"], dpps)
    return lins, optics_opt


INPUT_CREATED = _create_input()


def test_single_file():
    lins, optics_opt = INPUT_CREATED
    optics_opt["compensation"] = "model"
    optics_opt["outputdir"] = join(BASE_PATH, "single")
    inputs = measure_optics.InputFiles([lins[0]], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt)


def test_single_file_eq():
    lins, optics_opt = INPUT_CREATED
    optics_opt["compensation"] = "equation"
    optics_opt["outputdir"] = join(BASE_PATH, "single_eq")
    inputs = measure_optics.InputFiles([lins[0]], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt)


def test_3_onmom_files():
    lins, optics_opt = INPUT_CREATED
    optics_opt["compensation"] = "model"
    optics_opt["outputdir"] = join(BASE_PATH, "onmom")
    inputs = measure_optics.InputFiles(lins[:3], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt)


def test_3_onmom_files_eq():
    lins, optics_opt = INPUT_CREATED
    optics_opt["compensation"] = "equation"
    optics_opt["outputdir"] = join(BASE_PATH, "onmom_eq")
    inputs = measure_optics.InputFiles(lins[:3], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt)


def test_3_pseudo_onmom_files():
    lins, optics_opt = INPUT_CREATED
    optics_opt["compensation"] = "model"
    optics_opt["outputdir"] = join(BASE_PATH, "pseudo_onmom")
    inputs = measure_optics.InputFiles(lins[-3:], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt)


def test_3_pseudo_onmom_files_eq():
    lins, optics_opt = INPUT_CREATED
    optics_opt["compensation"] = "equation"
    optics_opt["outputdir"] = join(BASE_PATH, "pseudo_onmom_eq")
    inputs = measure_optics.InputFiles(lins[-3:], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt)


def test_offmom_files():
    lins, optics_opt = INPUT_CREATED
    optics_opt["compensation"] = "model"
    #optics_opt["chromatic_beating"] = True
    optics_opt["outputdir"] = join(BASE_PATH, "offmom")
    inputs = measure_optics.InputFiles(lins[:7], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt)


def test_offmom_files_eq():
    lins, optics_opt = INPUT_CREATED
    #optics_opt["chromatic_beating"] = True
    optics_opt["compensation"] = "equation"
    optics_opt["outputdir"] = join(BASE_PATH, "offmom_eq")
    inputs = measure_optics.InputFiles(lins[:7], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt)


def _run_evaluate_and_clean_up(inputs, optics_opt):
    with timeit(lambda spanned: print(f"\nTotal time for optics measurements: {spanned}")):
        measure_optics.measure_optics(inputs, optics_opt)
    evaluate_accuracy(optics_opt.outputdir)
    _clean_up(optics_opt.outputdir)


def evaluate_accuracy(meas_path):
    for f in [f for f in listdir(meas_path) if (isfile(join(meas_path, f)) and (".tfs" in f))]:
        a = tfs.read(join(meas_path, f))
        cols = [column for column in a.columns.values if column.startswith('DELTA')]
        if f == "normalised_dispersion_x.tfs":
            cols.remove("DELTADX")
        for col in cols:
            rms = stats.weighted_rms(a.loc[:, col].values, errors=a.loc[:, f"ERR{col}"].values)
            if col[5] in LIMITS.keys():
                assert rms < LIMITS[col[5]], "\nFile: {:25}  Column: {:15}   RMS: {:.6f}".format(f, col, rms)
            else:
                assert rms < DEFAULT_LIMIT, "\nFile: {:25}  Column: {:15}   RMS: {:.6f}".format(f, col, rms)
            print(f"\nFile: {f:25}  Column: {col[5:]:15}   RMS:    {rms:.6f}")


def _clean_up(path_dir):
    if isdir(path_dir):
        rmtree(path_dir, ignore_errors=True)