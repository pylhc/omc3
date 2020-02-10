from os import listdir
from os.path import abspath, dirname, isdir, isfile, join
from shutil import rmtree
import itertools
import tfs
import pytest
from omc3.hole_in_one import _optics_entrypoint  # <- Protected member of module. Make public?
from omc3.model import manager
from omc3.optics_measurements import measure_optics
from omc3.utils import stats
from omc3.utils.contexts import timeit
from .twiss_to_lin import optics_measurement_test_files

LIMITS = {'P': 1e-4, 'B': 3e-3, 'D': 1e-2, 'A': 6e-3}
LIMITS_3BPM = {'P': 1e-4, 'B': 3e-3, 'D': 1e-2, 'A': 0.95}
DEFAULT_LIMIT = 5e-3
BASE_PATH = abspath(join(dirname(__file__), "..", "results"))


MEASURE_OPTICS_INPUT = list(
    itertools.product(
        ["model", "equation", "none"],  # compensation
        [2],                            # coupling method
        [11],                           # range of bpms
        [True, False],                        # three bpm
        [False],                        # second order dispersion
    )
)


def _create_input(motion):
    dpps = [0, 0, 0, -4e-4, -4e-4, 4e-4, 4e-4, 5e-5, -3e-5, -2e-5]
    print(f"\nInput creation: {dpps}")
    opt_dict = dict(accel="lhc", year="2018", ats=True, beam=1, files=[""],
                    model_dir=join(dirname(__file__), "..", "inputs", "models", "25cm_beam1"),
                    outputdir=BASE_PATH)
    optics_opt, rest = _optics_entrypoint(opt_dict)
    optics_opt.accelerator = manager.get_accelerator(rest)
    lins = optics_measurement_test_files(opt_dict["model_dir"], dpps, motion)
    return lins, optics_opt

CREATE_INPUT=dict(free=_create_input("free"), driven=_create_input("driven"))

def set_optics_opt(optics_opt, compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp):
    optics_opt["compensation"] = compensation
    optics_opt["coupling_method"] = coupling_method
    optics_opt["range_of_bpms"] = range_of_bpms
    optics_opt["three_bpm_method"] = three_bpm_method
    optics_opt["second_order_dispersion"] = second_order_disp
    return optics_opt


@pytest.mark.parametrize("compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp", MEASURE_OPTICS_INPUT)
def test_single_file(compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp):
    lins, optics_opt = CREATE_INPUT["free" if compensation == 'none' else "driven"]
    optics_opt = set_optics_opt(optics_opt, compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp)
    optics_opt["outputdir"] = join(BASE_PATH, "single")
    inputs = measure_optics.InputFiles([lins[0]], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt, LIMITS_3BPM if three_bpm_method else LIMITS)


@pytest.mark.parametrize("compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp", MEASURE_OPTICS_INPUT)
def test_3_onmom_files(compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp):
    lins, optics_opt = CREATE_INPUT["free" if compensation == 'none' else "driven"]
    optics_opt = set_optics_opt(optics_opt, compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp)
    optics_opt["outputdir"] = join(BASE_PATH, "onmom")
    inputs = measure_optics.InputFiles(lins[:3], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt, LIMITS_3BPM if three_bpm_method else LIMITS)


@pytest.mark.parametrize("compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp", MEASURE_OPTICS_INPUT)
def test_3_pseudo_onmom_files(compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp):
    lins, optics_opt = CREATE_INPUT["free" if compensation == 'none' else "driven"]
    optics_opt = set_optics_opt(optics_opt, compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp)
    optics_opt["outputdir"] = join(BASE_PATH, "pseudo_onmom")
    inputs = measure_optics.InputFiles(lins[-3:], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt, LIMITS_3BPM if three_bpm_method else LIMITS)


@pytest.mark.parametrize("compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp", MEASURE_OPTICS_INPUT)
def test_offmom_files(compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp):
    lins, optics_opt = CREATE_INPUT["free" if compensation == 'none' else "driven"]
    optics_opt = set_optics_opt(optics_opt, compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp)
    optics_opt["chromatic_beating"] = True
    optics_opt["outputdir"] = join(BASE_PATH, "offmom")
    inputs = measure_optics.InputFiles(lins[:7], optics_opt)
    _run_evaluate_and_clean_up(inputs, optics_opt, LIMITS_3BPM if three_bpm_method else LIMITS)


def _run_evaluate_and_clean_up(inputs, optics_opt, limits):
    with timeit(lambda spanned: print(f"\nTotal time for optics measurements: {spanned}")):
        measure_optics.measure_optics(inputs, optics_opt)
    evaluate_accuracy(optics_opt.outputdir, limits)
    _clean_up(optics_opt.outputdir)


def evaluate_accuracy(meas_path, limits):
    for f in [f for f in listdir(meas_path) if (isfile(join(meas_path, f)) and (".tfs" in f))]:
        a = tfs.read(join(meas_path, f))
        cols = [column for column in a.columns.to_numpy() if column.startswith('DELTA')]
        if f == "normalised_dispersion_x.tfs":
            cols.remove("DELTADX")
        for col in cols:
            rms = stats.weighted_rms(a.loc[:, col].to_numpy(), errors=a.loc[:, f"ERR{col}"].to_numpy())
            if col[5] in limits.keys():
                assert rms < limits[col[5]], "\nFile: {:25}  Column: {:15}   RMS: {:.6f}".format(f, col, rms)
            else:
                assert rms < DEFAULT_LIMIT, "\nFile: {:25}  Column: {:15}   RMS: {:.6f}".format(f, col, rms)
            print(f"\nFile: {f:25}  Column: {col[5:]:15}   RMS:    {rms:.6f}")


def _clean_up(path_dir):
    if isdir(path_dir):
        rmtree(path_dir, ignore_errors=True)
