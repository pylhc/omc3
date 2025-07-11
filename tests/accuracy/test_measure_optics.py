import itertools
from pathlib import Path

import numpy as np
import pytest
import tfs

from omc3.hole_in_one import _optics_entrypoint  # <- Protected member of module. Make public?
from omc3.model import manager
from omc3.optics_measurements import measure_optics
from omc3.optics_measurements.constants import (
    ALPHA,
    BETA,
    DISPERSION,
    NORM_DISPERSION,
    PHASE,
    SPECIAL_PHASE_NAME,
)
from omc3.optics_measurements.data_models import InputFiles
from omc3.optics_measurements.phase import CompensationMode
from omc3.utils import logging_tools, stats
from omc3.utils.contexts import timeit
from tests.accuracy.twiss_to_lin import optics_measurement_test_files

LOG = logging_tools.get_logger(__name__)
# LOG = logging_tools.get_logger('__main__')  # debugging

LIMITS = {
    PHASE: 1e-4,
    ALPHA: 6e-3,
    BETA: 3e-3,
    DISPERSION: 1.1e-2,
    NORM_DISPERSION: 5e-3,
    '': 5e-3  # orbit
}
BASE_PATH = Path(__file__).parent.parent / "results"
INPUTS = Path(__file__).parent.parent / 'inputs'

DPPS = [0, 0, 0, -4e-4, -4e-4, 4e-4, 4e-4, 5e-5, -3e-5, -2e-5]  # defines the slicing

MEASURE_OPTICS_SETTINGS = dict(
    compensation=CompensationMode.all(),
    coupling_method=[2],
    range_of_bpms=[11],
    three_bpm_method=[False],
    second_order_disp=[False],
)
VALUES_GRID = list(itertools.product(*MEASURE_OPTICS_SETTINGS.values()))  # easy to add more tests in grid above
PARAMS = ", ".join(MEASURE_OPTICS_SETTINGS)


@pytest.mark.basic
def test_single_file_single_input(tmp_path, input_data):
    test_measure_optics(tmp_path, input_data, slice(0, 1), *VALUES_GRID[0])


@pytest.mark.basic
def test_3_onmom_files_single_input(tmp_path, input_data):
    test_measure_optics(tmp_path, input_data, slice(None, 3), *VALUES_GRID[1])


@pytest.mark.extended
@pytest.mark.parametrize('input_data', (1, 2), ids=["Beam1", "Beam2"], indirect=True)
@pytest.mark.parametrize(PARAMS, VALUES_GRID)
@pytest.mark.parametrize("lin_slice",
                         (slice(0, 1), slice(None, 3), slice(-3, None), slice(None, 7)),
                         ids=("single_file", "3_files_onmom", "3_files_pseudo_onmom", "offmom"))
def test_measure_optics(
        tmp_path, input_data, lin_slice,
        compensation, coupling_method, range_of_bpms, three_bpm_method, second_order_disp):
    data = input_data["free" if compensation == CompensationMode.NONE else "driven"]
    lins, optics_opt = data['lins'], data['optics_opt']
    optics_opt.update(
        outputdir=tmp_path,
        compensation=compensation,
        coupling_method=coupling_method,
        range_of_bpms=range_of_bpms,
        three_bpm_method=three_bpm_method,
        second_order_disp=second_order_disp,
        chromatic_beating=lin_slice == slice(None, 7),
    )
    inputs = InputFiles(lins[lin_slice], optics_opt)
    with timeit(lambda spanned: LOG.debug(f"\nTotal time for optics measurements: {spanned}")):
        measure_optics.measure_optics(inputs, optics_opt)
    evaluate_accuracy(optics_opt.outputdir, LIMITS)


# Helper ---


def evaluate_accuracy(meas_path, limits):
    for f in meas_path.glob("*.tfs"):  # maybe a simple list of files to test wouldn't be too bad?
        if "f10" in f.name or "phase_driven" in f.name:
            continue
        df = tfs.read(f)
        cols = df.columns[df.columns.str.startswith('DELTA')]
        for col in cols:
            if f.name.startswith('normalised_dispersion') and col.startswith('DELTAD'):
                continue

            rms = stats.weighted_rms(
                data=df.loc[:, col].to_numpy(),
                errors=df.loc[:, f"ERR{col}"].to_numpy()
            )
            assert rms < limits[col[5:-1]], f"\n{f.name:25}  {col:15}   RMS: {rms:.1e}"
            LOG.info(f"{f.name:25}  {col[5:]:15}   RMS: {rms:.1e}")
    assert ((meas_path / f"{SPECIAL_PHASE_NAME}x.tfs").is_file() and (meas_path / f"{SPECIAL_PHASE_NAME}y.tfs").is_file())



@pytest.fixture(scope="module", params=(1,), ids=("Beam1",))
def input_data(request, tmp_path_factory):
    """Creates the input lin data and optics_options."""
    data = {}
    beam = request.param
    for motion in ("free", "driven"):
        np.random.seed(12345678)
        output_path = tmp_path_factory.mktemp(f"input_{motion}_b{beam}")

        opt_dict = dict(accel="lhc", year="2018", ats=True, beam=beam, files=[""],
                        model_dir=INPUTS / "models" / f"2018_col_b{beam}_25cm",
                        outputdir=output_path)
        optics_opt, rest = _optics_entrypoint(opt_dict)
        optics_opt.accelerator = manager.get_accelerator(rest)
        lins = optics_measurement_test_files(opt_dict["model_dir"], DPPS, motion,
                                             beam_direction=(1 if beam == 1 else -1))
        data[motion] = {'lins': lins, 'optics_opt': optics_opt}
    return data
