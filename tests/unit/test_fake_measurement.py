from pathlib import Path
from shutil import copy

import matplotlib
import pytest
import numpy as np
import tfs
import pandas as pd
from matplotlib.figure import Figure

from omc3.correction.model_appenders import add_coupling_to_model
from omc3.optics_measurements.constants import (NAME, S, ERR, MDL, DELTA, NORM_DISP_NAME, PHASE_NAME, TOTAL_PHASE_NAME)
from omc3.correction.constants import NORM_DISP, DISP, BETA
from omc3.plotting.plot_spectrum import main as plot_spectrum
from omc3.plotting.spectrum.utils import PLANES, get_unique_filenames
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement, _get_data

INPUT_DIR = Path(__file__).parent.parent / "inputs"


# Input Tests ------------------------------------------------------------------

@pytest.mark.basic
def test_get_data_string(beam1_path):
    twiss, model = _get_data(str(beam1_path))
    assert twiss.any().any()
    assert twiss.equals(model)

@pytest.mark.basic
def test_get_data_path(beam1_path):
    twiss, model = _get_data(beam1_path)
    assert twiss.any().any()
    assert twiss.equals(model)


@pytest.mark.basic
def test_get_data_dataframe(beam1_path):
    twiss1 = tfs.read(beam1_path, index="NAME")
    twiss, model = _get_data(twiss1)
    assert twiss1.equals(twiss)
    assert twiss1.equals(model)


@pytest.mark.basic
def test_get_data_model(beam1_path, beam2_path):
    twiss, model = _get_data(twiss=beam1_path, model=beam2_path)
    assert twiss.any().any()
    assert model.any().any()
    assert not twiss.equals(model)


# Run Test ---------------------------------------------------------------------

@pytest.mark.basic
def test_run_and_output(tmp_path, beam1_path):
    results = fake_measurement(
        twiss=beam1_path,
        randomize=None,
        outputdir=tmp_path,
    )
    assert len(list(tmp_path.glob("*.tfs"))) == len(results)

    model = _full_model(beam1_path)
    for name, df in results.items():
        if name.startswith(PHASE_NAME):
            assert df[S].equals(model.loc[df.index, S])
        else:
            assert df[S].equals(model[S])

        error_columns = _error_columns(df)
        model_columns = _model_columns(df)
        delta_columns = _delta_columns(df)
        assert len(error_columns)
        assert len(model_columns)
        assert len(delta_columns)

        for col in list(error_columns) + list(delta_columns):
            assert (df[col] == 0).all()

        for col in model_columns:
            param = col[:-len(MDL)]
            if param in df.columns:
                assert df[col].equals(df[param])

            if name[:-1] not in (PHASE_NAME, TOTAL_PHASE_NAME):
                assert df[col].equals(model[param])


def test_run_random(beam1_path):
    # results = fake_measurement(
    #     twiss=beam1_path,
    #     # randomize is set automatically
    # )
    pass


@pytest.mark.basic
def test_beta():
    pass


@pytest.mark.basic
def test_dispersion():
    pass


@pytest.mark.basic
def test_normalized_dispersion():
    pass


@pytest.mark.basic
def test_phase():
    pass


@pytest.mark.basic
def test_diff_coupling():
    pass


@pytest.mark.basic
def test_sum_coupling():
    pass


# Fixtures ------

@pytest.fixture
def beam1_path():
    return INPUT_DIR / "models" / "25cm_beam1" / "twiss.dat"


@pytest.fixture
def beam2_path():
    return INPUT_DIR / "models" / "25cm_beam2" / "twiss.dat"


# Helper -----

def _error_columns(df):
    return df.columns[df.columns.str.startswith(ERR)]


def _model_columns(df):
    return df.columns[df.columns.str.endswith(MDL)]


def _delta_columns(df):
    return df.columns[df.columns.str.contains(DELTA)]


def _full_model(path: Path):
    model = tfs.read(path, index=NAME)
    model = add_coupling_to_model(model)
    model[f"{NORM_DISP}X"] = model[f"{DISP}X"] / np.sqrt(model[f"{BETA}X"])
    return model
