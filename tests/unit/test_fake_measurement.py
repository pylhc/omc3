from pathlib import Path
from shutil import copy

import matplotlib
import pytest
import numpy as np
import tfs
import pandas as pd
from matplotlib.figure import Figure
from optics_functions.constants import LENGTH

from omc3.correction.model_appenders import add_coupling_to_model
from omc3.optics_measurements.constants import (NAME, S, ERR, MDL, DELTA, NORM_DISP_NAME, PHASE_NAME, TOTAL_PHASE_NAME,
                                                BETA_NAME, AMP_BETA_NAME, DISPERSION_NAME)
from omc3.correction.constants import NORM_DISP, DISP, BETA, F1010, F1001, PHASE, TUNE, PHASE_ADV
from omc3.plotting.plot_spectrum import main as plot_spectrum
from scipy import stats
from omc3.plotting.spectrum.utils import PLANES, get_unique_filenames
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from omc3.scripts.fake_measurement_from_model import (_get_data, OPTICS_PARAMETERS, OUTPUTNAMES_MAP, VALUES, ERRORS)

INPUT_DIR = Path(__file__).parent.parent / "inputs"


# Input Tests ------------------------------------------------------------------

@pytest.mark.basic
def test_get_data_string(beam1_path):
    """ Test data loading with string paths. """
    twiss, model = _get_data(str(beam1_path))
    assert twiss.any().any()
    assert twiss.equals(model)

@pytest.mark.basic
def test_get_data_path(beam1_path):
    """ Test data loading with Path."""
    twiss, model = _get_data(beam1_path)
    assert twiss.any().any()
    assert twiss.equals(model)


@pytest.mark.basic
def test_get_data_dataframe(beam1_path):
    """ Tests DataFrames as input. """
    twiss1 = tfs.read(beam1_path, index="NAME")
    twiss, model = _get_data(twiss1)
    assert twiss1.equals(twiss)
    assert twiss1.equals(model)


@pytest.mark.basic
def test_get_data_model(beam1_path, beam2_path):
    """ Test that twiss and model are different if two are given. """
    twiss, model = _get_data(twiss=beam1_path, model=beam2_path)
    assert twiss.any().any()
    assert model.any().any()
    assert not twiss.equals(model)


# Run Test ---------------------------------------------------------------------

@pytest.mark.basic
def test_run_and_output(tmp_path, both_beams_path):
    """ Tests a full run and checks if output makes sense. No errors, no randomization. """
    results = fake_measurement(
        twiss=both_beams_path,
        randomize=None,
        outputdir=tmp_path,
    )
    assert len(list(tmp_path.glob("*.tfs"))) == len(results)

    model = _full_model(both_beams_path)
    for name, df in results.items():
        assert len(df.headers)
        assert f"{TUNE}1" in df.headers
        assert f"{TUNE}2" in df.headers

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
            assert (df[col] == 0).all()  # randomization is off and errors 0 ...

        for col in model_columns:
            param = col[:-len(MDL)]
            if param in df.columns:
                assert df[col].equals(df[param])  # ... so all values == model values

            if name[:-1] not in (PHASE_NAME, TOTAL_PHASE_NAME):
                assert df[col].equals(model[param])  # ... so all values == model values


@pytest.mark.basic
@pytest.mark.parametrize('randomize', [[VALUES, ERRORS], [VALUES], [ERRORS], []], ids=["errors,values", "values", "errors", "None"])
def test_run_random(both_beams_path, randomize):
    """ Tests errors and values and if applicable their randomization (very basically)."""
    error_val = 0.1
    results = fake_measurement(
        twiss=both_beams_path,
        randomize=randomize,
        relative_errors=[error_val],
        seed=2230,
    )

    # Test the outputs ---
    for name, df in results.items():
        _test_error_columns(name, df, ERRORS in randomize, error_val)
        _test_delta_columns(name, df, VALUES in randomize)
        _test_model_columns(name, df, VALUES in randomize)


def _test_error_columns(name, df, randomized, error_val):
    error_columns = _error_columns(df)
    assert len(error_columns)
    for col in error_columns:
        assert not df[col].isna().any()
        param = col.replace(ERR, "").replace(DELTA, "")
        if not df[f"{param}{MDL}"].any():
            continue  # skip all-zero model (e.g. DY)

        if randomized:
            if name.startswith(TOTAL_PHASE_NAME):
                assert sum(df[col] == 0) == 1  # first entry is zero
            else:
                assert (df[col] != 0).all()  # all should be different as randomized
        else:
            idx = df.index
            if name.startswith(TOTAL_PHASE_NAME):
                idx = df.index[1:]

            if name[:-1] in (PHASE_NAME, TOTAL_PHASE_NAME):
                # phase errors are equal to the relative error
                assert not any(df.loc[idx, col] - error_val)
            elif name[:-1] in (BETA_NAME, AMP_BETA_NAME) and DELTA in col:
                # errdeltabet (beating) errors are also equal to the relative error,
                # but with less precision
                assert all(np.abs(df.loc[idx, col]-error_val) < 1e-16)
            elif name[:-1] in (NORM_DISP_NAME):
                # not sure how to test this, but should already be tested with disp and beta
                assert all(df.loc[idx, col])
            else:
                # errors are just abs(error_val * val)
                assert all(df.loc[idx, col] == np.abs(df[f"{param}{MDL}"] * error_val))


def _test_delta_columns(name, df, randomized):
    delta_columns = _delta_columns(df)
    assert len(delta_columns)
    for col in delta_columns:
        assert not df[col].isna().any()
        param = col.replace(ERR, "").replace(DELTA, "")
        if not df[f"{param}{MDL}"].any():
            continue  # skip all-zero model (e.g. DY)

        if randomized:
            if name.startswith(TOTAL_PHASE_NAME):
                assert sum(df[col] == 0) == 1  # first entry is zero
            else:
                assert (df[col] != 0).all()  # all should be different as randomized
        else:
            assert not df[col].any()


def _test_model_columns(name, df, randomized):
    model_columns = _model_columns(df)
    assert len(model_columns)
    for col in model_columns:
        param = col[:-len(MDL)]
        assert not df[col].isna().any()

        if param in df.columns:
            if randomized:
                if df[param].any():
                    assert not df[col].equals(df[param])
                else:
                    assert not df[col].any()
            else:
                assert df[col].equals(df[param])
            assert not df[param].isna().any()


@pytest.mark.parametrize("parameter", ["PHASEX"])
# @pytest.mark.parametrize("parameter", OPTICS_PARAMETERS)
@pytest.mark.basic
def test_parameter(both_beams_path, parameter):
    """ Test each parameter individually and checks if the randomization makes sense."""
    relative_error = 0.1
    randomize = [VALUES, ERRORS]

    results = fake_measurement(
        twiss=both_beams_path,
        randomize=randomize,
        relative_errors=[relative_error],
        parameters=[parameter],
        seed=2021,
    )

    assert len(results)
    assert all(name in results.keys() for name in OUTPUTNAMES_MAP[parameter])

    name_tester_map = {
        TOTAL_PHASE_NAME: _test_total_phase,
        PHASE_NAME: _test_phase,
        BETA_NAME: _test_beta,
        AMP_BETA_NAME: _test_beta,
        DISPERSION_NAME: _test_disp,
        NORM_DISP_NAME: _test_norm_disp,
        F1010[:-1].lower(): _test_coupling,
        F1001[:-1].lower(): _test_coupling,
    }

    for name, df in results.items():
        plane = parameter[-1]
        assert S in df.columns
        assert f"{PHASE_ADV}{plane}{MDL}" in df.columns
        name_tester_map[name[:-1]](df, plane, relative_error)


def _test_beta(df, plane, relative_error):
    assert f"{BETA}{plane}" in df.columns
    assert f"{BETA}{plane}{MDL}" in df.columns
    assert f"{DELTA}{BETA}{plane}" in df.columns
    assert f"{ERR}{BETA}{plane}" in df.columns
    assert f"{ERR}{DELTA}{BETA}{plane}" in df.columns

    assert all(df[f"{BETA}{plane}"]) > 0
    assert all(df[f"{ERR}{BETA}{plane}"]) > 0
    assert all(df[f"{ERR}{BETA}{plane}"] <= 5 * relative_error * np.abs(df[f"{BETA}{plane}{MDL}"]))
    assert all(np.abs(((df[f"{DELTA}{BETA}{plane}"] * df[f"{BETA}{plane}{MDL}"]) - df[f"{BETA}{plane}"] + df[f"{BETA}{plane}{MDL}"])) < 1e-12)
    assert all(np.abs(df[f"{ERR}{DELTA}{BETA}{plane}"] * df[f"{BETA}{plane}{MDL}"] - df[f"{ERR}{BETA}{plane}"]) < 1e-12)
    assert _gaussian_distribution_test(df[f"{BETA}{plane}"], df[f"{BETA}{plane}{MDL}"], df[f"{ERR}{BETA}{plane}"])


def _test_phase(df, plane, relative_error):
    assert f"{PHASE}{plane}" in df.columns
    assert f"{PHASE}{plane}{MDL}" in df.columns
    assert f"{DELTA}{PHASE}{plane}" in df.columns
    assert f"{ERR}{PHASE}{plane}" in df.columns
    assert f"{ERR}{DELTA}{PHASE}{plane}" in df.columns

    assert all(df[f"{PHASE}{plane}"] <= 0.5)
    assert all(df[f"{PHASE}{plane}"] >= -0.5)
    assert all(df[f"{ERR}{PHASE}{plane}"] > 0)
    assert all(df[f"{ERR}{PHASE}{plane}"] < 0.5)
    assert all(np.abs(df[f"{DELTA}{PHASE}{plane}"] - df[f"{PHASE}{plane}"] + df[f"{PHASE}{plane}{MDL}"]) % 1 < 1e-12)
    assert all(np.abs(df[f"{ERR}{DELTA}{PHASE}{plane}"] - df[f"{ERR}{PHASE}{plane}"]) == 0)
    assert _gaussian_distribution_test(df[f"{PHASE}{plane}"], df[f"{PHASE}{plane}{MDL}"], df[f"{ERR}{PHASE}{plane}"])


def _test_total_phase(df, plane, relative_error):
    assert f"{PHASE}{plane}" in df.columns
    assert f"{PHASE}{plane}{MDL}" in df.columns
    assert f"{DELTA}{PHASE}{plane}" in df.columns
    assert f"{ERR}{PHASE}{plane}" in df.columns
    assert f"{ERR}{DELTA}{PHASE}{plane}" in df.columns
    indx = df.index[1:]

    assert all(df[f"{PHASE}{plane}"] < 1)
    assert all(df[f"{PHASE}{plane}"] >= 0)
    assert all(df.loc[indx, f"{ERR}{PHASE}{plane}"] > 0)
    assert all(df.loc[indx, f"{ERR}{PHASE}{plane}"] < 0.5)
    assert all(np.abs(df[f"{DELTA}{PHASE}{plane}"] - df[f"{PHASE}{plane}"] + df[f"{PHASE}{plane}{MDL}"]) % 1 < 1e-12)
    assert all(np.abs(df[f"{ERR}{DELTA}{PHASE}{plane}"] - df[f"{ERR}{PHASE}{plane}"]) == 0)
    assert _gaussian_distribution_test(df.loc[indx, f"{PHASE}{plane}"], df.loc[indx, f"{PHASE}{plane}{MDL}"], df.loc[indx, f"{ERR}{PHASE}{plane}"])


def _test_disp(df, plane, relative_error):
    pass


def _test_norm_disp(df, plane, relative_error):
    pass


def _test_coupling(df, plane, relative_error):
    pass


def _gaussian_distribution_test(value, mean, std=None):
    """ Very rough test if between 63% and 70% of values are within one sigma
    and fullfills the Kolmogorov-Smirnov test. """
    if std is None:
        std = mean
    normalized = (value - mean) / std
    absdiff = np.abs(value-mean)

    ks = stats.kstest(normalized, "norm")

    ratio = sum(absdiff < std) / len(value)
    return (0.63 <= ratio <= 0.70) and all(absdiff > 0) and (ks.pvalue > 0.05)



# Fixtures ------

@pytest.fixture(params=["beam1", "beam2"])  # doesn't really add much, but tests are quick anyway
def both_beams_path(request):
    return beam_path(request.param)


@pytest.fixture
def beam1_path():
    return beam_path("beam1")


@pytest.fixture
def beam2_path():
    return beam_path("beam2")


def beam_path(beam):
    return INPUT_DIR / "models" / f"25cm_{beam}" / "twiss.dat"


# Helper -----

def _error_columns(df):
    return df.columns[df.columns.str.startswith(ERR)]   # includes ERRDELTA


def _model_columns(df):
    return df.columns[df.columns.str.endswith(MDL)]


def _delta_columns(df):
    return df.columns[df.columns.str.startswith(DELTA)]  # not ERRDELTA


def _full_model(path: Path):
    model = tfs.read(path, index=NAME)
    model = add_coupling_to_model(model)
    model[f"{NORM_DISP}X"] = model[f"{DISP}X"] / np.sqrt(model[f"{BETA}X"])
    return model
