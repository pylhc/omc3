import pickle
import shutil

import numpy as np
import pytest
import tfs

from omc3 import model
from omc3.global_correction import global_correction_entrypoint
from omc3.response_creator import create_response_entrypoint
from omc3.correction.constants import PHASE_ADV, BETA, DISP, NORM_DISP, F1001, F1010, TUNE

from pathlib import Path


# Paths ---
INPUTS = Path(__file__).parent / 'inputs'
CORRECTION_DIR = INPUTS / "correction"
MODEL_DIR = INPUTS / "models" / "inj_beam1"
FULLRESPONSE_PATH = CORRECTION_DIR / "Fullresponse_pandas"
GENERATED_MEASUREMENT_PATH = CORRECTION_DIR / "twiss_quadrupole_error.dat"
FULLRESPONSE_PATH_SKEW = CORRECTION_DIR / "Fullresponse_pandas_skew"

GENERATED_MEASUREMENT_PATH_SKEW = CORRECTION_DIR / "twiss_skew_quadrupole_error.dat"
ERROR_FILE_SKEW = CORRECTION_DIR / "skew_quadrupole_error.madx"

# Correction Input Parameters ---
MAX_ITER = 1
ACCEL_SETTINGS = dict(ats=True, beam=1, model_dir=MODEL_DIR, year="2018", accel="lhc", energy=0.45)
OPTICS_PARAMS = [f"{PHASE_ADV}X", f"{PHASE_ADV}Y",
                 f"{BETA}X", f"{BETA}Y",
                 f"{DISP}X", f"{NORM_DISP}X", f"{TUNE}"]
OPTICS_PARAMS_SKEW = [f"{F1001}R", f"{F1001}I", f"{F1010}R", f"{F1010}I"]

RMS_TOL_DICT_SKEW = {rdt: 0.001 for rdt in OPTICS_PARAMS_SKEW}
VARIABLE_CATEGORIES = ["MQY"]
VARIABLE_CATEGORIES_SKEW = ["MQSl"]
WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]
WEIGHTS_SKEW = [1.0, 1.0, 1.0, 1.0]
RMS_TOL_DICT = {
    f"{PHASE_ADV}X": 0.001,
    f"{PHASE_ADV}Y": 0.001,
    f"{BETA}X": 0.01,
    f"{BETA}Y": 0.01,
    f"{DISP}X": 0.0015,
    f"{NORM_DISP}X": 0.001,
    f"{TUNE}": 1e-05,
}
RMS_TOL_DICT_CORRECTION = {
    f"{PHASE_ADV}X": 3.0,
    f"{PHASE_ADV}Y": 3.0,
    f"{BETA}X": 25.0,
    f"{BETA}Y": 25.0,
    f"{DISP}X": 3.0,
    f"{DISP}Y": 1,
    f"{NORM_DISP}X": 2.0,
    f"{NORM_DISP}Y": 1.0,
    f"{TUNE}": 3.0,
    f"{F1001}R": 1.0,
    f"{F1001}I": 1.0,
    f"{F1010}R": 1.0,
    f"{F1010}I": 1.0,
}


@pytest.mark.basic
def test_global_correct_quad(tmp_path):
    _assert_global_correct(
        ACCEL_SETTINGS,
        tmp_path,
        OPTICS_PARAMS,
        VARIABLE_CATEGORIES,
        WEIGHTS,
        MAX_ITER,
        FULLRESPONSE_PATH,
        GENERATED_MEASUREMENT_PATH,
        RMS_TOL_DICT,
    )


@pytest.mark.basic
def test_global_correct_skew(tmp_path):
    _assert_global_correct(
        ACCEL_SETTINGS,
        tmp_path,
        OPTICS_PARAMS_SKEW,
        VARIABLE_CATEGORIES_SKEW,
        WEIGHTS_SKEW,
        MAX_ITER,
        FULLRESPONSE_PATH_SKEW,
        GENERATED_MEASUREMENT_PATH_SKEW,
        RMS_TOL_DICT_SKEW,
    )


@pytest.mark.basic
def test_fullresponse_madx_quad(tmp_path):
    _assert_response_madx(
        ACCEL_SETTINGS, tmp_path, VARIABLE_CATEGORIES, OPTICS_PARAMS, FULLRESPONSE_PATH
    )


@pytest.mark.basic
def test_fullresponse_madx_skew(tmp_path):
    _assert_response_madx(
        ACCEL_SETTINGS,
        tmp_path,
        VARIABLE_CATEGORIES_SKEW,
        OPTICS_PARAMS_SKEW,
        FULLRESPONSE_PATH_SKEW,
    )


@pytest.mark.basic
def test_fullresponse_twiss(tmp_path):
    _assert_response_twiss(
        ACCEL_SETTINGS,
        tmp_path,
        VARIABLE_CATEGORIES,
        FULLRESPONSE_PATH,
        RMS_TOL_DICT_CORRECTION,
    )


@pytest.mark.basic
def test_fullresponse_twiss_skew(tmp_path):
    _assert_response_twiss(
        ACCEL_SETTINGS,
        tmp_path,
        VARIABLE_CATEGORIES_SKEW,
        FULLRESPONSE_PATH_SKEW,
        RMS_TOL_DICT_CORRECTION,
    )


@pytest.mark.basic
def test_iteration_convergence(tmp_path):
    _assert_iteration_convergence(
        ACCEL_SETTINGS,
        tmp_path,
        OPTICS_PARAMS,
        VARIABLE_CATEGORIES,
        WEIGHTS,
        FULLRESPONSE_PATH,
        GENERATED_MEASUREMENT_PATH,
    )


def _get_rms_dict(
    accel_settings,
    correction_dir,
    optics_params,
    variable_categories,
    weights,
    max_iter,
    fullresponse_path,
    generated_measurement_path,
):
    model_dir = accel_settings["model_dir"]
    model_path = model_dir / "twiss.dat"
    corrected_path = correction_dir / f"twiss_{max_iter}.tfs"

    _convert_model_to_optics_measurement_tfs(model_path, generated_measurement_path, optics_params, correction_dir)
    global_correction_entrypoint(
        **accel_settings,
        meas_dir=correction_dir,
        variable_categories=variable_categories,
        fullresponse_path=fullresponse_path,
        optics_params=optics_params,
        output_dir=correction_dir,
        weights=weights,
        svd_cut=0.01,
        max_iter=max_iter,
    )

    # calculate RMS difference between generated measurement and correction
    gm_df = tfs.read(generated_measurement_path, index=NAME)
    cor_df = tfs.read(corrected_path, index=NAME)
    cor_df = cor_df.loc[gm_df.index, :]
    model_df = tfs.read(model_path, index=NAME)

    gm_df = _add_coupling(gm_df)
    cor_df = _add_coupling(cor_df)
    model_df = _add_coupling(model_df)

    name_l = gm_df.index[:-1:].to_numpy()
    name2_l = gm_df.index[1::].to_numpy()

    RMS_dict = {}
    for parameter in optics_params:
        if parameter.startswith(f"{PHASE_ADV}"):
            delta = (gm_df.loc[name2_l, parameter].to_numpy() - gm_df.loc[name_l, parameter].to_numpy()) - (
                cor_df.loc[name2_l, parameter].to_numpy() - cor_df.loc[name_l, parameter].to_numpy()
            )
        elif parameter.startswith(f"{BETA}"):
            delta = np.divide(
                gm_df.loc[:, parameter] - cor_df.loc[:, parameter], model_df.loc[:, parameter]
            ).to_numpy()
        elif parameter.startswith(f"{DISP}"):
            delta = (gm_df.loc[:, parameter] - cor_df.loc[:, parameter]).to_numpy()
        elif parameter.startswith("F"):
            delta = (gm_df.loc[:, parameter] - cor_df.loc[:, parameter]).to_numpy()
        elif parameter == f"{TUNE}":
            delta_Q1 = np.divide(gm_df[f"{TUNE}1"] - cor_df[f"{TUNE}1"], model_df[f"{TUNE}1"])
            delta_Q2 = np.divide(gm_df[f"{TUNE}2"] - cor_df[f"{TUNE}2"], model_df[f"{TUNE}2"])
            delta = np.array([delta_Q1, delta_Q2])
        elif parameter == f"{NORM_DISP}X":
            NDX_gm = np.divide(gm_df.loc[:, f"{DISP}X"], np.sqrt(gm_df.loc[:, f"{BETA}X"])).to_numpy()
            NDX_cor = np.divide(cor_df.loc[:, f"{DISP}X"], np.sqrt(cor_df.loc[:, f"{BETA}X"])).to_numpy()
            delta = NDX_gm - NDX_cor
        RMS_dict[parameter] = np.sqrt(np.mean((delta) ** 2))
    return RMS_dict


def _assert_response_madx(
    accel_settings,
    correction_dir,
    variable_categories,
    optics_params,
    comparison_fullresponse_path,
    delta_k=0.00002,
):
    fullresponse_path = correction_dir / "Fullresponse_pandas_omc3"
    create_response_entrypoint(
        **accel_settings,
        creator="madx",
        delta_k=delta_k,
        variable_categories=variable_categories,
        outfile_path=fullresponse_path,
    )

    with open(fullresponse_path, "rb") as fullresponse_file:
        fullresponse_data = pickle.load(fullresponse_file)

    with open(comparison_fullresponse_path, "rb") as comparison_fullresponse_file:
        comparison_fullresponse_data = pickle.load(comparison_fullresponse_file)

    # is_equal = True
    for key in fullresponse_data.keys():
        if key in optics_params:
            assert np.allclose(
                fullresponse_data[key][comparison_fullresponse_data[key].columns].to_numpy(),
                comparison_fullresponse_data[key].to_numpy(),
                rtol=1e-04,
                atol=1e-06,
            ), f"Fulresponse does not match for a key {key}"


def _assert_response_twiss(
    accel_settings,
    correction_dir,
    variable_categories,
    comparison_fullresponse_path,
    RMS_tol_dict,
    delta_k=0.00002,
):
    fullresponse_path = correction_dir / "Fullresponse_pandas_omc3"
    create_response_entrypoint(
        **accel_settings,
        creator="twiss",
        delta_k=delta_k,
        variable_categories=variable_categories,
        outfile_path=fullresponse_path,
    )

    with open(fullresponse_path, "rb") as fullresponse_file:
        fullresponse_data = pickle.load(fullresponse_file)

    with open(comparison_fullresponse_path, "rb") as comparison_fullresponse_file:
        comparison_fullresponse_data = pickle.load(comparison_fullresponse_file)

    # is_equal = True
    for key in fullresponse_data.keys():
        index = comparison_fullresponse_data[key].index
        columns = comparison_fullresponse_data[key].columns
        delta = (
            fullresponse_data[key].loc[index, columns].to_numpy()
            - comparison_fullresponse_data[key].to_numpy()
        )
        assert np.sqrt(np.mean(delta ** 2)) < RMS_tol_dict[key], (
            f"RMS difference between twiss and madx response is not within "
            f"tolerance {RMS_tol_dict[key]} for key {key}"
        )


def _assert_global_correct(
    accel_settings,
    correction_dir,
    optics_params,
    variable_categories,
    weights,
    max_iter,
    fullresponse_path,
    generated_measurement_path,
    RMS_tol_dict,
):
    RMS_dict = _get_rms_dict(
        accel_settings,
        correction_dir,
        optics_params,
        variable_categories,
        weights,
        max_iter,
        fullresponse_path,
        generated_measurement_path,
    )

    for key in RMS_dict.keys():
        assert RMS_dict[key] < RMS_tol_dict[key], f"RMS of {key} is not within tolerance"


def _assert_iteration_convergence(
    accel_settings,
    correction_dir,
    optics_params,
    variable_categories,
    weights,
    fullresponse_path,
    generated_measurement_path,
):
    RMS_dict1 = _get_rms_dict(
        accel_settings,
        correction_dir,
        optics_params,
        variable_categories,
        weights,
        MAX_ITER,
        fullresponse_path,
        generated_measurement_path,
    )

    RMS_dict2 = _get_rms_dict(
        accel_settings,
        correction_dir,
        optics_params,
        variable_categories,
        weights,
        MAX_ITER + 1,
        fullresponse_path,
        generated_measurement_path,
    )
    for key in RMS_dict1.keys():
        assert RMS_dict2[key] < RMS_dict1[key], f"RMS of {key} is got worse after repeated correction"


@pytest.fixture(scope="module")
def model_inj_beam1(tmp_path: Path):
    correction_inputs_path = INPUTS / "correction"
    macros_path = tmp_path / "macros"
    macros_path.mkdir()

    shutil.copytree(INPUTS / "models" / "inj_beam1", tmp_path)
    shutil.copytree(Path(model.__file__).parent / "madx_macros", macros_path)
    return tmp_path