import os
import pickle
import tempfile

import numpy as np
import pytest
import tfs
from optics_functions.coupling import coupling_via_cmatrix

from omc3.correction.constants import DELTA, ERR, PHASE_ADV, BETA, DISP, PHASE, NORM_DISP, TUNE, F1001, F1010
from omc3.global_correction import global_correction_entrypoint
from omc3.optics_measurements.constants import BETA_NAME, DISPERSION_NAME, EXT, NORM_DISP_NAME, PHASE_NAME
from omc3.response_creator import create_response_entrypoint

NAME = "NAME"
MAX_ITER = 1
CORRECTION_DIR = os.path.join(os.path.dirname(__file__), "..", "inputs", "correction") + "/"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "inputs", "models", "inj_beam1") + "/"
FULLRESPONSE_PATH = CORRECTION_DIR + "Fullresponse_pandas"
GENERATED_MEASUREMENT_PATH = CORRECTION_DIR + "twiss_quadrupole_error.dat"
ACCEL_SETTINGS = dict(ats=True, beam=1, model_dir=MODEL_DIR, year="2018", accel="lhc", energy=0.45)
OPTICS_PARAMS = [f"{PHASE_ADV}X", f"{PHASE_ADV}Y", f"{BETA}X", f"{BETA}Y", f"{DISP}X", f"{NORM_DISP}X",
                 f"{TUNE}"]
VARIABLE_CATEGORIES = ["MQY"]
WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]
GENERATED_MEASUREMENT_PATH_SKEW = CORRECTION_DIR + "twiss_skew_quadrupole_error.dat"
ERROR_FILE_SKEW = CORRECTION_DIR + "skew_quadrupole_error.madx"
OPTICS_PARAMS_SKEW = [f"{F1001}R", f"{F1001}I", f"{F1010}R", f"{F1010}I"]
VARIABLE_CATEGORIES_SKEW = ["MQSl"]
WEIGHTS_SKEW = [1.0, 1.0, 1.0, 1.0]
FULLRESPONSE_PATH_SKEW = CORRECTION_DIR + "Fullresponse_pandas_skew"
RMS_TOL_DICT_SKEW = {rdt: 0.001 for rdt in OPTICS_PARAMS_SKEW}
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


def _add_coupling(tfs_df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """
    Computes the coupling RDTs from the input TfsDataFrame and returns a copy of said TfsDataFrame with
    columns for the real and imaginary parts of the computed coupling RDTs.

    Args:
        tfs_df (tfs.TfsDataFrame): Twiss dataframe.

    Returns:
        A TfsDataFrame with the added columns.
    """
    result_tfs_df = tfs_df.copy()
    coupling_rdts_df = coupling_via_cmatrix(result_tfs_df)
    result_tfs_df[f"{F1001}R"] = np.real(coupling_rdts_df[f"{F1001}"]).astype(np.float64)
    result_tfs_df[f"{F1001}I"] = np.imag(coupling_rdts_df[f"{F1001}"]).astype(np.float64)
    result_tfs_df[f"{F1010}R"] = np.real(coupling_rdts_df[f"{F1010}"]).astype(np.float64)
    result_tfs_df[f"{F1010}I"] = np.imag(coupling_rdts_df[f"{F1010}"]).astype(np.float64)
    return result_tfs_df


def _tfs_converter(twiss_model_file, twiss_file, optics_parameters, Output_dir):
    """
    Takes a twiss file and writes the parameters in optics_parameters to Output_dir in the format
    global_correction_entrypoint uses (same format you would get from hole_in_one).

    Args:
        twiss_model_file:
        twiss_file:
        optics_parameters:
        Output_dir:

    Returns:

    """
    err_low = 0.01
    err_high = 0.02

    df_twiss = _add_coupling(tfs.read(twiss_file, NAME))
    df_model = _add_coupling(tfs.read(twiss_model_file, NAME))

    h_dict = {f"{TUNE}1": df_twiss[f"{TUNE}1"], f"{TUNE}2": df_twiss[f"{TUNE}2"]}
    for parameter in optics_parameters:
        col = parameter if f"{PHASE_ADV}" not in parameter else f"{PHASE}{parameter[-1]}"
        if parameter.startswith(f"{PHASE_ADV}"):
            new = tfs.TfsDataFrame(index=df_twiss.index[:-1:])  # ???????
            new[NAME] = df_twiss.index[:-1:]
            new[f"{NAME}2"] = df_twiss.index[1::]
            new[col] = (
                df_twiss.loc[new[f"{NAME}2"], parameter].to_numpy()
                - df_twiss.loc[new[NAME], parameter].to_numpy()
            )

            mean_abs = np.mean(abs(new[col].to_numpy()))
            new[f"{ERR}{col}"] = np.random.uniform(
                err_low * mean_abs, err_high * mean_abs, len(df_twiss.index) - 1
            )
            new[f"{ERR}{DELTA}{col}"] = new[f"{ERR}{col}"]
            new[f"{DELTA}{col}"] = new[col] - (
                df_model.loc[new[f"{NAME}2"], parameter].to_numpy()
                - df_model.loc[new[NAME], parameter].to_numpy()
            )
            write_file = f"{PHASE_NAME}{parameter[-1].lower()}{EXT}"

        elif parameter.startswith(f"{BETA}"):
            new = tfs.TfsDataFrame(index=df_twiss.index)
            new[NAME] = df_twiss.index
            new[col] = df_twiss.loc[:, parameter]

            mean_abs = np.mean(abs(new[col].to_numpy()))
            new[f"{ERR}{col}"] = np.random.uniform(
                err_low * mean_abs, err_high * mean_abs, len(df_twiss.index)
            )
            new[f"{ERR}{DELTA}{col}"] = new[f"{ERR}{col}"] / df_model.loc[:, parameter]
            new[f"{DELTA}{col}"] = (new[col] - df_model.loc[:, parameter]) / df_model.loc[:, parameter]
            write_file = f"{BETA_NAME}{parameter[-1].lower()}{EXT}"

        elif parameter.startswith(f"{DISP}"):
            new = tfs.TfsDataFrame(index=df_twiss.index)
            new[NAME] = df_twiss.index
            new[col] = df_twiss.loc[:, parameter]

            mean_abs = np.mean(abs(new[col].to_numpy()))
            new[f"{ERR}{col}"] = np.random.uniform(
                err_low * mean_abs, err_high * mean_abs, len(df_twiss.index)
            )
            new[f"{ERR}{DELTA}{col}"] = new[f"{ERR}{col}"]
            new[f"{DELTA}{col}"] = new[col] - df_model.loc[:, parameter]
            write_file = f"{DISPERSION_NAME}{parameter[-1].lower()}{EXT}"

        elif parameter.startswith("F") and parameter.endswith("R"):
            Re = parameter
            Im = parameter[:-1] + "I"
            new = tfs.TfsDataFrame(index=df_twiss.index)
            new[NAME] = df_twiss.index
            new[Re] = df_twiss.loc[:, Re]

            mean_abs_Re = np.mean(abs(new[Re].to_numpy()))
            new[f"{ERR}{Re}"] = np.random.uniform(
                err_low * mean_abs_Re, err_high * mean_abs_Re, len(df_twiss.index)
            )
            new[f"{ERR}{DELTA}{Re}"] = new[f"{ERR}{Re}"]
            new[f"{DELTA}{Re}"] = new[Re] - df_model.loc[:, Re]

            new[Im] = df_twiss.loc[:, Im]
            mean_abs_Im = np.mean(abs(new[Im].to_numpy()))
            new[f"{ERR}{Im}"] = np.random.uniform(
                err_low * mean_abs_Im, err_high * mean_abs_Im, len(df_twiss.index)
            )
            new[f"{ERR}{DELTA}{Im}"] = new[f"{ERR}{Im}"]
            new[f"{DELTA}{Im}"] = new[Im] - df_model.loc[:, Im]
            write_file = f"{parameter[:-1]}{EXT}"

        elif parameter == f"{NORM_DISP}X":
            new = tfs.TfsDataFrame(index=df_twiss.index)
            new[NAME] = df_twiss.index
            new[col] = np.divide(df_twiss.loc[:, f"{DISP}X"], np.sqrt(df_twiss.loc[:, f"{BETA}X"]))

            mean_abs = np.mean(abs(new[col].to_numpy()))
            new[f"{ERR}{col}"] = np.random.uniform(
                err_low * mean_abs, err_high * mean_abs, len(df_twiss.index)
            )
            new[f"{ERR}{DELTA}{col}"] = new[f"{ERR}{col}"]
            new[f"{DELTA}{col}"] = new[col] - np.divide(
                df_model.loc[:, f"{DISP}X"], np.sqrt(df_model.loc[:, f"{BETA}X"])
            )
            write_file = f"{NORM_DISP_NAME}{parameter[-1].lower()}{EXT}"

        tfs.write(f"{Output_dir}{write_file}", new, headers_dict=h_dict, save_index="index_column")


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
    model_path = model_dir + "twiss.dat"
    corrected_path = correction_dir + f"twiss_{max_iter}.tfs"

    _tfs_converter(model_path, generated_measurement_path, optics_params, correction_dir)
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
    fullresponse_path = correction_dir + "Fullresponse_pandas_omc3"
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
    fullresponse_path = correction_dir + "Fullresponse_pandas_omc3"
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


@pytest.mark.basic
def test_global_correct_quad():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_global_correct(
            ACCEL_SETTINGS,
            temp_dir,
            OPTICS_PARAMS,
            VARIABLE_CATEGORIES,
            WEIGHTS,
            MAX_ITER,
            FULLRESPONSE_PATH,
            GENERATED_MEASUREMENT_PATH,
            RMS_TOL_DICT,
        )


@pytest.mark.basic
def test_global_correct_skew():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_global_correct(
            ACCEL_SETTINGS,
            temp_dir,
            OPTICS_PARAMS_SKEW,
            VARIABLE_CATEGORIES_SKEW,
            WEIGHTS_SKEW,
            MAX_ITER,
            FULLRESPONSE_PATH_SKEW,
            GENERATED_MEASUREMENT_PATH_SKEW,
            RMS_TOL_DICT_SKEW,
        )


@pytest.mark.basic
def test_fullresponse_madx_quad():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_response_madx(
            ACCEL_SETTINGS, temp_dir, VARIABLE_CATEGORIES, OPTICS_PARAMS, FULLRESPONSE_PATH
        )


@pytest.mark.basic
def test_fullresponse_madx_skew():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_response_madx(
            ACCEL_SETTINGS,
            temp_dir,
            VARIABLE_CATEGORIES_SKEW,
            OPTICS_PARAMS_SKEW,
            FULLRESPONSE_PATH_SKEW,
        )


@pytest.mark.basic
def test_fullresponse_twiss():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_response_twiss(
            ACCEL_SETTINGS,
            temp_dir,
            VARIABLE_CATEGORIES,
            FULLRESPONSE_PATH,
            RMS_TOL_DICT_CORRECTION,
        )


@pytest.mark.basic
def test_fullresponse_twiss_skew():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_response_twiss(
            ACCEL_SETTINGS,
            temp_dir,
            VARIABLE_CATEGORIES_SKEW,
            FULLRESPONSE_PATH_SKEW,
            RMS_TOL_DICT_CORRECTION,
        )


@pytest.mark.basic
def test_itteration_convergence():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_iteration_convergence(
            ACCEL_SETTINGS,
            temp_dir,
            OPTICS_PARAMS,
            VARIABLE_CATEGORIES,
            WEIGHTS,
            FULLRESPONSE_PATH,
            GENERATED_MEASUREMENT_PATH,
        )
