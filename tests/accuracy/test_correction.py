# 4.remember beam 2:-)

import time
import matplotlib.pyplot as plt
import numpy as np
import tfs
import pickle
import pytest
import tempfile
import os


from omc3.response_creator import create_response_entrypoint
from omc3.global_correct import global_correction_entrypoint
from omc3.optics_measurements.constants import EXT, PHASE_NAME, DISPERSION_NAME, NORM_DISP_NAME, BETA_NAME
from omc3.correction.constants import ERR, DELTA
from omc3.correction import optics_class

# constants replace with constant.py files?
NAME = "NAME"


def _add_coupling(tfs_df):
    cpl = optics_class.get_coupling(tfs_df)
    tfs_df["F1001R"] = np.real(cpl["F1001"])
    tfs_df["F1001I"] = np.imag(cpl["F1001"])
    tfs_df["F1010R"] = np.real(cpl["F1010"])
    tfs_df["F1010I"] = np.imag(cpl["F1010"])
    return tfs_df


def _tfs_converter(twiss_model_file, twiss_file, optics_parameters, Output_dir):
    err_low = 0.01
    err_high = 0.02

    df_twiss = _add_coupling(tfs.read(twiss_file, NAME))
    df_model = _add_coupling(tfs.read(twiss_model_file, NAME))

    h_dict = {"Q1": df_twiss["Q1"], "Q2": df_twiss["Q2"]}
    for parameter in optics_parameters:
        col = parameter if "MU" not in parameter else f"PHASE{parameter[-1]}"
        if parameter.startswith("MU"):
            new = tfs.TfsDataFrame(index=df_twiss.index[:-1:])
            new[NAME] = df_twiss.index[:-1:]
            new[f"{NAME}2"] = df_twiss.index[1::]
            new[col] = df_twiss.loc[new[f"{NAME}2"], parameter].to_numpy() - \
                df_twiss.loc[new[NAME], parameter].to_numpy()
            mean_abs = np.mean(abs(new[col].to_numpy()))
            new[f"{ERR}{col}"] = np.random.uniform(
                err_low*mean_abs, err_high*mean_abs, len(df_twiss.index)-1)
            new[f"{ERR}{DELTA}{col}"] = new[f"{ERR}{col}"]
            new[f"{DELTA}{col}"] = new[col] - (df_model.loc[new[f"{NAME}2"], parameter].to_numpy() -
                                               df_model.loc[new[NAME], parameter].to_numpy())
            write_file = f"{PHASE_NAME}{parameter[-1].lower()}{EXT}"
        elif parameter.startswith("BET"):
            new = tfs.TfsDataFrame(index=df_twiss.index)
            new[NAME] = df_twiss.index
            new[col] = df_twiss.loc[:, parameter]

            mean_abs = np.mean(abs(new[col].to_numpy()))
            new[f"{ERR}{col}"] = np.random.uniform(
                err_low*mean_abs, err_high*mean_abs, len(df_twiss.index))
            new[f"{ERR}{DELTA}{col}"] = new[f"{ERR}{col}"] / \
                df_model.loc[:, parameter]
            new[f"{DELTA}{col}"] = (new[col] - df_model.loc[:,
                                                            parameter]) / df_model.loc[:, parameter]
            write_file = f"{BETA_NAME}{parameter[-1].lower()}{EXT}"
        elif parameter.startswith("D"):
            new = tfs.TfsDataFrame(index=df_twiss.index)
            new[NAME] = df_twiss.index
            new[col] = df_twiss.loc[:, parameter]

            mean_abs = np.mean(abs(new[col].to_numpy()))
            new[f"{ERR}{col}"] = np.random.uniform(
                err_low*mean_abs, err_high*mean_abs, len(df_twiss.index))
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
                err_low*mean_abs_Re, err_high*mean_abs_Re, len(df_twiss.index))
            new[f"{ERR}{DELTA}{Re}"] = new[f"{ERR}{Re}"]
            new[f"{DELTA}{Re}"] = new[Re] - df_model.loc[:, Re]

            new[Im] = df_twiss.loc[:, Im]

            mean_abs_Im = np.mean(abs(new[Im].to_numpy()))
            new[f"{ERR}{Im}"] = np.random.uniform(
                err_low*mean_abs_Im, err_high*mean_abs_Im, len(df_twiss.index))
            new[f"{ERR}{DELTA}{Im}"] = new[f"{ERR}{Im}"]
            new[f"{DELTA}{Im}"] = new[Im] - df_model.loc[:, Im]

            write_file = f"{parameter[:-1]}{EXT}"

        elif parameter == "NDX":
            new = tfs.TfsDataFrame(index=df_twiss.index)
            new[NAME] = df_twiss.index
            new[col] = np.divide(df_twiss.loc[:, "DX"],
                                 np.sqrt(df_twiss.loc[:, "BETX"]))

            mean_abs = np.mean(abs(new[col].to_numpy()))
            new[f"{ERR}{col}"] = np.random.uniform(
                err_low*mean_abs, err_high*mean_abs, len(df_twiss.index))
            new[f"{ERR}{DELTA}{col}"] = new[f"{ERR}{col}"]
            new[f"{DELTA}{col}"] = new[col] - \
                np.divide(df_model.loc[:, "DX"],
                          np.sqrt(df_model.loc[:, "BETX"]))
            write_file = f"{NORM_DISP_NAME}{parameter[-1].lower()}{EXT}"
            
        tfs.write(f"{Output_dir}{write_file}", new,
                  headers_dict=h_dict, save_index="index_column")


def _assert_response_madx(accel_settings, correction_dir, variable_categories, comparison_fullresponse_path, delta_k=0.00002):
    fullresponse_path = correction_dir + "Fullresponse_pandas_omc3"

    create_response_entrypoint(**accel_settings,
                               creator="madx",
                               delta_k=delta_k,
                               variable_categories=variable_categories,
                               outfile_path=fullresponse_path,
                               )

    with open(fullresponse_path, "rb") as fullresponse_file:
        fullresponse_data = pickle.load(fullresponse_file)

    with open(comparison_fullresponse_path, "rb") as comparison_fullresponse_file:
        comparison_fullresponse_data = pickle.load(
            comparison_fullresponse_file)

    is_equal = True
    for key in fullresponse_data.keys():
        assert np.all(np.isclose(fullresponse_data[key][comparison_fullresponse_data[key].columns].to_numpy(
        ), comparison_fullresponse_data[key].to_numpy()),atol=1e-07), f"Fulresponse does not match for a key {key}"


def _assert_response_twiss(accel_settings, correction_dir, variable_categories, comparison_fullresponse_path, RMS_tol_dict, delta_k=0.00002):
    fullresponse_path = correction_dir + "Fullresponse_pandas_omc3"

    create_response_entrypoint(**accel_settings,
                               creator="twiss",
                               delta_k=delta_k,
                               variable_categories=variable_categories,
                               outfile_path=fullresponse_path,
                               )

    with open(fullresponse_path, "rb") as fullresponse_file:
        fullresponse_data = pickle.load(fullresponse_file)

    with open(comparison_fullresponse_path, "rb") as comparison_fullresponse_file:
        comparison_fullresponse_data = pickle.load(
            comparison_fullresponse_file)

    is_equal = True
    for key in fullresponse_data.keys():
        index = comparison_fullresponse_data[key].index
        columns = comparison_fullresponse_data[key].columns
        delta = fullresponse_data[key].loc[index, columns].to_numpy(
        ) - comparison_fullresponse_data[key].to_numpy()
        assert np.sqrt(np.mean(
            delta**2)) < RMS_tol_dict[key], f"RMS difference between twiss and madx responseis not within tolerance {RMS_tol_dict[key]} for key {key}"


def _assert_global_correct(accel_settings, correction_dir, optics_params, variable_categories, weights, fullresponse_path, generated_measurement_path, RMS_tol_dict):
    model_dir = accel_settings["model_dir"]
    model_path = model_dir + "twiss.dat"
    corrected_path = correction_dir + "twiss_1.tfs"

    _tfs_converter(model_path, generated_measurement_path,
                   optics_params, correction_dir)

    global_correction_entrypoint(**accel_settings,
                                 meas_dir=correction_dir,
                                 variable_categories=variable_categories,
                                 fullresponse_path=fullresponse_path,
                                 optics_params=optics_params,
                                 output_dir=correction_dir,
                                 weights=weights,
                                 svd_cut=0.01,
                                 max_iter=1)

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
        if parameter.startswith("MU"):
            delta = (gm_df.loc[name2_l, parameter].to_numpy() - gm_df.loc[name_l, parameter].to_numpy()) - (cor_df.loc[name2_l, parameter].to_numpy() -
                                                                                                            cor_df.loc[name_l, parameter].to_numpy())
        elif parameter.startswith("BET"):
            delta = np.divide(
                gm_df.loc[:, parameter] - cor_df.loc[:, parameter], model_df.loc[:, parameter]).to_numpy()
        elif parameter.startswith("D"):
            delta = (gm_df.loc[:, parameter] -
                     cor_df.loc[:, parameter]).to_numpy()
        elif parameter.startswith("F"):
            delta = (gm_df.loc[:, parameter] -
                     cor_df.loc[:, parameter]).to_numpy()
        elif parameter == "Q":
            delta_Q1 = np.divide(gm_df["Q1"] - cor_df["Q1"], model_df["Q1"])
            delta_Q2 = np.divide(gm_df["Q2"] - cor_df["Q2"], model_df["Q2"])
            delta = np.array([delta_Q1, delta_Q2])
        elif parameter == "NDX":
            NDX_gm = np.divide(gm_df.loc[:, "DX"],
                               np.sqrt(gm_df.loc[:, "BETX"])).to_numpy()
            NDX_cor = np.divide(
                cor_df.loc[:, "DX"], np.sqrt(cor_df.loc[:, "BETX"])).to_numpy()
            delta = NDX_gm - NDX_cor

        RMS_dict[parameter] = np.sqrt(np.mean((delta)**2))

    for key in RMS_dict.keys():
        assert RMS_dict[key] < RMS_tol_dict[key], f"RMS of {key} is not within tolerance"



CORRECTION_DIR = os.path.join(os.path.dirname(__file__),'..','inputs','correction') + '/'
FULLRESPONSE_PATH = CORRECTION_DIR + "Fullresponse_pandas"
MODEL_DIR = CORRECTION_DIR + "model_dir/"
GENERATED_MEASUREMENT_PATH = CORRECTION_DIR + "twiss_quadrupole_error.dat"
ACCEL_SETTINGS = dict(beam=1, model_dir=MODEL_DIR,
                      year="2018", accel="lhc", energy=0.45)
OPTICS_PARAMS = ["MUX", "MUY", "BETX", "BETY", "DX", "NDX", "Q"]
VARIABLE_CATEGORIES = ["MQY"]
WEIGHTS = [1., 1., 1., 1., 1., 1., 10.]
RMS_TOL_DICT = {"MUX": 0.001, "MUY": 0.001, "BETX": 0.01, "BETY": 0.01,
                "DX": 0.001, "NDX": 0.001, "Q": 1e-05}


RMS_TOL_DICT_CORRECTION = {"Q": 3., "MUX": 3., "MUY": 3., "BETX": 15., "BETY": 15.,
                           "DX": 2., "DY": 1, "NDX": 2., "NDY": 1., "F1001R": 1., "F1001I": 1., "F1010R": 1., "F1010I": 1.}

GENERATED_MEASUREMENT_PATH_SKEW = CORRECTION_DIR + \
    "twiss_skew_quadrupole_error.dat"
ERROR_FILE_SKEW = CORRECTION_DIR + "skew_quadrupole_error.madx"
OPTICS_PARAMS_SKEW = ["F1001R", "F1001I",
                      "F1010R", "F1010I"]
VARIABLE_CATEGORIES_SKEW = ["MQSl"]
WEIGHTS_SKEW = [1., 1., 1., 1.]
FULLRESPONSE_PATH_SKEW = CORRECTION_DIR + "Fullresponse_pandas_skew"
RMS_TOL_DICT_SKEW = {"F1001R": 0.001, "F1001I": 0.001,
                     "F1010R": 0.001, "F1010I": 0.001}



@pytest.mark.basic
def test_global_correct_quad():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_global_correct(ACCEL_SETTINGS, temp_dir,
                            OPTICS_PARAMS, VARIABLE_CATEGORIES, WEIGHTS, FULLRESPONSE_PATH, GENERATED_MEASUREMENT_PATH, RMS_TOL_DICT)


@pytest.mark.basic
def test_global_correct_skew():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_global_correct(ACCEL_SETTINGS, temp_dir,
                            OPTICS_PARAMS_SKEW, VARIABLE_CATEGORIES_SKEW, WEIGHTS_SKEW, FULLRESPONSE_PATH_SKEW, GENERATED_MEASUREMENT_PATH_SKEW, RMS_TOL_DICT_SKEW)


@pytest.mark.basic
def test_fullresponse_madx_quad():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_response_madx(ACCEL_SETTINGS, temp_dir,
                           VARIABLE_CATEGORIES, FULLRESPONSE_PATH)


@pytest.mark.basic
def test_fullresponse_madx_skew():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_response_madx(ACCEL_SETTINGS, temp_dir,
                           VARIABLE_CATEGORIES_SKEW, FULLRESPONSE_PATH_SKEW)


@pytest.mark.basic
def test_fullresponse_twiss():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_response_twiss(ACCEL_SETTINGS, temp_dir,
                            VARIABLE_CATEGORIES, FULLRESPONSE_PATH, RMS_TOL_DICT_CORRECTION)


@pytest.mark.basic
def test_fullresponse_twiss_skew():
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = temp + "/"
        _assert_response_twiss(ACCEL_SETTINGS, temp_dir,
                            VARIABLE_CATEGORIES_SKEW, FULLRESPONSE_PATH_SKEW, RMS_TOL_DICT_CORRECTION)
