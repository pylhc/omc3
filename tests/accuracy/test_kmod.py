import shutil
from pathlib import Path

import numpy as np
import pytest
import tfs

from omc3.definitions.constants import BETA, ERR, EXT, PLANES, STAR
from omc3.run_kmod import INSTRUMENTS_FILE_NAME, RESULTS_FILE_NAME, analyse_kmod

LIMITS = {"Accuracy": 1e-5, "Meas Accuracy": 0.05, "Num Precision": 1e-15, "Meas Precision": 0.1}


@pytest.mark.extended
def test_kmod_phase_simulation_ip5b1(tmp_path, _kmod_inputs_path):
    analyse_kmod(
        betastar_and_waist=[19.2, 0.0],
        working_directory=_kmod_inputs_path,
        beam=1,
        simulation=True,
        no_sig_digits=True,
        no_plots=False,
        interaction_point="ip5",
        cminus=0.0,
        misalignment=0.0,
        errorK=0.0,
        errorL=0.0,
        tune_uncertainty=0.0e-5,
        phase_weight=0.5,
        model_dir=_kmod_inputs_path,
        outputdir=tmp_path,
    )
    results = tfs.read(tmp_path / "ip5B1" / f"{RESULTS_FILE_NAME}{EXT}")
    beta_twiss = {"X": 19.2, "Y": 19.2}

    for plane in PLANES:
        beta_sim = beta_twiss[plane]
        beta_meas = results[f"{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_meas - beta_sim)) / beta_sim < LIMITS["Accuracy"]
        beta_err_meas = results[f"{ERR}{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_err_meas)) < LIMITS["Num Precision"]


@pytest.mark.extended
def test_kmod_phase_measured_ip5b1(tmp_path, _kmod_inputs_path):
    analyse_kmod(
        betastar_and_waist=[19.2, 0.0],
        working_directory=_kmod_inputs_path,
        beam=1,
        simulation=True,
        no_sig_digits=True,
        no_plots=False,
        interaction_point="ip5",
        cminus=0.0,
        misalignment=0.0,
        errorK=0.0,
        errorL=0.0,
        tune_uncertainty=0.0e-5,
        phase_weight=0.5,
        measurement_dir=_kmod_inputs_path,
        outputdir=tmp_path,
    )
    results = tfs.read(tmp_path / "ip5B1" / f"{RESULTS_FILE_NAME}{EXT}")
    beta_twiss = {"X": 19.2, "Y": 19.2}

    for plane in PLANES:
        beta_sim = beta_twiss[plane]
        beta_meas = results[f"{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_meas - beta_sim)) / beta_sim < LIMITS["Accuracy"]
        beta_err_meas = results[f"{ERR}{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_err_meas)) < LIMITS["Num Precision"]


@pytest.mark.extended
def test_kmod_simulation_ip1b1(tmp_path, _kmod_inputs_path):
    analyse_kmod(
        betastar_and_waist=[0.25, 0.0],
        working_directory=_kmod_inputs_path,
        beam=1,
        simulation=True,
        no_sig_digits=True,
        no_plots=False,
        interaction_point="ip1",
        cminus=0.0,
        misalignment=0.0,
        errorK=0.0,
        errorL=0.0,
        tune_uncertainty=0.0e-5,
        outputdir=tmp_path,
    )
    results = tfs.read(tmp_path / "ip1B1" / f"{RESULTS_FILE_NAME}{EXT}")
    beta_twiss = {"X": 0.25, "Y": 0.25}

    for plane in PLANES:
        beta_sim = beta_twiss[plane]
        beta_meas = results[f"{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_meas - beta_sim)) / beta_sim < LIMITS["Accuracy"]
        beta_err_meas = results[f"{ERR}{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_err_meas)) < LIMITS["Num Precision"]


@pytest.mark.extended
def test_kmod_simulation_ip1b2(tmp_path, _kmod_inputs_path):
    analyse_kmod(
        betastar_and_waist=[0.25, 0.25, 0.0],
        working_directory=_kmod_inputs_path,
        beam=2,
        simulation=True,
        no_sig_digits=True,
        no_plots=False,
        no_autoclean=True,
        interaction_point="ip1",
        cminus=0.0,
        misalignment=0.0,
        errorK=0.0,
        errorL=0.0,
        tune_uncertainty=0.0e-5,
        outputdir=tmp_path,
    )
    results = tfs.read(tmp_path / "ip1B2" / f"{RESULTS_FILE_NAME}{EXT}")
    beta_twiss = {"X": 0.25, "Y": 0.25}

    for plane in PLANES:
        beta_sim = beta_twiss[plane]
        beta_meas = results[f"{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_meas - beta_sim)) / beta_sim < LIMITS["Accuracy"]
        beta_err_meas = results[f"{ERR}{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_err_meas)) < LIMITS["Num Precision"]


@pytest.mark.extended
def test_kmod_meas_ip1b1(tmp_path, _kmod_inputs_path):
    analyse_kmod(
        betastar_and_waist=[0.44, 0.44, 0.0, 0.0],
        working_directory=_kmod_inputs_path,
        beam=1,
        simulation=False,
        no_sig_digits=True,
        no_plots=False,
        interaction_point="ip1",
        cminus=0.0,
        misalignment=0.0,
        errorK=0.0,
        errorL=0.0,
        tune_uncertainty=2.5e-5,
        outputdir=tmp_path,
    )
    results = tfs.read(tmp_path / "ip1B1" / f"{RESULTS_FILE_NAME}{EXT}")
    beta_prev = {"X": 0.45, "Y": 0.43}
    for plane in PLANES:

        beta_meas = results[f"{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_meas - beta_prev[plane])) / beta_prev[plane] < LIMITS["Meas Accuracy"]
        beta_err_meas = results[f"{ERR}{BETA}{STAR}{plane}"].loc[0]
        assert (beta_err_meas / beta_meas) < LIMITS["Meas Precision"]


@pytest.mark.extended
def test_kmod_meas_ip1b2(tmp_path, _kmod_inputs_path):
    analyse_kmod(
        betastar_and_waist=[0.44, 0.0],
        working_directory=_kmod_inputs_path,
        beam=2,
        simulation=False,
        no_sig_digits=True,
        no_plots=False,
        interaction_point="ip1",
        cminus=0.0,
        misalignment=0.0,
        errorK=0.0,
        errorL=0.0,
        tune_uncertainty=2.5e-5,
        outputdir=tmp_path,
    )
    results = tfs.read(tmp_path / "ip1B2" / f"{RESULTS_FILE_NAME}{EXT}")
    beta_prev = {"X": 0.387, "Y": 0.410}
    for plane in PLANES:

        beta_meas = results[f"{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_meas - beta_prev[plane])) / beta_prev[plane] < LIMITS["Meas Accuracy"]
        beta_err_meas = results[f"{ERR}{BETA}{STAR}{plane}"].loc[0]
        assert (beta_err_meas / beta_meas) < LIMITS["Meas Precision"]


@pytest.mark.extended
def test_kmod_meas_ip4b1(tmp_path, _kmod_inputs_path):
    analyse_kmod(
        betastar_and_waist=[200.0, -100.0],
        working_directory=_kmod_inputs_path,
        beam=1,
        simulation=False,
        no_sig_digits=True,
        no_plots=False,
        circuits=["RQ6.R4B1", "RQ7.R4B1"],
        cminus=0.0,
        misalignment=0.0,
        errorK=0.0,
        errorL=0.0,
        tune_uncertainty=0.5e-5,
        outputdir=tmp_path,
    )
    results = tfs.read(
        tmp_path / "MQY.6R4.B1-MQM.7R4.B1" / f"{INSTRUMENTS_FILE_NAME}{EXT}", index="NAME"
    )

    original = {
        "BPMCS.7R4.B1": (17.5074335336, 157.760070696),
        "BPM.7R4.B1": (17.6430538896, 157.972911909),
        "BQSH.7R4.B1": (455.457631868, 124.586686684),
        "BPLH.7R4.B1": (423.68951095, 123.578577484),
    }

    for inst in results.index:
        beta_x, beta_y = original[inst]
        betas = dict(X=beta_x, Y=beta_y)
        for plane in PLANES:
            beta_meas = results[f"{BETA}{plane}"].loc[inst]
            assert (np.abs(beta_meas - betas[plane])) / betas[plane] < LIMITS["Meas Accuracy"]
            beta_err_meas = results[f"{ERR}{BETA}{plane}"].loc[inst]
            assert (beta_err_meas / beta_meas) < LIMITS["Meas Precision"]


@pytest.mark.extended
def test_kmod_simulation_ip4b1(tmp_path, _kmod_inputs_path):
    analyse_kmod(
        betastar_and_waist=[200.0, -100.0],
        working_directory=_kmod_inputs_path,
        beam=1,
        simulation=True,
        no_sig_digits=True,
        no_plots=False,
        circuits=["RQ6.R4B1", "RQ7.R4B1"],
        cminus=0.0,
        misalignment=0.0,
        errorK=0.0,
        errorL=0.0,
        tune_uncertainty=0.5e-5,
        outputdir=tmp_path,
    )
    results = tfs.read(
        tmp_path / "MQY.6R4.B1-MQM.7R4.B1" / f"{INSTRUMENTS_FILE_NAME}{EXT}", index="NAME"
    )

    original = {
        "BPMCS.7R4.B1": (3.64208332528655e01, 9.46041254954643e01),
        "BPM.7R4.B1": (3.61317067929723e01, 9.48945562104017e01),
        "BQSH.7R4.B1": (5.07121388372368e02, 9.07140610660815e01),
        "BPLH.7R4.B1": (4.79632975072045e02, 8.65331699893341e01),
    }
    for inst in results.index:
        beta_x, beta_y = original[inst]
        betas = dict(X=beta_x, Y=beta_y)
        for plane in PLANES:
            beta_meas = results[f"{BETA}{plane}"].loc[inst]
            assert (np.abs(beta_meas - betas[plane])) / betas[plane] < LIMITS["Meas Accuracy"]
            beta_err_meas = results[f"{ERR}{BETA}{plane}"].loc[inst]
            assert (beta_err_meas / beta_meas) < LIMITS["Meas Precision"]


@pytest.mark.extended
def test_kmod_meas_ip4b2(tmp_path, _kmod_inputs_path):
    analyse_kmod(
        betastar_and_waist=[200.0, -100.0],
        working_directory=_kmod_inputs_path,
        beam=2,
        simulation=False,
        no_sig_digits=True,
        no_plots=False,
        circuits=["RQ7.L4B2", "RQ6.L4B2"],
        cminus=0.0,
        misalignment=0.0,
        errorK=0.0,
        errorL=0.0,
        tune_uncertainty=0.5e-5,
        outputdir=tmp_path,
    )
    results = tfs.read(
        tmp_path / "MQM.7L4.B2-MQY.6L4.B2" / f"{INSTRUMENTS_FILE_NAME}{EXT}", index="NAME"
    )

    original = {
        "BPMYA.6L4.B2": (456.789268726, 149.073169556),
        "BGVCA.B7L4.B2": (119.359634764, 152.116072289),
        "BPLH.B7L4.B2": (434.440558008, 148.460642194),
        "BPLH.A7L4.B2": (441.781928671, 148.654814221),
    }

    for inst in results.index:
        beta_x, beta_y = original[inst]
        betas = dict(X=beta_x, Y=beta_y)
        for plane in PLANES:
            beta_meas = results[f"{BETA}{plane}"].loc[inst]
            assert (np.abs(beta_meas - betas[plane])) / betas[plane] < LIMITS["Meas Accuracy"]
            beta_err_meas = results[f"{ERR}{BETA}{plane}"].loc[inst]
            assert (beta_err_meas / beta_meas) < LIMITS["Meas Precision"]


@pytest.mark.extended
def test_kmod_outputdir_default(tmp_path, _kmod_inputs_path):
    """
    Copy input files to tmp_path, use it as working_directory and assert the results go there when
    no outputdir is specified.
    """
    [shutil.copy(kmod_input, tmp_path) for kmod_input in _kmod_inputs_path.glob("*L4B2*")]
    analyse_kmod(
        betastar_and_waist=[200.0, -100.0],
        working_directory=tmp_path,
        beam=2,
        simulation=False,
        no_sig_digits=True,
        no_plots=False,
        circuits=["RQ7.L4B2", "RQ6.L4B2"],
        cminus=0.0,
        misalignment=0.0,
        errorK=0.0,
        errorL=0.0,
        tune_uncertainty=0.5e-5,
    )
    assert (tmp_path / "MQM.7L4.B2-MQY.6L4.B2").exists()


@pytest.fixture()
def _kmod_inputs_path() -> Path:
    return Path(__file__).parent.parent / "inputs" / "kmod"
