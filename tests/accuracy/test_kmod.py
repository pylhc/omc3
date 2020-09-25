from os.path import dirname, join, isdir, pardir

import tfs
import shutil
import numpy as np
import pytest
from omc3.run_kmod import analyse_kmod
from omc3.kmod.constants import BETA, ERR, STAR
from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import EXT
from omc3.run_kmod import RESULTS_FILE_NAME, INSTRUMENTS_FILE_NAME, LSA_FILE_NAME
CURRENT_DIR = dirname(__file__)
LIMITS = {'Accuracy': 1E-5,
          'Meas Accuracy': 0.05,
          'Num Precision': 1E-15,
          'Meas Precision': 0.1}


@pytest.mark.extended
def test_kmod_phase_simulation_ip5b1(_workdir_path):
    analyse_kmod(betastar_and_waist=[19.2, 0.0],
                 working_directory=_workdir_path,
                 beam='B1',
                 simulation=True,
                 no_sig_digits=True,
                 no_plots=False,
                 interaction_point='ip5',
                 cminus=0.0,
                 misalignment=0.0,
                 errorK=0.0,
                 errorL=0.0,
                 tune_uncertainty=0.0E-5,
                 phase_weight=0.5,
                 model_dir=_workdir_path)
    results = tfs.read(join(_workdir_path, "ip5B1", f"{RESULTS_FILE_NAME}{EXT}"))
    beta_twiss = {'X': 19.2, 'Y': 19.2}

    for plane in PLANES:
        beta_sim = beta_twiss[plane]
        beta_meas = results[f"{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_meas - beta_sim)) / beta_sim < LIMITS['Accuracy']
        beta_err_meas = results[f"{ERR}{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_err_meas)) < LIMITS['Num Precision']


@pytest.mark.extended
def test_kmod_phase_measured_ip5b1(_workdir_path):
    analyse_kmod(betastar_and_waist=[19.2, 0.0],
                 working_directory=_workdir_path,
                 beam='B1',
                 simulation=True,
                 no_sig_digits=True,
                 no_plots=False,
                 interaction_point='ip5',
                 cminus=0.0,
                 misalignment=0.0,
                 errorK=0.0,
                 errorL=0.0,
                 tune_uncertainty=0.0E-5,
                 phase_weight=0.5,
                 measurement_dir=_workdir_path)
    results = tfs.read(join(_workdir_path, "ip5B1", f"{RESULTS_FILE_NAME}{EXT}"))
    beta_twiss = {'X': 19.2, 'Y': 19.2}

    for plane in PLANES:
        beta_sim = beta_twiss[plane]
        beta_meas = results[f"{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_meas - beta_sim)) / beta_sim < LIMITS['Accuracy']
        beta_err_meas = results[f"{ERR}{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_err_meas)) < LIMITS['Num Precision']


@pytest.mark.extended
def test_kmod_simulation_ip1b1(_workdir_path):

    analyse_kmod(betastar_and_waist=[0.25, 0.0],
                    working_directory=_workdir_path,
                    beam='B1',
                    simulation=True,
                    no_sig_digits=True,
                    no_plots=False,
                    interaction_point='ip1',
                    cminus=0.0,
                    misalignment=0.0,
                    errorK=0.0,
                    errorL=0.0,
                    tune_uncertainty=0.0E-5)
    results = tfs.read(join(_workdir_path, "ip1B1", f"{RESULTS_FILE_NAME}{EXT}"))
    beta_twiss = {'X': 0.25, 'Y': 0.25}

    for plane in PLANES:
        beta_sim = beta_twiss[plane]
        beta_meas = results[f"{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_meas-beta_sim))/beta_sim < LIMITS['Accuracy']
        beta_err_meas = results[f"{ERR}{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_err_meas)) < LIMITS['Num Precision']


@pytest.mark.extended
def test_kmod_simulation_ip1b2(_workdir_path):

    analyse_kmod(betastar_and_waist=[0.25, 0.25, 0.0],
                    working_directory=_workdir_path,
                    beam='B2',
                    simulation=True,
                    no_sig_digits=True,
                    no_plots=False,
                    no_autoclean=True,
                    interaction_point='ip1',
                    cminus=0.0,
                    misalignment=0.0,
                    errorK=0.0,
                    errorL=0.0,
                    tune_uncertainty=0.0E-5)
    results = tfs.read(join(_workdir_path, "ip1B2", f"{RESULTS_FILE_NAME}{EXT}"))
    beta_twiss = {'X': 0.25, 'Y': 0.25}

    for plane in PLANES:
        beta_sim = beta_twiss[plane]
        beta_meas = results[f"{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_meas-beta_sim))/beta_sim < LIMITS['Accuracy']
        beta_err_meas = results[f"{ERR}{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_err_meas)) < LIMITS['Num Precision']


@pytest.mark.extended
def test_kmod_meas_ip1b1(_workdir_path):

    analyse_kmod(betastar_and_waist=[0.44, 0.44, 0.0, 0.0],
                    working_directory=_workdir_path,
                    beam='B1',
                    simulation=False,
                    no_sig_digits=True,
                    no_plots=False,
                    interaction_point='ip1',
                    cminus=0.0,
                    misalignment=0.0,
                    errorK=0.0,
                    errorL=0.0,
                    tune_uncertainty=2.5E-5)
    results = tfs.read(join(_workdir_path, "ip1B1", f"{RESULTS_FILE_NAME}{EXT}"))
    beta_prev = {'X': 0.45, 'Y': 0.43}
    for plane in PLANES:

        beta_meas = results[f"{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_meas-beta_prev[plane]))/beta_prev[plane] < LIMITS['Meas Accuracy']
        beta_err_meas = results[f"{ERR}{BETA}{STAR}{plane}"].loc[0]
        assert (beta_err_meas/beta_meas) < LIMITS['Meas Precision']


@pytest.mark.extended
def test_kmod_meas_ip1b2(_workdir_path):

    analyse_kmod(betastar_and_waist=[0.44, 0.0],
                    working_directory=_workdir_path,
                    beam='B2',
                    simulation=False,
                    no_sig_digits=True,
                    no_plots=False,
                    interaction_point='ip1',
                    cminus=0.0,
                    misalignment=0.0,
                    errorK=0.0,
                    errorL=0.0,
                    tune_uncertainty=2.5E-5)
    results = tfs.read(join(_workdir_path, "ip1B2", f"{RESULTS_FILE_NAME}{EXT}"))
    beta_prev = {'X': 0.387, 'Y': 0.410}
    for plane in PLANES:

        beta_meas = results[f"{BETA}{STAR}{plane}"].loc[0]
        assert (np.abs(beta_meas-beta_prev[plane]))/beta_prev[plane] < LIMITS['Meas Accuracy']
        beta_err_meas = results[f"{ERR}{BETA}{STAR}{plane}"].loc[0]
        assert (beta_err_meas/beta_meas) < LIMITS['Meas Precision']


@pytest.mark.extended
def test_kmod_meas_ip4b1(_workdir_path):

    analyse_kmod(betastar_and_waist=[200.0, -100.0],
                    working_directory=_workdir_path,
                    beam='B1',
                    simulation=False,
                    no_sig_digits=True,
                    no_plots=False,
                    circuits=['RQ6.R4B1', 'RQ7.R4B1'],
                    cminus=0.0,
                    misalignment=0.0,
                    errorK=0.0,
                    errorL=0.0,
                    tune_uncertainty=0.5E-5)
    results = tfs.read(join(_workdir_path, "MQY.6R4.B1-MQM.7R4.B1", f"{INSTRUMENTS_FILE_NAME}{EXT}"), index='NAME')

    original = {
                'BPMCS.7R4.B1': (17.5074335336, 157.760070696),
                'BPM.7R4.B1': (17.6430538896, 157.972911909),
                'BQSH.7R4.B1': (455.457631868, 124.586686684),
                'BPLH.7R4.B1': (423.68951095, 123.578577484)
    }

    for inst in results.index:
        beta_x, beta_y = original[inst]
        betas = dict(X=beta_x, Y=beta_y)
        for plane in PLANES:
            beta_meas = results[f"{BETA}{plane}"].loc[inst]
            assert (np.abs(beta_meas-betas[plane]))/betas[plane] < LIMITS['Meas Accuracy']
            beta_err_meas = results[f"{ERR}{BETA}{plane}"].loc[inst]
            assert (beta_err_meas/beta_meas) < LIMITS['Meas Precision']


@pytest.mark.extended
def test_kmod_simulation_ip4b1(_workdir_path):

    analyse_kmod(betastar_and_waist=[200.0, -100.0],
                 working_directory=_workdir_path,
                 beam='B1',
                 simulation=True,
                 no_sig_digits=True,
                 no_plots=False,
                 circuits=['RQ6.R4B1', 'RQ7.R4B1'],
                 cminus=0.0,
                 misalignment=0.0,
                 errorK=0.0,
                 errorL=0.0,
                 tune_uncertainty=0.5E-5)
    results = tfs.read(join(_workdir_path, "MQY.6R4.B1-MQM.7R4.B1", f"{INSTRUMENTS_FILE_NAME}{EXT}"), index='NAME')

    original = {
        'BPMCS.7R4.B1': (3.64208332528655e+01, 9.46041254954643e+01),
        'BPM.7R4.B1': (3.61317067929723e+01, 9.48945562104017e+01),
        'BQSH.7R4.B1': (5.07121388372368e+02, 9.07140610660815e+01),
        'BPLH.7R4.B1': (4.79632975072045e+02, 8.65331699893341e+01)
    }
    for inst in results.index:
        beta_x, beta_y = original[inst]
        betas = dict(X=beta_x, Y=beta_y)
        for plane in PLANES:
            beta_meas = results[f"{BETA}{plane}"].loc[inst]
            assert (np.abs(beta_meas - betas[plane])) / betas[plane] < LIMITS['Meas Accuracy']
            beta_err_meas = results[f"{ERR}{BETA}{plane}"].loc[inst]
            assert (beta_err_meas / beta_meas) < LIMITS['Meas Precision']


@pytest.mark.extended
def test_kmod_meas_ip4b2(_workdir_path):

    analyse_kmod(betastar_and_waist=[200.0, -100.0],
                 working_directory=_workdir_path,
                 beam='B2',
                 simulation=False,
                 no_sig_digits=True,
                 no_plots=False,
                 circuits=['RQ7.L4B2', 'RQ6.L4B2'],
                 cminus=0.0,
                 misalignment=0.0,
                 errorK=0.0,
                 errorL=0.0,
                 tune_uncertainty=0.5E-5)
    results = tfs.read(join(_workdir_path, "MQM.7L4.B2-MQY.6L4.B2", f"{INSTRUMENTS_FILE_NAME}{EXT}"), index='NAME')

    original = {
                'BPMYA.6L4.B2': (456.789268726, 149.073169556),
                'BGVCA.B7L4.B2': (119.359634764, 152.116072289),
                'BPLH.B7L4.B2': (434.440558008, 148.460642194),
                'BPLH.A7L4.B2': (441.781928671, 148.654814221)
    }

    for inst in results.index:
        beta_x, beta_y = original[inst]
        betas = dict(X=beta_x, Y=beta_y)
        for plane in PLANES:
            beta_meas = results[f"{BETA}{plane}"].loc[inst]
            assert (np.abs(beta_meas - betas[plane])) / betas[plane] < LIMITS['Meas Accuracy']
            beta_err_meas = results[f"{ERR}{BETA}{plane}"].loc[inst]
            assert (beta_err_meas / beta_meas) < LIMITS['Meas Precision']


@pytest.fixture()
def _workdir_path():
    try:
        workdir = join(CURRENT_DIR, pardir, "inputs", "kmod")
        yield workdir
    finally:
        if isdir(join(workdir, 'ip1B1')):
            shutil.rmtree(join(workdir, 'ip1B1'))

        if isdir(join(workdir, 'ip1B2')):
            shutil.rmtree(join(workdir, 'ip1B2'))

        if isdir(join(workdir, 'ip5B1')):
            shutil.rmtree(join(workdir, 'ip5B1'))

        if isdir(join(workdir, 'ip5B2')):
            shutil.rmtree(join(workdir, 'ip5B2'))

        if isdir(join(workdir, 'MQY.6R4.B1-MQM.7R4.B1')):
            shutil.rmtree(join(workdir, 'MQY.6R4.B1-MQM.7R4.B1'))

        if isdir(join(workdir, 'MQM.7L4.B2-MQY.6L4.B2')):
            shutil.rmtree(join(workdir, 'MQM.7L4.B2-MQY.6L4.B2'))