import pytest
import os
import tfs
import shutil
import numpy as np
from . import context
from run_kmod import analyse_kmod
from kmod import kmod_constants

CURRENT_DIR = os.path.dirname(__file__)
PLANES = ('X', 'Y')
LIMITS = {'Accuracy': 0.01,
          'Meas Accuracy': 0.02,
          'Num Precision': 1E-4,
          'Meas Precision': 0.01}


def test_kmod_simulation_ip1b1(_workdir_path):

    analyse_kmod(betastar=[0.25, 0.0],
                 work_dir=_workdir_path,
                 beam='b1',
                 simulation=True,
                 ip='ip1',
                 cminus=0.0,
                 misalignment=0.0,
                 errorK=0.0,
                 errorL=0.0,
                 tunemeasuncertainty=0.0E-5)
    results = tfs.read(os.path.join(_workdir_path, "ip1B1", "results.tfs"))
    original_twiss = tfs.read(os.path.join(_workdir_path, "twiss.tfs"), index='NAME')

    for plane in PLANES:
        beta_sim = float(original_twiss.loc['IP1', f"BET{plane}"])
        beta_meas = float(results[kmod_constants.get_betastar_col(plane)].loc[0])
        assert (np.abs(beta_meas-beta_sim))/beta_sim < LIMITS['Accuracy']
        beta_err_meas = float(results[kmod_constants.get_betastar_err_col(plane)].loc[0])
        assert (np.abs(beta_err_meas)) < LIMITS['Num Precision']


def test_kmod_simulation_ip1b2(_workdir_path):

    analyse_kmod(betastar=[0.25, 0.0],
                 work_dir=_workdir_path,
                 beam='b2',
                 simulation=True,
                 ip='ip1',
                 cminus=0.0,
                 misalignment=0.0,
                 errorK=0.0,
                 errorL=0.0,
                 tunemeasuncertainty=0.0E-5)
    results = tfs.read(os.path.join(_workdir_path, "ip1B2", "results.tfs"))
    original_twiss = tfs.read(os.path.join(_workdir_path, "twiss.tfs"), index='NAME')

    for plane in PLANES:
        beta_sim = float(original_twiss.loc['IP1', f"BET{plane}"])
        beta_meas = float(results[kmod_constants.get_betastar_col(plane)].loc[0])
        assert (np.abs(beta_meas-beta_sim))/beta_sim < LIMITS['Accuracy']
        beta_err_meas = float(results[kmod_constants.get_betastar_err_col(plane)].loc[0])
        assert (np.abs(beta_err_meas)) < LIMITS['Num Precision']


def test_kmod_meas_ip1b1(_workdir_path):

    analyse_kmod(betastar=[0.44, 0.0],
                 work_dir=_workdir_path,
                 beam='b1',
                 simulation=False,
                 ip='ip1',
                 cminus=0.0,
                 misalignment=0.0,
                 errorK=0.0,
                 errorL=0.0,
                 tunemeasuncertainty=0.0E-5)
    results = tfs.read(os.path.join(_workdir_path, "ip1B1", "results.tfs"))

    for plane in PLANES:
        beta_sim = 0.45
        beta_meas = float(results[kmod_constants.get_betastar_col(plane)].loc[0])
        assert (np.abs(beta_meas-beta_sim))/beta_sim < LIMITS['Meas Accuracy']
        beta_err_meas = float(results[kmod_constants.get_betastar_err_col(plane)].loc[0])
        assert (beta_err_meas/beta_meas) < LIMITS['Meas Precision']


def test_kmod_meas_ip1b2(_workdir_path):

    analyse_kmod(betastar=[0.44, 0.0],
                 work_dir=_workdir_path,
                 beam='b2',
                 simulation=False,
                 ip='ip1',
                 cminus=0.0,
                 misalignment=0.0,
                 errorK=0.0,
                 errorL=0.0,
                 tunemeasuncertainty=0.0E-5)
    results = tfs.read(os.path.join(_workdir_path, "ip1B2", "results.tfs"))

    for plane in PLANES:
        beta_sim = 0.44
        beta_meas = float(results[kmod_constants.get_betastar_col(plane)].loc[0])
        assert (np.abs(beta_meas-beta_sim))/beta_sim < LIMITS['Meas Accuracy']
        beta_err_meas = float(results[kmod_constants.get_betastar_err_col(plane)].loc[0])
        assert (beta_err_meas/beta_meas) < LIMITS['Meas Precision']


@pytest.fixture()
def _workdir_path():
    try:
        workdir = os.path.join(CURRENT_DIR, os.pardir, "inputs", "kmod")
        yield workdir
    finally:
        if os.path.isdir(os.path.join(workdir, 'ip1B1')):
            # shutil.rmtree(os.path.join(workdir, 'ip1B1'))
            pass
        if os.path.isdir(os.path.join(workdir, 'ip1B2')):
            # shutil.rmtree(os.path.join(workdir, 'ip1B2'))
            pass
