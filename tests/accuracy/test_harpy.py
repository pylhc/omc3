import os
import string
import random
import pytest
import pandas as pd
import numpy as np
from . import context
import tbt
import tfs

from hole_in_one import hole_in_one_entrypoint

CURRENT_DIR = os.path.dirname(__file__)
PLANES = ('X', 'Y')
LIMITS = dict(F1=1e-6, A1=1.5e-3, P1=3e-4, F2=1.5e-4, A2=1.5e-1, P2=0.03)
NOISE = 0.032
COUPLING = 0.01
NTURNS = 1024
NBPMS = 100


def test_harpy(_test_file, _model_file):
    model = _get_model_dataframe()
    tfs.write(_model_file, model, save_index="NAME")
    _write_tbt_file(model)
    hole_in_one_entrypoint(harpy=True, clean=True, autotunes="transverse", outputdir=CURRENT_DIR,
                           files=[_test_file], model=_model_file, to_write=["lin"], turn_bits=18)
    lin = dict(X=tfs.read(f"{_test_file}.linx"), Y=tfs.read(f"{_test_file}.liny"))
    model = tfs.read(_model_file)
    for plane in PLANES:
        # main and secondary frequencies
        assert _rms(_diff(lin[plane].loc[:, f"TUNE{plane}"].values,
                          model.loc[:, f"TUNE{plane}"].values)) < LIMITS["F1"]
        assert _rms(_diff(lin[plane].loc[:, f"FREQ{_couple(plane)}"].values,
                          model.loc[:, f"TUNE{_other(plane)}"].values)) < LIMITS["F2"]
        #main and secondary amplitudes
        # TODO remove factor 2 - only for backwards compatibility with Drive
        assert _rms(_rel_diff(lin[plane].loc[:, f"AMP{plane}"].values * 2,
                              model.loc[:, f"AMP{plane}"].values)) < LIMITS["A1"]
        assert _rms(_rel_diff(lin[plane].loc[:, f"AMP{_couple(plane)}"].values *
                              lin[plane].loc[:, f"AMP{plane}"].values * 2,
                              COUPLING * model.loc[:, f"AMP{_other(plane)}"].values)) < LIMITS["A2"]
        # main and secondary phases
        assert _rms(_angle_diff(lin[plane].loc[:, f"MU{plane}"].values,
                                model.loc[:, f"MU{plane}"].values)) < LIMITS["P1"]
        assert _rms(_angle_diff(lin[plane].loc[:, f"PHASE{_couple(plane)}"].values,
                                model.loc[:, f"MU{_other(plane)}"].values)) < LIMITS["P2"]


def _get_model_dataframe():
    return pd.DataFrame(data=dict(S=np.arange(NBPMS, dtype=float),
                                  AMPX=np.random.rand(NBPMS) + 1, AMPY=np.random.rand(NBPMS) + 1,
                                  MUX=np.random.rand(NBPMS) - 0.5, MUY=np.random.rand(NBPMS) - 0.5,
                                  TUNEX=0.25 + np.random.rand(1)[0] / 40,
                                  TUNEY=0.3 + np.random.rand(1)[0] / 40),
                        index=np.array([''.join(random.choices(string.ascii_uppercase, k=7))
                                        for _ in range(NBPMS)]))


def _write_tbt_file(model):
    ints = np.arange(NTURNS) - NTURNS / 2
    data_x = model.loc[:, "AMPX"].values[:, None] * np.cos(
        2 * np.pi * (model.loc[:, "MUX"].values[:, None] +
                     model.loc[:, "TUNEX"].values[:, None] * ints[None, :]))
    data_y = model.loc[:, "AMPY"].values[:, None] * np.cos(
        2 * np.pi * (model.loc[:, "MUY"].values[:, None] +
                     model.loc[:, "TUNEY"].values[:, None] * ints[None, :]))
    mats = dict(X=pd.DataFrame(data=np.random.randn(model.index.size, NTURNS) * NOISE + data_x
                               + COUPLING * data_y, index=model.index),
                Y=pd.DataFrame(data=np.random.randn(model.index.size, NTURNS) * NOISE + data_y
                               + COUPLING * data_x, index=model.index))
    tbt.write(os.path.join(CURRENT_DIR, "test_file.sdds"), tbt.TbtData([mats], None, [0], NTURNS))


def _other(plane):
    return "X" if plane == "Y" else "Y"


def _couple(plane):
    return "10" if plane == "Y" else "01"


def _rms(a):
    return np.sqrt(np.mean(np.square(a)))


def _diff(a, b):
    return a - b


def _rel_diff(a, b):
    return (a / b) - 1


def _angle_diff(a, b):
    ang = a - b
    return np.where(np.abs(ang) > 0.5, ang - np.sign(ang), ang)





@pytest.fixture()
def _test_file():
    test_file = os.path.join(CURRENT_DIR, "test_file.sdds")
    try:
        yield test_file
    finally:
        if os.path.isfile(test_file):
            os.remove(test_file)
        if os.path.isfile(test_file + ".linx"):
            os.remove(test_file + ".linx")
        if os.path.isfile(test_file + ".liny"):
            os.remove(test_file + ".liny")


@pytest.fixture()
def _model_file():
    test_file = os.path.join(CURRENT_DIR, "model.tfs")
    try:
        yield test_file
    finally:
        if os.path.isfile(test_file):
            os.remove(test_file)
