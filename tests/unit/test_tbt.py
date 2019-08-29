import os
import tempfile

import pytest
import numpy as np
import pandas as pd
from . import context
from datetime import datetime
from tbt import handler, iota_handler, ptc_handler


CURRENT_DIR = os.path.dirname(__file__)
PLANES = ('X', 'Y')


def test_tbt_write_read_sdds_binary(_sdds_file, _test_file):
    origin = handler.read_tbt(_sdds_file)
    handler.write_tbt(_test_file, origin)
    new = handler.read_tbt(f'{_test_file}.sdds')
    _compare_tbt(origin, new, False)


def test_tbt_read_hdf5(_hdf5_file):

    origin = handler.TbtData(
        matrices=[
                  {'X': pd.DataFrame(
                    index=['IBPMA1C', 'IBPME2R'],
                    data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 2, np.sin),
                    dtype=float),
                   'Y': pd.DataFrame(
                    index=['IBPMA1C', 'IBPME2R'],
                    data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 2, np.cos),
                    dtype=float)}],
        date=datetime.now(),
        bunch_ids=[1],
        nturns=2000)
    new = iota_handler.read_tbt(_hdf5_file)
    _compare_tbt(origin, new, False)


def test_tbt_read_ptc(_ptc_file):
    BPMS = ['C1.BPM1']
    NTURNS = 1000
    origin = handler.TbtData(
        matrices=[
            {'X': pd.DataFrame(
                index=BPMS,
                columns=range(NTURNS),
                data=[
                    _create_x(0.001, 0, NTURNS, 2.067,
                              21.7172216, -3.11587987),
                    #_create_x(0.0002679129997, 0, NTURNS, 2.067,
                    #          8.814519469, 1.917380654),
                ],
                dtype=float),
             'Y': pd.DataFrame(
                index=BPMS,
                columns=range(NTURNS),
                data=[
                    _create_x(0.001, 0, NTURNS, 2.155,
                              2.442183557, 0.1734995035),
                    #_create_x(0.001732087, 0, NTURNS, 2.155,
                    #           8.460497676, -1.865668818),
                ],
                dtype=float),
             },
            {'X': pd.DataFrame(
                index=BPMS,
                columns=range(NTURNS),
                data=[
                    _create_x(0.0011, 0, NTURNS, 2.067,
                              21.7172216, -3.11587987),
                    #_create_x(0.0002947042997, 0, NTURNS, 2.155,
                    #          8.814519469, 1.917380654),
                ],
                dtype=float),
             'Y': pd.DataFrame(
                index=BPMS,
                columns=range(NTURNS),
                data=[
                    _create_x(0.0011, 0, NTURNS, 2.155,
                              2.442183557, 0.1734995035),
                    #_create_x(0.0019052957, 0, NTURNS, 2.155,
                    #           8.460497676, -1.865668818),
                ],
                dtype=float),
             },
        ],
        date=datetime.now(),
        bunch_ids=[1, 2],
        nturns=NTURNS)
    new = ptc_handler.read_tbt(_ptc_file)
    _compare_tbt(origin, new, True)


def test_tbt_read_ptc_looseparticles(_ptc_file_losses):
    new = ptc_handler.read_tbt(_ptc_file_losses)
    assert len(new.matrices) == 3
    assert len(new.matrices[0]["X"].columns) == 1024
    assert all(new.matrices[0]["X"].index == np.array([f"BPM{i+1}" for i in range(3)]))
    assert not new.matrices[0]["X"].isna().any().any()


def _create_data(nturns, nbpm, function):
    return np.ones((nbpm, len(nturns))) * function(nturns)


def _create_x(x0, px0, turns, Qx, beta, alfa):
    GAMMA = (1 + alfa**2) / beta
    MU = Qx * np.pi * 2.

    ONETURN = np.array([[np.cos(MU) + alfa*np.sin(MU), beta * np.sin(MU)],
                        [-GAMMA*np.sin(MU), np.cos(MU) - alfa*np.sin(MU)]])
    x_px = [np.array([x0, px0])]

    for nturn in range(turns-1):
        x_px.append(np.matmul(ONETURN, x_px[-1]))
    return [x[0] for x in x_px]


def test_tbt_write_read_ascii(_sdds_file, _test_file):
    origin = handler.read_tbt(_sdds_file)
    handler.write_lhc_ascii(_test_file, origin)
    new = handler.read_tbt(_test_file)
    _compare_tbt(origin, new, True)


def _compare_tbt(origin, new, no_binary):
    assert new.nturns == origin.nturns
    assert new.nbunches == origin.nbunches
    assert new.bunch_ids == origin.bunch_ids
    for index in range(origin.nbunches):
        for plane in PLANES:
            assert np.all(new.matrices[index][plane].index == origin.matrices[index][plane].index)
            origin_mat = origin.matrices[index][plane].values
            new_mat = new.matrices[index][plane].values
            if no_binary:
                ascii_precision = 0.5 / np.power(10, handler.PRINT_PRECISION)
                assert np.max(np.abs(origin_mat - new_mat)) < ascii_precision
            else:
                assert np.all(origin_mat == new_mat)


@pytest.fixture()
def _sdds_file():
    return os.path.join(CURRENT_DIR, os.pardir, "inputs", "test_file.sdds")


@pytest.fixture()
def _hdf5_file():
    return os.path.join(CURRENT_DIR, os.pardir, "inputs", "test_file.hdf5")


@pytest.fixture()
def _test_file():
    with tempfile.TemporaryDirectory() as cwd:
        yield os.path.join(cwd, "test_file")


@pytest.fixture()
def _ptc_file():
    return os.path.join(CURRENT_DIR, os.pardir, "inputs", "test_trackone")


@pytest.fixture()
def _ptc_file_losses():
    return os.path.join(CURRENT_DIR, os.pardir, "inputs", "test_trackone_losses")
