import os
import tempfile
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pytest

from omc3.definitions.constants import PLANES
from omc3.tbt import handler, reader_iota, reader_ptc, reader_trackone
from omc3.tbt_converter import converter_entrypoint

CURRENT_DIR = os.path.dirname(__file__)


def test_converter_one_file(_sdds_file, _test_file):
    converter_entrypoint(files=[_sdds_file], outputdir=os.path.dirname(_test_file))
    origin = handler.read_tbt(_sdds_file)
    new = handler.read_tbt(f'{_test_file}.sdds')
    _compare_tbt(origin, new, False)


def test_converter_more_files(_sdds_file, _test_file):
    rep = 2
    converter_entrypoint(files=[_sdds_file], outputdir=os.path.dirname(_test_file), realizations=rep)
    origin = handler.read_tbt(_sdds_file)
    for i in range(rep):
        new = handler.read_tbt(f'{_test_file}_r{i}.sdds')
        _compare_tbt(origin, new, False)


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
    new = reader_iota.read_tbt(_hdf5_file)
    _compare_tbt(origin, new, False)


def test_tbt_read_hdf5_v2(_hdf5_file_v2):

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
    new = reader_iota.read_tbt(_hdf5_file_v2)
    _compare_tbt(origin, new, False)


def test_compare_average_Tbtdata():
    npart = 10
    data = {plane: np.concatenate(
                                  [[_create_data(np.linspace(1, 10, 10, endpoint=False, dtype=int), 2, (lambda x: np.random.randn(len(x))))]
                                   for _ in range(npart)
                                   ],
                                  axis=0)
            for plane in PLANES}

    origin = handler.TbtData(
        matrices=[
                  {'X': pd.DataFrame(
                    index=['IBPMA1C', 'IBPME2R'],
                    data=data['X'][i],
                    dtype=float),
                   'Y': pd.DataFrame(
                    index=['IBPMA1C', 'IBPME2R'],
                    data=data['Y'][i],
                    dtype=float)}
                  for i in range(npart)],
        date=datetime.now(),
        bunch_ids=range(npart),
        nturns=10)

    new = handler.TbtData(
        matrices=[
                  {'X': pd.DataFrame(
                    index=['IBPMA1C', 'IBPME2R'],
                    data=np.mean(data['X'], axis=0),
                    dtype=float),
                   'Y': pd.DataFrame(
                    index=['IBPMA1C', 'IBPME2R'],
                    data=np.mean(data['Y'], axis=0),
                    dtype=float)}],
        date=datetime.now(),
        bunch_ids=[1],
        nturns=10)

    _compare_tbt(handler.generate_average_tbtdata(origin), new, False)


def test_tbt_read_ptc(_ptc_file):
    new = reader_ptc.read_tbt(_ptc_file)
    origin = _original_trackone()
    _compare_tbt(origin, new, True)


def test_tbt_read_trackone(_ptc_file):
    new = reader_trackone.read_tbt(_ptc_file)
    origin = _original_trackone(True)
    _compare_tbt(origin, new, True)


def test_tbt_read_ptc_sci(_ptc_file_sci):
    new = reader_ptc.read_tbt(_ptc_file_sci)
    origin = _original_trackone()
    _compare_tbt(origin, new, True)


def test_tbt_read_trackone_sci(_ptc_file_sci):
    new = reader_trackone.read_tbt(_ptc_file_sci)
    origin = _original_trackone(True)
    _compare_tbt(origin, new, True)


def test_tbt_read_ptc_looseparticles(_ptc_file_losses):
    new = reader_ptc.read_tbt(_ptc_file_losses)
    assert len(new.matrices) == 3
    assert len(new.matrices[0]["X"].columns) == 9
    assert all(new.matrices[0]["X"].index == np.array([f"BPM{i+1}" for i in range(3)]))
    assert not new.matrices[0]["X"].isna().any().any()


def test_tbt_read_trackone_looseparticles(_ptc_file_losses):
    new = reader_trackone.read_tbt(_ptc_file_losses)
    assert len(new.matrices) == 3
    assert len(new.matrices[0]["X"].columns) == 9
    assert all(new.matrices[0]["X"].index == np.array([f"BPM{i+1}" for i in range(3)]))
    assert not new.matrices[0]["X"].isna().any().any()


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
            origin_mat = origin.matrices[index][plane].to_numpy()
            new_mat = new.matrices[index][plane].to_numpy()
            if no_binary:
                ascii_precision = 0.5 / np.power(10, handler.PRINT_PRECISION)
                assert np.max(np.abs(origin_mat - new_mat)) < ascii_precision
            else:
                assert np.all(origin_mat == new_mat)


def _original_trackone(track=False):
    names = np.array(["C1.BPM1"])
    matrix = [
        dict(X=pd.DataFrame(index=names, data=[[0.001, -0.0003606, -0.00165823, -0.00266631]]),
             Y=pd.DataFrame(index=names, data=[[0.001, 0.00070558, -0.00020681, -0.00093807]])),
        dict(
            X=pd.DataFrame(index=names, data=[[0.0011, -0.00039666, -0.00182406, -0.00293294]]),
            Y=pd.DataFrame(index=names,
                           data=[[0.0011, 0.00077614, -0.00022749, -0.00103188]]))]
    origin = handler.TbtData(matrix, None, [0, 1] if track else [1, 2], 4)
    return origin


def _create_data(nturns, nbpm, function):
    return np.ones((nbpm, len(nturns))) * function(nturns)


@pytest.fixture()
def _sdds_file():
    return os.path.join(CURRENT_DIR, os.pardir, "inputs", "test_file.sdds")


@pytest.fixture()
def _hdf5_file():
    with tempfile.TemporaryDirectory() as cwd:
        with h5py.File(os.path.join(cwd, f'test_file.hdf5'), 'w') as hd5_file:
            hd5_file.create_dataset("N:IBE2RH", data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten())
            hd5_file.create_dataset("N:IBE2RV", data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.cos).flatten())
            hd5_file.create_dataset("N:IBE2RS", data=_create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten())

            hd5_file.create_dataset("N:IBA1CH", data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten())
            hd5_file.create_dataset("N:IBA1CV", data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.cos).flatten())
            hd5_file.create_dataset("N:IBA1CS", data=_create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten())

        yield os.path.join(cwd, f'test_file.hdf5')


@pytest.fixture()
def _hdf5_file_v2():
    with tempfile.TemporaryDirectory() as cwd:
        with h5py.File(os.path.join(cwd, f'test_file_v2.hdf5'), 'w') as hd5_file:

            hd5_file.create_group('A1C')
            hd5_file['A1C'].create_dataset("Horizontal", data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten())
            hd5_file['A1C'].create_dataset("Vertical", data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.cos).flatten())
            hd5_file['A1C'].create_dataset("Intensity", data=_create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten())

            hd5_file.create_group('E2R')
            hd5_file['E2R'].create_dataset("Horizontal", data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten())
            hd5_file['E2R'].create_dataset("Vertical", data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.cos).flatten())
            hd5_file['E2R'].create_dataset("Intensity", data=_create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten())

        yield os.path.join(cwd, f'test_file_v2.hdf5')


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


@pytest.fixture()
def _ptc_file_sci():
    return os.path.join(CURRENT_DIR, os.pardir, "inputs", "test_trackone_sci")
