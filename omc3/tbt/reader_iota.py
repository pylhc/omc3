"""
Iota data handler
---------------------

Takes Hdf5 file path containing the TbT data and returns a TbtData class to be read and processed by harpy

"""
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

from omc3.tbt import handler
from omc3.utils import logging_tools

LOGGER = logging_tools.getLogger(__name__)

PLANES = ('X', 'Y')

VERSIONS = (1, 2)

PLANES_CONV = {1: {'X': 'H', 'Y': 'V'},
               2: {'X': 'Horizontal', 'Y': 'Vertical'}}


def read_tbt(file_path):
    """
    Reads TbTData object from provided file_path
    Args:
        file_path: path to a file containing TbtData

    Returns:
        TbtData
    """

    hdf_file = h5py.File(file_path, 'r')
    bunch_ids = [1]
    date = datetime.now()

    for vers in VERSIONS[::-1]:
        try:
            bpm_names = FUNCTIONS[vers]['get_bpm_names'](hdf_file)
            nturns = FUNCTIONS[vers]['get_nturns'](hdf_file, vers)
            matrices = [{k: pd.DataFrame(index=bpm_names,
                                         data=FUNCTIONS[vers]['get_tbtdata'](hdf_file, k, vers),
                                         dtype=float) for k in PLANES}]

            return handler.TbtData(matrices, date, bunch_ids, nturns)

        except TypeError:
            pass


def _get_turn_by_turn_data_v1(hd5, plane, version):

    keys = [key for key in hd5.keys() if (key.endswith(PLANES_CONV[version][plane]))]
    nbpm = len(keys)
    nturn = FUNCTIONS[version]['get_nturns'](hd5, version)
    data = np.zeros((nbpm, nturn))
    for i, key in enumerate(keys):
        data[i, :] = hd5[key][: nturn]

    return data


def _get_list_of_bpmnames_v1(hd5):
    bpms = [f'IBPM{key[4:-1]}' for key in list(hd5.keys()) if check_key_v1(key)]
    return np.unique(bpms)


def _get_number_of_turns_v1(hd5, version):
    lengths = [len(hd5[key]) for key in list(hd5.keys()) if check_key_v1(key)]
    return np.min(lengths)


def _get_turn_by_turn_data_v2(hd5, plane, version):

    keys = [key for key in hd5.keys() if not key.startswith('N:')]
    if keys == []:
        raise TypeError('Wrong version of converter was used.')
    nbpm = len(keys)
    nturn = FUNCTIONS[version]['get_nturns'](hd5, version)
    data = np.zeros((nbpm, nturn))
    for i, key in enumerate(keys):
        data[i, :] = hd5[key][PLANES_CONV[version][plane]][:nturn]

    return data


def _get_list_of_bpmnames_v2(hd5):
    bpms = [f'IBPM{key}' for key in list(hd5.keys()) if check_key_v2(key)]
    if bpms == []:
        raise TypeError('Wrong version of converter was used.')
    return np.unique(bpms)


def _get_number_of_turns_v2(hd5, version):
    lengths = np.array([(len(hd5[key][PLANES_CONV[version]['X']]), len(hd5[key][PLANES_CONV[version]['Y']])) for key in list(hd5.keys()) if check_key_v2(key)])
    return np.min(lengths)


def check_key_v2(key):
    return not (('NL' in key) or key.startswith('N:'))


def check_key_v1(key):
    return ('state' not in key) or key.startswith('N:')


FUNCTIONS = {1: {'get_bpm_names': _get_list_of_bpmnames_v1,
                 'get_nturns': _get_number_of_turns_v1,
                 'get_tbtdata': _get_turn_by_turn_data_v1},
             2: {'get_bpm_names': _get_list_of_bpmnames_v2,
                 'get_nturns': _get_number_of_turns_v2,
                 'get_tbtdata': _get_turn_by_turn_data_v2}}
