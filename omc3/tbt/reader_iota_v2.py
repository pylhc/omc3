"""
Iota data handler
---------------------

Takes Hdf5 file path containing the TbT data and returns a TbtData class to be read and processed by harpy

"""
from datetime import datetime
import numpy as np
import pandas as pd
import h5py
from tbt import handler
from utils import logging_tools

LOGGER = logging_tools.getLogger(__name__)

PLANES = ('X', 'Y')
PLANES_CONV = {'X': 'Horizontal', 'Y': 'Vertical'}


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

    bpm_names = _get_list_of_bpmnames(hdf_file)
    nturns = _get_number_of_turns(hdf_file)

    matrices = [{k: pd.DataFrame(index=bpm_names,
                                 data=_get_turn_by_turn_data(hdf_file, k),
                                 dtype=float) for k in PLANES}]

    return handler.TbtData(matrices, date, bunch_ids, nturns)


def _get_turn_by_turn_data(hd5, plane):

    keys = [key for key in hd5.keys()]
    nbpm = len(keys)
    nturn = _get_number_of_turns(hd5)
    data = np.zeros((nbpm, nturn))
    for i, key in enumerate(keys):
        data[i, :] = hd5[key][PLANES_CONV[plane]][:nturn]

    return data


def _get_list_of_bpmnames(hd5):
    bpms = [f'IBPM{key}' for key in list(hd5.keys()) if not ('NL' in key)]
    return np.unique(bpms)


def _get_number_of_turns(hd5):
    lengths = np.array([(len(hd5[key][PLANES_CONV['X']]), len(hd5[key][PLANES_CONV['Y']])) for key in list(hd5.keys()) if not ('NL' in key)])
    return np.min(lengths.flatten())
