"""
Iota data handler
---------------------

Takes Hdf5 file path containing the TbT data and returns a TbtData class to be read and processed by harpy

"""
from datetime import datetime
import time
import numpy as np
import pandas as pd
import h5py

from tbt import data_class
from utils import logging_tools


LOGGER = logging_tools.getLogger(__name__)

PLANES = ('H', 'V')
PRINT_PRECISION = 6
FORMAT_STRING = " {:." + str(PRINT_PRECISION) + "f}"


def read_tbt(file_path):
    """
    Reads TbTData object from provided file_path
    Args:
        file_path: path to a file containing TbtData

    Returns:
        TbtData
    """

    hdf_file = h5py.File(file_path, 'r')
    nbunches = 1
    bunch_ids = 1

    bpm_dfs=[]

    for plane in PLANES:
        bpm_dfs.append(return_tbt_df(hdf_file, plane))

    nturns = sdds_file.values["nbOfCapTurns"]

    date = datetime.fromtimestamp(sdds_file.values["acqStamp"] / 1e9)
    
    data_x = sdds_file.values[POSITIONS['X']].reshape((nbpms, nbunches, nturns))
    data_y = sdds_file.values[POSITIONS['Y']].reshape((nbpms, nbunches, nturns))
    matrices = []
    matrices.append({
            'X': pd.DataFrame(index=bpm_dfs[0].index, data=data_x[:, 0, :], dtype=float),
            'Y': pd.DataFrame(index=bpm_dfs[1].index, data=data_y[:, 0, :], dtype=float)})
    return data_class.TbtData(matrices, date, bunch_ids, nturns)


def return_tbt_df(hd5, plane):

    tbt_df = pd.DataFrame(index=hd5.keys())

    for key in hd5.keys():
        if (key.endswith(plane) and key.startswith('N:')):
            tbt_df.loc[key] = hd5[key][1:]

    tbt_df = tbt_df.reindex(['IBPM' + key[4:-1] for key in tbt_df.index])
    return tbt_df
