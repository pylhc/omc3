"""
handler
---------------------


Basic tbt io-functionality.

"""
from datetime import datetime
import time
import numpy as np
import pandas as pd
import sdds
from tbt import data_class
from utils import logging_tools

LOGGER = logging_tools.getLogger(__name__)

PLANES = ('X', 'Y')


def read_tbt(file_path):
    """
    Reads TbTData object from provided file_path
    Args:
        file_path: path to a file containing TbtData

    Returns:
        TbtData
    """

    np_file = np.load(file_path, allow_pickle=True)  # allow_pickle necessary as some/one objects is apparently internally pickled, to be investigated
    nbunches = np_file['NBUNCHES']
    bunch_ids = np_file['BUNCH_IDS']
    if len(bunch_ids) > nbunches:
        bunch_ids = bunch_ids[:nbunches]
    nturns = np_file['NTURNS']
    date = np_file['DATE']
    bpm_names = np_file['NBPM']
    nbpms = len(bpm_names)
    data = {k: np_file[k].reshape((nbpms, nbunches, nturns)) for k in PLANES}
    matrices = [{k: pd.DataFrame(index=bpm_names,
                                 data=data[k][:, idx, :],
                                 dtype=float) for k in data} for idx in range(nbunches)]

    return data_class.TbtData(matrices, date, bunch_ids, nturns)
