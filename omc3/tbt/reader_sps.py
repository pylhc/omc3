"""
LHC Turn-by-Turn Data Handler
--------------------------------


Basic tbt io-functionality.

"""
from datetime import datetime

import numpy as np
import pandas as pd
import sdds

from dateutil import tz

from omc3.definitions.constants import PLANES
from omc3.tbt import handler
from omc3.utils import logging_tools

LOGGER = logging_tools.getLogger(__name__)

NUM_TO_PLANE = {"0": "H", "90": "Y"}
PLANE_TO_NUM = {"X": "1", "Y": "90"}
PLANE_CORRES = {"H": "X", "V": "Y"}

# BINARY IDs
N_BUNCHES = "nbOfCapBunches"
N_TURNS = "nbOfCapTurns"
TIMESTAMP = "acqStamp"
BPM_NAMES = "bpmNames"
VER_BPMS = "verBunchId"
HOR_BPMS = "horBunchId"


def read_tbt(file_path):
    """
    Reads TbTData object from provided file_path
    Args:
        file_path: path to a file containing TbtData

    Returns:
        TbtData
    """
    
    sdds_file = sdds.read(file_path)
    date = datetime.utcfromtimestamp(sdds_file.values[TIMESTAMP] / 1000).replace(tzinfo=tz.tzutc())
    nbunches = sdds_file.values[N_BUNCHES]
    bpm_names = sdds_file.values[BPM_NAMES]
    nturns = sdds_file.values[N_TURNS]
    nbpms = len(bpm_names)

    data = {'X': [], 'Y': []}
    for plane, key in ('X', HOR_BPMS), ('Y', VER_BPMS):
        data[plane].extend(sdds_file.values[key])
        data[plane] = np.reshape(data[plane], (nbpms, nbunches, nturns))

    matrices = [{k: pd.DataFrame(index=bpm_names,
                                 data=data[k][:, idx, :],
                                 dtype=float) for k in data} for idx in range(nbunches)]

    return handler.TbtData(matrices, date, [0], nturns)
