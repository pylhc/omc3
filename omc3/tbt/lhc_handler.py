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
NUM_TO_PLANE = {"0": "X", "1": "Y"}
PLANE_TO_NUM = {"X": "0", "Y": "1"}
POSITIONS = {"X": "horPositionsConcentratedAndSorted", "Y": "verPositionsConcentratedAndSorted"}
PRINT_PRECISION = 6
FORMAT_STRING = " {:." + str(PRINT_PRECISION) + "f}"
_ACQ_DATE_PREFIX = "#Acquisition date: "


def read_tbt(file_path):
    """
    Reads TbTData object from provided file_path
    Args:
        file_path: path to a file containing TbtData

    Returns:
        TbtData
    """
    if _is_ascii_file(file_path):
        matrices, date = _read_ascii(file_path)
        return data_class.TbtData(matrices, date, [0], matrices[0]["X"].shape[1])
    sdds_file = sdds.read(file_path)
    nbunches = sdds_file.values["nbOfCapBunches"]
    bunch_ids = sdds_file.values["BunchId" if "BunchId" in sdds_file.values else "horBunchId"]
    if len(bunch_ids) > nbunches:
        bunch_ids = bunch_ids[:nbunches]
    nturns = sdds_file.values["nbOfCapTurns"]
    date = datetime.fromtimestamp(sdds_file.values["acqStamp"] / 1e9)
    bpm_names = sdds_file.values["bpmNames"]
    nbpms = len(bpm_names)
    data_x = sdds_file.values[POSITIONS['X']].reshape((nbpms, nbunches, nturns))
    data_y = sdds_file.values[POSITIONS['Y']].reshape((nbpms, nbunches, nturns))
    matrices = []
    for index in range(nbunches):
        matrices.append({
            'X': pd.DataFrame(index=bpm_names, data=data_x[:, index, :], dtype=float),
            'Y': pd.DataFrame(index=bpm_names, data=data_y[:, index, :], dtype=float)})
    return data_class.TbtData(matrices, date, bunch_ids, nturns)
