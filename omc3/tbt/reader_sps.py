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
N_BUNCHES = "BunchNos"
N_TURNS = "nbOfTurns"
TIMESTAMP = "timestampSecond"
BPM_NAMES = "MonNames"
CHANNEL_NAMES = "ChannelNames"


def read_tbt(file_path):
    """
    Reads TbTData object from provided file_path
    Args:
        file_path: path to a file containing TbtData

    Returns:
        TbtData
    """
    
    sdds_file = sdds.read(file_path)
    date = datetime.utcfromtimestamp(sdds_file.values[TIMESTAMP]).replace(tzinfo=tz.tzutc())
    nbunches = sdds_file.values[N_BUNCHES].max()
    bpm_names = sdds_file.values[BPM_NAMES]
    channel_names = sdds_file.values[CHANNEL_NAMES]
    nturns = sdds_file.values[N_TURNS]
    nbpms = len(bpm_names)
    nchannels = len(channel_names)

    data = {'X': [], 'Y': []}
    inverted_plane = {'X': 'Y', 'Y': 'X'}
    for channel in channel_names:
        # The last letter of a channel gives the plane, e.g. SPS.BPH.10208.H/H
        plane = PLANE_CORRES[channel[-1]]  

        # Both planes are filled here to have the same matrices sizes
        data[plane].extend(sdds_file.values[channel])
        data[inverted_plane[plane]].extend([0] * len(sdds_file.values[channel]))

    for plane in data.keys():
        data[plane] = np.reshape(data[plane], (nchannels, nbunches, nturns))

    # Remove the SPS. and .H/H or .V/V ad the end of the channel names
    stripped_names = ['.'.join(s.split('.')[1:-1]) for s in channel_names]
    matrices = [{k: pd.DataFrame(index=stripped_names,
                                 data=data[k][:, idx, :],
                                 dtype=float) for k in data} for idx in range(nbunches)]

    return handler.TbtData(matrices, date, [0], nturns)
