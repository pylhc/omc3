from datetime import datetime
import numpy as np
import pandas as pd
import sdds
from utils import logging_tools
from tbt import reader_esrf, reader_iota, reader_lhc, reader_ptc, reader_trackone
LOGGER = logging_tools.getLogger(__name__)

PLANES = ('X', 'Y')
POSITIONS = {"X": "horPositionsConcentratedAndSorted", "Y": "verPositionsConcentratedAndSorted"}
NUM_TO_PLANE = {"0": "X", "1": "Y"}
PLANE_TO_NUM = {"X": 0, "Y": 1}
PRINT_PRECISION = 6
FORMAT_STRING = " {:." + str(PRINT_PRECISION) + "f}"
DATA_READERS = dict(lhc=reader_lhc,
                    iota=reader_iota,
                    esrf=reader_esrf,
                    ptc=reader_ptc,
                    trackone=reader_trackone)


class TbtData(object):
    """
    Object holding a representation of a Turn-by-Turn Data

    """
    def __init__(self, matrices, date, bunch_ids, nturns):
        self.matrices = matrices  # list per bunch containing dict per plane of DataFrames
        self.date = date if date is not None else datetime.now()
        self.nbunches = len(bunch_ids)
        self.nturns = nturns
        self.bunch_ids = bunch_ids


def read_tbt(file_path, datatype="lhc"):
    return DATA_READERS[datatype].read_tbt(file_path)


def write_tbt(output_path, tbt_data, noise=None):
    LOGGER.info('TbTdata is written in binary SDDS (LHC) format')
    data = _matrices_to_array(tbt_data)
    if noise is not None:
        data = _add_noise(data, noise)
    definitions = [
        sdds.classes.Parameter("acqStamp", "long"),
        sdds.classes.Parameter("nbOfCapBunches", "long"),
        sdds.classes.Parameter("nbOfCapTurns", "long"),
        sdds.classes.Array("BunchId", "long"),
        sdds.classes.Array("bpmNames", "string"),
        sdds.classes.Array(POSITIONS['X'], "float"),
        sdds.classes.Array(POSITIONS['Y'], "float")
    ]
    values = [
        tbt_data.date.timestamp()*1000,
        tbt_data.nbunches,
        tbt_data.nturns,
        tbt_data.bunch_ids,
        tbt_data.matrices[0]["X"].index,
        np.ravel(data[PLANE_TO_NUM['X']]),
        np.ravel(data[PLANE_TO_NUM['Y']])
    ]
    sdds.write(sdds.SddsFile("SDDS1", None, definitions, values), f'{output_path}.sdds')


def _matrices_to_array(tbt_data):
    nbpms = tbt_data.matrices[0]["X"].index.size
    data = np.empty((2, nbpms, tbt_data.nbunches, tbt_data.nturns), dtype=float)
    for index in range(tbt_data.nbunches):
        for plane in PLANES:
            data[PLANE_TO_NUM[plane], :, index, :] = tbt_data.matrices[index][plane].to_numpy()
    return data


def _add_noise(data, noise):
    if noise <= 0.0:
        return data
    return data + noise * np.random.randn(data.shape)


def write_lhc_ascii(output_path, tbt_data):
    LOGGER.info('TbTdata is written in ascii SDDS (LHC) format')

    for index in range(tbt_data.nbunches):
        suffix = f"_{tbt_data.bunch_ids[index]}" if tbt_data.nbunches > 1 else ""
        with open(output_path + suffix, "w") as output_file:
            _write_header(tbt_data, index, output_file)
            _write_tbt_data(tbt_data, index, output_file)


def _write_header(tbt_data, index, output_file):
    output_file.write("#SDDSASCIIFORMAT v1\n")
    output_file.write(f"#Created: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} "
                      f"By: Python SDDS converter\n")
    output_file.write(f"#Number of turns: {tbt_data.nturns}\n")
    output_file.write(
        f"#Number of horizontal monitors: {tbt_data.matrices[index]['X'].index.size}\n")
    output_file.write(f"#Number of vertical monitors: {tbt_data.matrices[index]['Y'].index.size}\n")
    output_file.write(f"#Acquisition date: {tbt_data.date.strftime('%Y-%m-%d at %H:%M:%S')}\n")


def _write_tbt_data(tbt_data, bunch_id, output_file):
    row_format = "{} {} {}  " + FORMAT_STRING * tbt_data.nturns + "\n"
    for plane in PLANES:
        for bpm_index, bpm_name in enumerate(tbt_data.matrices[bunch_id][plane].index):
            samples = tbt_data.matrices[bunch_id][plane].loc[bpm_name, :].to_numpy()
            output_file.write(row_format.format(PLANE_TO_NUM[plane], bpm_name, bpm_index, *samples))


def numpy_to_tbts(names, matrix):
    """Converts turn by turn data and names into TbTData.

    Arguments:
        names: Numpy array of BPM names
        matrix: 4D Numpy array [quantity, BPM, particle/bunch No., turn No.]
            quantities in order [x, y]
    """
    # get list of TbTFile from 4D matrix ...
    _, nbpms, nbunches, nturns = matrix.shape
    matrices = []
    indices = []
    for index in range(nbunches):
        matrices.append({"X": pd.DataFrame(index=names, data=matrix[0, :, index, :]),
                         "Y": pd.DataFrame(index=names, data=matrix[1, :, index, :])})
        indices.append(index)
    return TbtData(matrices, None, indices, nturns)
