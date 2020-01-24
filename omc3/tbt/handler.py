from datetime import datetime

import numpy as np
import pandas as pd
import sdds

from omc3.tbt import (reader_esrf, reader_iota, reader_lhc, reader_ptc,
                      reader_trackone)
from omc3.utils import logging_tools

LOGGER = logging_tools.getLogger(__name__)

PLANES = ('X', 'Y')
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


def generate_average_tbtdata(tbtdata):
    """
    Takes a TbtData object and returns TbtData object containing the average over all
    bunches/particles at all used BPMs.
    """
    data = tbtdata.matrices
    bpm_names = data[0]['X'].index

    matrices = [{plane: pd.DataFrame(index=bpm_names,
                                     data=get_averaged_data(bpm_names, data, plane, tbtdata.nturns),
                                     dtype=float) for plane in PLANES}]
    return TbtData(matrices, tbtdata.date, [1], tbtdata.nturns)


def get_averaged_data(bpm_names, data, plane, turns):

    bpm_data = np.empty((len(bpm_names), len(data), turns))
    bpm_data.fill(np.nan)
    for idx, bpm in enumerate(bpm_names):
        for i in range(len(data)):
            bpm_data[idx, i, :len(data[i][plane].loc[bpm])] = data[i][plane].loc[bpm]

    return np.nanmean(bpm_data, axis=1)


def read_tbt(file_path, datatype="lhc"):
    return DATA_READERS[datatype].read_tbt(file_path)


def write_tbt(output_path, tbt_data, noise=None):
    LOGGER.info('TbTdata is written in binary SDDS (LHC) format')
    defs = reader_lhc
    data = _matrices_to_array(tbt_data)
    if noise is not None:
        data = _add_noise(data, noise)
    definitions = [
        sdds.classes.Parameter(defs.ACQ_STAMP, "llong"),
        sdds.classes.Parameter(defs.N_BUNCHES, "long"),
        sdds.classes.Parameter(defs.N_TURNS, "long"),
        sdds.classes.Array(defs.BUNCH_ID, "long"),
        sdds.classes.Array(defs.BPM_NAMES, "string"),
        sdds.classes.Array(defs.POSITIONS['X'], "float"),
        sdds.classes.Array(defs.POSITIONS['Y'], "float")
    ]
    values = [
        tbt_data.date.timestamp()*1e9,
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
    return data + noise * np.random.standard_normal(data.shape)


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
