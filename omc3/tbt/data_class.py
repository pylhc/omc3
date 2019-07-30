from datetime import datetime
import numpy as np
import pandas as pd
import sdds
from utils import logging_tools

LOGGER = logging_tools.getLogger(__name__)

PLANES = ('X', 'Y')
POSITIONS = {"X": "horPositionsConcentratedAndSorted", "Y": "verPositionsConcentratedAndSorted"}
NUM_TO_PLANE = {"0": "X", "1": "Y"}
PLANE_TO_NUM = {"X": "0", "Y": "1"}
PRINT_PRECISION = 6
FORMAT_STRING = " {:." + str(PRINT_PRECISION) + "f}"


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


def write_tbt_data(output_path, tbtdata, fileformat):
    if fileformat in TBTFORMATS.keys():
        write_function = TBTFORMATS[fileformat]
        write_function(output_path, tbtdata)
    else:
        raise AttributeError(f'No write functions found for specified fileformat {fileformat}')


def _matrices_to_array(tbt_data):

    nbpms = tbt_data.matrices[0]["X"].index.size
    data = {'X': np.empty((nbpms, tbt_data.nbunches, tbt_data.nturns), dtype=float),
            'Y': np.empty((nbpms, tbt_data.nbunches, tbt_data.nturns), dtype=float)}
    for index in range(tbt_data.nbunches):
        for plane in PLANES:
            data[plane][:, index, :] = tbt_data.matrices[index][plane].values

    return data


def write_npz(output_path, tbt_data):
    LOGGER.info('TbTdata is written in .npz format')
    data = _matrices_to_array(tbt_data)
    np.savez(
        file=f'{output_path}.npz',
        DATE=tbt_data.date.strftime("%Y-%m-%d %H:%M:%S"),
        NBUNCHES=tbt_data.nbunches,
        NTURNS=tbt_data.nturns,
        BUNCH_IDS=tbt_data.bunch_ids,
        NBPM=tbt_data.matrices[0]["X"].index,
        X=data['X'],
        Y=data['Y']
     )


def write_lhc_sdds(output_path, tbt_data):
    LOGGER.info('TbTdata is written in binary SDDS (LHC) format')
    data = _matrices_to_array(tbt_data)
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
        np.ravel(data['X']),
        np.ravel(data['Y'])
    ]
    sdds.write(sdds.SddsFile("SDDS1", None, definitions, values), f'{output_path}.sdds')


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
            samples = tbt_data.matrices[bunch_id][plane].loc[bpm_name, :].values
            output_file.write(row_format.format(PLANE_TO_NUM[plane], bpm_name, bpm_index, *samples))


TBTFORMATS = {
    'LHCSDDS': write_lhc_sdds,
    'NUMPY': write_npz,
    'LHCSDDS_ASCII': write_lhc_ascii,
    # 'PICKLE': write_pickle, not yet implement
}
