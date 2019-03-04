"""
Module tbt.handler
---------------------


Basic tbt io-functionality.

"""
from datetime import datetime
import time
import numpy as np
import pandas as pd

import sdds
from utils import logging_tools

LOGGER = logging_tools.getLogger(__name__)

PLANES = ('X', 'Y')
NUM_TO_PLANE = {"0": "X", "1": "Y"}
PLANE_TO_NUM = {"X": "0", "Y": "1"}
POSITIONS = {"X": "horPositionsConcentratedAndSorted", "Y": "verPositionsConcentratedAndSorted"}
PRINT_PRECISION = 6
FORMAT_STRING = " {:." + str(PRINT_PRECISION) + "f}"
_ACQ_DATE_PREFIX = "#Acquisition date: "


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
        return TbtData(matrices, date, [0], matrices[0]["X"].shape[1])
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
    return TbtData(matrices, date, bunch_ids, nturns)


def write_tbt(output_path, tbt_data, headers=None, no_binary=False):
    """
    Writes TbtData either into Sdds file or ascii file in output_path
    Args:
        output_path: path to an output file
        tbt_data: TbtData to be written
        headers: optional header dictionary (for the moment only in ASCII output)
        no_binary: If True ascii file is output
    """
    if no_binary:
        _write_ascii(output_path, tbt_data, headers_dict=headers)
        return
    nbpms = tbt_data.matrices[0]["X"].index.size
    data = {'X': np.empty((nbpms, tbt_data.nbunches, tbt_data.nturns), dtype=float),
            'Y': np.empty((nbpms, tbt_data.nbunches, tbt_data.nturns), dtype=float)}
    for index in range(tbt_data.nbunches):
        for plane in PLANES:
            data[plane][:, index, :] = tbt_data.matrices[index][plane].values
    definitions = [
        sdds.classes.Parameter("acqStamp", "double"),
        sdds.classes.Parameter("nbOfCapBunches", "long"),
        sdds.classes.Parameter("nbOfCapTurns", "long"),
        sdds.classes.Array("BunchId", "long"),
        sdds.classes.Array("bpmNames", "string"),
        sdds.classes.Array(POSITIONS['X'], "float"),
        sdds.classes.Array(POSITIONS['Y'], "float")
    ]
    values = [
        int(time.time()) * 1e9,
        tbt_data.nbunches,
        tbt_data.nturns,
        tbt_data.bunch_ids,
        tbt_data.matrices[0]["X"].index.values,
        np.ravel(data['X']),
        np.ravel(data['Y'])
    ]
    sdds.write(sdds.SddsFile("SDDS1", None, definitions, values), output_path)

##################################################################################
#                   ASCII
##################################################################################


def _write_ascii(output_path, tbt_data, headers_dict=None):
    for index in range(tbt_data.nbunches):
        suffix = f"_{tbt_data.bunch_ids[index]}" if tbt_data.nbunches > 1 else ""
        with open(output_path + suffix, "w") as output_file:
            _write_header(tbt_data, index, output_file, headers_dict)
            _write_tbt_data(tbt_data, index, output_file)


def _write_header(tbt_data, index, output_file, headers_dict):
    output_file.write("#SDDSASCIIFORMAT v1\n")
    output_file.write(f"#Created: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} "
                      f"By: Python SDDS converter\n")
    output_file.write(f"#Number of turns: {tbt_data.nturns}\n")
    output_file.write(
        f"#Number of horizontal monitors: {tbt_data.matrices[index]['X'].index.size}\n")
    output_file.write(f"#Number of vertical monitors: {tbt_data.matrices[index]['Y'].index.size}\n")
    output_file.write(f"#Acquisition date: {tbt_data.date.strftime('%Y-%m-%d at %H:%M:%S')}\n")
    if headers_dict is not None:
        for name in headers_dict.keys():
            output_file.write(f"#{name}: {headers_dict[name]}\n")


def _write_tbt_data(tbt_data, bunch_id, output_file):
    row_format = "{} {} {}  " + FORMAT_STRING * tbt_data.nturns + "\n"
    for plane in PLANES:
        for bpm_index, bpm_name in enumerate(tbt_data.matrices[bunch_id][plane].index):
            samples = tbt_data.matrices[bunch_id][plane].loc[bpm_name, :].values
            output_file.write(row_format.format(PLANE_TO_NUM[plane], bpm_name, bpm_index, *samples))


def _is_ascii_file(file_path):
    """
    Returns true only if the file looks like a redable tbt ASCII file.
    """
    with open(file_path, "r") as file_data:
        try:
            for line in file_data:
                if line.strip() == "":
                    continue
                return line.startswith("#")
        except UnicodeDecodeError:
            return False
    return False


def _read_ascii(file_path):
    bpm_names = {"X": [], "Y": []}
    matrix = {"X": [], "Y": []}
    date = None
    with open(file_path, "r") as file_data:
        for line in file_data:
            line = line.strip()
            # Empty lines and comments:
            if line == "" or "#" in line:
                continue
            if _ACQ_DATE_PREFIX in line:
                date = _parse_date(line)
                continue
            # Samples:
            parts = line.split()
            plane_num = parts.pop(0)
            bpm_name = parts.pop(0)
            parts.pop(0)
            bpm_samples = np.array([float(part) for part in parts])
            try:
                bpm_names[NUM_TO_PLANE[plane_num]].append(bpm_name)
                matrix[NUM_TO_PLANE[plane_num]].append(bpm_samples)
            except KeyError:
                raise ValueError(f"Wrong plane found in: {file_path}")
    matrices = {}
    for plane in PLANES:
        matrices[plane] = pd.DataFrame(index=bpm_names[plane], data=np.array(matrix[plane]))
    return [matrices], date 


def _parse_date(line):
    date_str = line.replace(_ACQ_DATE_PREFIX, "")
    try:
        return datetime.strptime(date_str, "%Y-%m-%d at %H:%M:%S")
    except ValueError:
        return datetime.today()
