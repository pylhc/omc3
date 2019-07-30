"""
handler
---------------------


Basic tbt io-functionality.

"""
from datetime import datetime
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
    data = {k: sdds_file.values[POSITIONS[k]].reshape((nbpms, nbunches, nturns)) for k in PLANES}
    matrices = [{k: pd.DataFrame(index=bpm_names,
                                 data=data[k][:, idx, :],
                                 dtype=float) for k in data} for idx in range(nbunches)]

    return data_class.TbtData(matrices, date, bunch_ids, nturns)


def _is_ascii_file(file_path):
    """
    Returns true only if the file looks like a readable tbt ASCII file.
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
