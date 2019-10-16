"""
LHC Turn-by-Turn Data Handler
--------------------------------


Basic tbt io-functionality.

"""
from datetime import datetime
import numpy as np
import pandas as pd
import sdds
from tbt import handler
from utils import logging_tools

LOGGER = logging_tools.getLogger(__name__)

PLANES = ('X', 'Y')
NUM_TO_PLANE = {"0": "X", "1": "Y"}
PLANE_TO_NUM = {"X": "0", "Y": "1"}
POSITIONS = {"X": "horPositionsConcentratedAndSorted", "Y": "verPositionsConcentratedAndSorted"}
PRINT_PRECISION = 6
FORMAT_STRING = f" {{:.{PRINT_PRECISION:d}f}}"

# ASCII IDs
_ASCII_COMMENT = "#"
_ACQ_DATE_PREFIX = "Acquisition date:"
_ACQ_DATE_FORMAT = "%Y-%m-%d at %H:%M:%S"

# BINARY IDs
N_BUNCHES = "nbOfCapBunches"
BUNCH_ID = "BunchId"
HOR_BUNCH_ID = "horBunchId"
N_TURNS = "nbOfCapTurns"
ACQ_STAMP = "acqStamp"
BPM_NAMES = "bpmNames"


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
        return handler.TbtData(matrices, date, [0], matrices[0]["X"].shape[1])

    sdds_file = sdds.read(file_path)
    nbunches = sdds_file.values[N_BUNCHES]
    bunch_ids = sdds_file.values[BUNCH_ID if BUNCH_ID in sdds_file.values else HOR_BUNCH_ID]
    if len(bunch_ids) > nbunches:
        bunch_ids = bunch_ids[:nbunches]
    nturns = sdds_file.values[N_TURNS]
    date = datetime.fromtimestamp(sdds_file.values[ACQ_STAMP] / 1e9)
    bpm_names = sdds_file.values[BPM_NAMES]
    nbpms = len(bpm_names)
    data = {k: sdds_file.values[POSITIONS[k]].reshape((nbpms, nbunches, nturns)) for k in PLANES}
    matrices = [{k: pd.DataFrame(index=bpm_names,
                                 data=data[k][:, idx, :],
                                 dtype=float) for k in data} for idx in range(nbunches)]
    return handler.TbtData(matrices, date, bunch_ids, nturns)


def _is_ascii_file(file_path):
    """
    Returns true only if the file looks like a readable tbt ASCII file.
    """
    with open(file_path, "r") as file_data:
        try:
            for line in file_data:
                if line.strip() == "":
                    continue
                return line.startswith(_ASCII_COMMENT)
        except UnicodeDecodeError:
            return False
    return False


def _read_ascii(file_path):
    """ Read the ascii file. """
    with open(file_path, "r") as file_data:
        data_lines = file_data.readlines()

    bpm_names = {"X": [], "Y": []}
    bpm_data = {"X": [], "Y": []}
    date = None

    for line in data_lines:
        line = line.strip()

        # acquisition date
        if _ACQ_DATE_PREFIX in line:
            date = _parse_date(line)
            continue

        # empty line or comments
        if line == "" or line.startswith(_ASCII_COMMENT):
            continue

        # samples:
        plane_num, bpm_name, bpm_samples = _parse_samples(line)
        try:
            bpm_names[NUM_TO_PLANE[plane_num]].append(bpm_name)
            bpm_data[NUM_TO_PLANE[plane_num]].append(bpm_samples)
        except KeyError:
            raise ValueError(f"Plane number '{plane_num}' found in file '{file_path}'.\n"
                             "Only '0' and '1' are allowed.")

    matrices = [{p: pd.DataFrame(index=bpm_names[p], data=np.array(bpm_data[p])) for p in PLANES}]
    return matrices, date


# ASCII-File Helper ------------------------------------------------------------

def _parse_samples(line):
    parts = line.split()
    plane_num = parts[0]
    bpm_name = parts[1]
    # index = part[2]  # not used, comment for clarification
    bpm_samples = np.array([float(part) for part in parts[3:]])
    return plane_num, bpm_name, bpm_samples


def _parse_date(line):
    date_str = line.replace(_ACQ_DATE_PREFIX, "").replace(_ASCII_COMMENT, "").strip()
    try:
        return datetime.strptime(date_str, _ACQ_DATE_FORMAT)
    except ValueError:
        return datetime.today()
