# this file is in great parts a copy of tbt.handler because the ptc_trackone files do have a very
# similar format
from collections import namedtuple

import pandas as pd
import numpy as np
from datetime import datetime

from tbt import TbtData
from utils.logging_tools import get_logger

HEADER = "@"
NAMES = "*"
TYPES = "$"
SEGMENTS = "#segment"
Segment = namedtuple("Segment", ["number", "nturns", "particle", "element", "name"])
COLX = "X"
COLY = "Y"
COLTURN = "TURN"
COLPARTICLE = "NUMBER"

LOGGER = get_logger(__name__)


def read_tbt(file_path):
    """
    Reads TbtData object from PTC trackone output.

    """

    LOGGER.debug(f"Reading path: {file_path}")
    lines = []

    with open(file_path, "r") as tfs_data:
        lines = tfs_data.readlines()

    data, bpms, particles, n_turns, n_particles = _read(lines)
    matrix_listx, matrix_listy = _create_matrices(data, bpms, n_turns, n_particles)

    matrices = [
        {
            "X": pd.DataFrame(data=matrix_listx[bid]).transpose(),
            "Y": pd.DataFrame(data=matrix_listy[bid]).transpose(),
        }
        for bid in range(n_particles)
    ]

    # TODO: read date from file
    return TbtData(matrices, datetime.now(), particles, n_turns)


def _read(lines):
    bpms = []
    data = []
    particles = []
    column_indices = None
    segment = None
    for line in lines:
        parts = line.split()
        if len(parts) == 0 or parts[0] == HEADER or parts[0] == TYPES:
            continue

        if parts[0] == NAMES:  # read column names
            if column_indices is not None:
                raise KeyError(f"{NAMES} are defined twice in tbt file!")
            column_indices = _read_names(parts[1:])
            continue

        if parts[0] == SEGMENTS:  # read segments, append to index
            segment = Segment(*parts[1:])
            if segment.name == "start":
                continue
            if segment.name == "end":
                break

            bpms.append(segment.name)
            data.append(segment)
        else:
            if column_indices is None:
                raise IOError("Columns not defined before data.")
            new_data = {col: parts[col_idx] for col, col_idx in column_indices.items()}
            data.append(new_data)
            if new_data[COLPARTICLE] not in particles:
                particles.append(new_data[COLPARTICLE])

    if segment is None or len(data) == 0:
        raise IOError("No data found in TbT file!")
    n_turns = int(segment.nturns) - 1
    n_particles = int(segment.particle)
    return data, bpms, particles, n_turns, n_particles


def _read_names(parts):
    col_idx = {k: None for k in [COLX, COLY, COLTURN, COLPARTICLE]}
    LOGGER.debug("Setting column names.")
    for idx, column_name in enumerate(parts):
        if column_name not in col_idx:
            raise KeyError(f"'{column_name}' is not a valid identifier.")
        if col_idx[column_name] is None:
            raise KeyError(f"'{column_name}' is defined twice.")
        col_idx[column_name] = idx
    missing = [c for c in col_idx.values() if c is None]
    if any(missing):
        raise ValueError(f"The following columns are missing in ptc file: '{str(missing)}'")


def _create_matrices(data, bpms, n_turns, n_particles):
    # prepare matrices:
    matrix_listx = [{bpm: np.zeros(n_turns) for bpm in bpms} for bid in range(n_particles)]
    matrix_listy = [{bpm: np.zeros(n_turns) for bpm in bpms} for bid in range(n_particles)]

    current_segment = None
    for d in data:
        if isinstance(d, Segment):
            current_segment = d
            continue

        if current_segment is None:
            raise IOError("Data defined before Segment defintion!")

        part_id = int(d[COLPARTICLE]) - 1
        turn_nr = int(d[COLTURN]) - 1
        matrix_listx[part_id][current_segment.name][turn_nr] = float(d[COLX])
        matrix_listy[part_id][current_segment.name][turn_nr] = float(d[COLY])
