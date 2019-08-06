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
SEGMENT_MARKER = ('start', 'end')
COLX = "X"
COLY = "Y"
COLTURN = "TURN"
COLPARTICLE = "NUMBER"

Segment = namedtuple("Segment", ["number", "turns", "particles", "element", "name"])

LOGGER = get_logger(__name__)


def read_tbt(file_path):
    """
    Reads TbtData object from PTC trackone output.

    """
    LOGGER.debug(f"Reading path: {file_path}")

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
    first_segment = True
    n_turns = 0
    n_particles = 0

    for line in lines:
        parts = line.split()
        if len(parts) == 0 or parts[0] == HEADER or parts[0] == TYPES:
            continue

        if parts[0] == NAMES:  # read column names
            if column_indices is not None:
                raise KeyError(f"{NAMES} are defined twice in tbt file!")
            column_indices = _read_column_names(parts[1:])
            continue

        if parts[0] == SEGMENTS:  # read segments, append to index
            segment = Segment(*parts[1:])
            data.append(segment)

            if first_segment and segment.name not in SEGMENT_MARKER:
                bpms.append(segment.name)

            if first_segment and segment.name == SEGMENT_MARKER[1]:  # end of first segment
                n_turns = int(segment.turns) - 1
                n_particles = int(segment.particles)
                first_segment = False
        else:
            if column_indices is None:
                raise IOError("Columns not defined in Tbt file!")

            new_data = {col: parts[col_idx] for col, col_idx in column_indices.items()}
            data.append(new_data)
            particle = int(new_data[COLPARTICLE])
            if first_segment and particle not in particles:
                particles.append(particle)

    if first_segment:
        raise IOError("First segment in Tbt file never ended.")

    if len(data) == 0:
        raise IOError("No data found in TbT file!")
    return data, bpms, particles, n_turns, n_particles


def _read_column_names(parts):
    col_idx = {k: None for k in [COLX, COLY, COLTURN, COLPARTICLE]}
    LOGGER.debug("Setting column names.")
    for idx, column_name in enumerate(parts):
        if column_name not in col_idx:
            LOGGER.debug(f"Column '{column_name}' will be ignored.")
            continue
        if col_idx[column_name] is not None:
            raise KeyError(f"'{column_name}' is defined twice.")
        col_idx[column_name] = idx
    missing = [c for c in col_idx.values() if c is None]
    if any(missing):
        raise ValueError(f"The following columns are missing in ptc file: '{str(missing)}'")
    return col_idx


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

        if current_segment.name in SEGMENT_MARKER:
            continue

        part_id = int(d[COLPARTICLE]) - 1
        turn_nr = int(d[COLTURN]) - 1
        matrix_listx[part_id][current_segment.name][turn_nr] = float(d[COLX])
        matrix_listy[part_id][current_segment.name][turn_nr] = float(d[COLY])

    return matrix_listx, matrix_listy
