"""
PTC Turn-by-Turn Data Handler
---------------------


"""
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

PLANES = ("X", "Y")

Segment = namedtuple("Segment", ["number", "turns", "particles", "element", "name"])

LOGGER = get_logger(__name__)


def read_tbt(file_path):
    """
    Reads TbtData object from PTC trackone output.

    """
    LOGGER.debug(f"Reading path: {file_path}")

    with open(file_path, "r") as tfs_data:
        lines = tfs_data.readlines()

    bpms, particles, column_indices, n_turns, n_particles = _read_from_first_turn(lines)
    matrix_dict = {p: [{bpm: np.zeros(n_turns) for bpm in bpms} for bid in range(n_particles)] for p in PLANES}
    matrix_dict = _read(lines, matrix_dict, column_indices)

    matrices = [
        {p: pd.DataFrame(data=matrix_dict[p][bid]).transpose() for p in ("X", "Y")}
        for bid in range(n_particles)
    ]

    # TODO: read date from file
    return TbtData(matrices, datetime.now(), particles, n_turns)


def _read_from_first_turn(lines):
    bpms = []
    particles = []
    column_indices = None
    n_turns = 0
    n_particles = 0
    first_segment = True

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

            if segment.name not in SEGMENT_MARKER:
                first_segment = False
                bpms.append(segment.name)

            if segment.name == SEGMENT_MARKER[1]:  # end of first segment
                n_turns = int(segment.turns) - 1
                n_particles = int(segment.particles)
                break

        elif first_segment:
            if column_indices is None:
                raise IOError("Columns not defined in Tbt file!")

            new_data = _get_data(column_indices, parts)
            particle = int(new_data[COLPARTICLE])
            particles.append(particle)

    if len(particles) == 0:
        raise IOError("No data found in TbT file!")
    return bpms, particles, column_indices, n_turns, n_particles


def _get_data(column_indices, parts):
    return {col: parts[col_idx] for col, col_idx in column_indices.items()}


def _read(lines, matrix_dict, column_indices):

    segment = None
    column_map = {"X": COLX, "Y": COLY}

    for line in lines:
        parts = line.split()
        if len(parts) == 0 or parts[0] == HEADER or parts[0] == TYPES or parts[0] == NAMES:
            continue

        if parts[0] == SEGMENTS:  # read segments, append to index
            segment = Segment(*parts[1:])
            continue

        if segment is None:
            raise IOError("Data defined before Segment definition!")

        if segment.name in SEGMENT_MARKER:
            continue

        data = _get_data(column_indices, parts)
        part_id = int(data[COLPARTICLE]) - 1
        turn_nr = int(data[COLTURN]) - 1

        for p in PLANES:
            matrix_dict[p][part_id][segment.name][turn_nr] = float(data[column_map[p]])
    return matrix_dict


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

