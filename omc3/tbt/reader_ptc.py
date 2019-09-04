"""
PTC Turn-by-Turn Data Handler
---------------------


"""
from collections import namedtuple

import pandas as pd
import numpy as np
from datetime import datetime

from tbt import handler
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
DATE = "DATE"
TIME = "TIME"
TIME_FORMAT = "%d/%m/%y %H.%M.%S"

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

    # header
    date, header_length = _read_header(lines)
    lines = lines[header_length:]

    # parameters
    bpms, particles, column_indices, n_turns, n_particles = _read_from_first_turn(lines)

    # data (read into dict first for speed, then convert to DF)
    matrices = [{p: {bpm: np.zeros(n_turns) for bpm in bpms} for p in PLANES} for _ in range(n_particles)]
    matrices = _read_data(lines, matrices, column_indices)
    for bunch in range(n_particles):
        for plane in PLANES:
            matrices[bunch][plane] = pd.DataFrame(matrices[bunch][plane]).transpose()

    LOGGER.debug(f"Read Tbt data from : {file_path}")
    return handler.TbtData(matrices, date, particles, n_turns)


# Read all lines ---------------------------------------------------------------


def _read_header(lines):
    """ Reads header length and datetime from header. """
    idx_line = 0
    date_str = {k: None for k in [DATE, TIME]}
    for idx_line, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 0:
            continue

        if parts[0] != HEADER:
            break

        if parts[1] in date_str.keys():
            date_str[parts[1]] = parts[-1].strip("\'\" ")

    if any(ds is None for ds in date_str.keys()):
        LOGGER.warning("Date and Time could not be read from Tbt File! Using now()!")
        return datetime.utcnow(), idx_line

    return datetime.strptime(f"{date_str[DATE]} {date_str[TIME]}", TIME_FORMAT), idx_line


def _read_from_first_turn(lines):
    """ Reads the bpms, particles, column indices and number of turns and particles
        from the data of the first turn. """
    LOGGER.debug("Reading first turn to define boundary parameters.")
    bpms = []
    particles = []
    column_indices = None
    n_turns = 0
    n_particles = 0
    first_segment = True

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0 or parts[0] in [HEADER, TYPES]:
            continue

        if parts[0] == NAMES:  # read column names
            if column_indices is not None:
                raise KeyError(f"{NAMES} are defined twice in tbt file!")
            column_indices = _parse_column_names_to_indices(parts[1:])
            continue

        if parts[0] == SEGMENTS:  # read segments, append to index
            segment = Segment(*parts[1:])
            if segment.name == SEGMENT_MARKER[0]:  # start of first segment
                n_turns = int(segment.turns) - 1
                n_particles = int(segment.particles)

            elif segment.name == SEGMENT_MARKER[1]:  # end of first segment
                break

            else:
                first_segment = False
                bpms.append(segment.name)

        elif first_segment:
            if column_indices is None:
                raise IOError("Columns not defined in Tbt file!")

            new_data = _parse_data(column_indices, parts)
            particle = int(new_data[COLPARTICLE])
            particles.append(particle)

    if len(particles) == 0:
        raise IOError("No data found in TbT file!")
    return bpms, particles, column_indices, n_turns, n_particles


def _read_data(lines, matrices, column_indices):
    """ Read the data into the matrices. """
    LOGGER.debug("Reading data.")
    segment = None
    column_map = {"X": COLX, "Y": COLY}

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0 or parts[0] in (HEADER, TYPES, NAMES):
            continue

        if parts[0] == SEGMENTS:  # start of a new segment
            segment = Segment(*parts[1:])
            continue

        if segment is None:
            raise IOError("Data defined before Segment definition!")

        if segment.name in SEGMENT_MARKER:
            continue

        data = _parse_data(column_indices, parts)
        part_id = int(data[COLPARTICLE]) - 1
        turn_nr = int(data[COLTURN]) - 1
        for plane in PLANES:
            matrices[part_id][plane][segment.name][turn_nr] = float(data[column_map[plane]])
    return matrices


# Parse single lines -----------------------------------------------------------


def _parse_data(column_indices, parts):
    """ Converts the ``parts`` into a dictionary based on the indices in ``column_indices``. """
    return {col: parts[col_idx] for col, col_idx in column_indices.items()}


def _parse_column_names_to_indices(parts):
    """ Parses the column names from the line into a dictionary with indices. """
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

