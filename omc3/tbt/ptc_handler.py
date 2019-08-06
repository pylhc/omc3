# this file is in great parts a copy of tbt.handler because the ptc_trackone files do have a very
# similar format
import pandas as pd
import numpy as np
from datetime import datetime

from tbt import TbtData
from utils.logging_tools import get_logger

HEADER = "@"
NAMES = "*"
TYPES = "$"
SEGMENTS = "#segment"
COLX = "X"
COLY = "Y"
COLTURN = "TURN"
COLPART = "NUMBER"

LOGGER = get_logger(__name__)


def read_tbt(file_path):
    """
    Reads TbtData object from PTC trackone output.

    """
    colx = 0
    coly = 0
    colpart = 0
    colturn = 0

    nturns = 0
    bpms = []
    particles = []

    LOGGER.debug(f"Reading path: {file_path}")
    lines = []

    with open(file_path, "r") as tfs_data:
        for line in tfs_data:
            parts = line.split()

            if parts[0] == NAMES:  # read column names
                LOGGER.debug("Setting column names.")
                column_names = np.array(parts[1:])
                for i in range(len(column_names)):
                    if column_names[i] == COLX:
                        colx = i
                    elif column_names[i] == COLY:
                        coly = i
                    elif column_names[i] == COLPART:
                        colpart = i
                    elif column_names[i] == COLTURN:
                        colturn = i
            elif parts[0] == SEGMENTS:  # read segments, append to index
                [segnr, nturns, npart, elnr, name] = parts[1:]
                if name == "end":
                    break
                elif name != "start":
                    bpms.append(name)
                    lines.append(line)
                    
            elif parts[0] == HEADER or parts[0] == TYPES or len(parts) == 0:
                continue
            else:
                lines.append(line)
                partnr = int(parts[colpart])
                if partnr not in particles:
                    particles.append(partnr)

        n_turns = int(nturns) - 1
        n_particles = int(npart)

        matrix_listx = [{bpm: np.zeros(n_turns) for bpm in bpms} for bid in range(n_particles)]
        matrix_listy = [{bpm: np.zeros(n_turns) for bpm in bpms} for bid in range(n_particles)]
        # prepare matrices:
        segmentparts = [segnr, nturns, npart, elnr, name]
        for line in lines:
            segmentparts = _parse_line(line, segmentparts, matrix_listx, matrix_listy,
                                       colx, coly, colpart, colturn)

        segmentparts = [segnr, nturns, npart, elnr, name]
        for line in tfs_data:
            segmentparts = _parse_line(line, segmentparts, matrix_listx, matrix_listy,
                                       colx, coly, colpart, colturn)

        matrices = [
            {
                "X": pd.DataFrame(data=matrix_listx[bid]).transpose(),
                "Y": pd.DataFrame(data=matrix_listy[bid]).transpose(),
            }
            for bid in range(n_particles)
        ]

    # TODO: read date from file
    return TbtData(matrices, datetime.now(), particles, n_turns)


def _parse_line(line, segmentparts, matrix_listx, matrix_listy,
                colx, coly, colpart, colturn):

    [segnr, nturns, npart, elnr, name] = segmentparts
    parts = line.split()
    if len(parts) == 0:
        return segmentparts
    elif parts[0] == SEGMENTS:
        return parts[1:]
    elif name != "start" and name != "end":
        part_id = int(parts[colpart]) - 1
        turn_nr = int(parts[colturn]) - 1
        matrix_listx[part_id][name][turn_nr] = float(parts[colx])
        matrix_listy[part_id][name][turn_nr] = float(parts[coly])
    return segmentparts
