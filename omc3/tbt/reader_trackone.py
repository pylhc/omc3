"""
Trackone TbT Data Handler
-------------------------

Tbt data handling from ``MAD-X`` trackone.
"""
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from omc3.tbt import handler


def read_tbt(infile: Union[str, Path]):
    """
    Reads TbtData object from ``MAD-X`` **trackone** output.

    Args:
        infile (Union[str, Path]): path to a file containing TbtData.

    Returns:
        A ``TbtData`` object with the loaded data.
    """
    nturns, npart = get_trackone_stats(infile)
    names, matrix = get_structure_from_trackone(nturns, npart, infile)
    # matrix[0, 2] contains just (x, y) samples.
    return handler.numpy_to_tbts(names, matrix[[0, 2]])


def load_dict(file_name: Union[str, Path]):  # check length?
    loaded = np.load(file_name)
    return loaded[loaded.files[0]].item()  # np.ndarray.item()


def get_trackone_stats(infile: Union[str, Path]) -> Tuple[int, int]:
    """
    Determines the number of particles and turns in the data from the provided ``MAD-X``
    **trackone** file.

    Args:
        infile (Union[str, Path]): path to a file containing TbtData.

    Returns:
        A tuple with the number of turns and particles.
    """
    stats_string = ""
    nturns, nparticles = 0, 0
    first_seg = True
    with Path(infile).open("r") as input_file:
        for line in input_file:
            if len(line.strip()) == 0:
                continue
            if line.strip()[0] in ["@", "*", "$"]:
                stats_string = stats_string + line
                continue
            parts = line.split()
            if parts[0] == "#segment":
                if not first_seg:
                    break
                nturns = int(parts[2])
                nparticles = int(parts[3])
                first_seg = False
            if parts[0] == "-1":
                nparticles = 1
            stats_string = stats_string + line
    with open("stats.txt", "w") as stats_file:
        stats_file.write(stats_string)
    return nturns - 1, nparticles


def get_structure_from_trackone(
    nturns: int = 0, npart: int = 0, infile: Union[str, Path] = "trackone"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts BPM names and particle coordinates in the **trackone** file produced by ``MAD-X``.

    Args:
        nturns (int): Number of turns tracked in the **trackone**, i.e. obtained from
            ``get_trackone_stats()``.
        npart (int):  Number of particles tracked in the **trackone**, i.e. obtained from
            ``get_trackone_stats()``.
        infile (Union[str, Path]): path to trackone file to be read.

    Returns:
        A Numpy array of BPM names and a 4D Numpy array [quantity, BPM, particle/bunch No.,
        turn No.] quantities in order [x, px, y, py, t, pt, s, E].
    """
    bpms = dict()
    with Path(infile).open("r") as input_file:
        for line in input_file:
            if len(line.strip()) == 0:
                continue
            if line.strip()[0] in ["@", "*", "$"]:
                continue
            parts = line.split()
            if parts[0] == "#segment":
                bpm_name = parts[-1].upper()
                if (np.all([k not in bpm_name.lower() for k in ["start", "end"]])) and (
                    bpm_name not in bpms.keys()
                ):
                    bpms[bpm_name] = np.empty([npart, nturns, 8], dtype=float)
            elif np.all([k not in bpm_name.lower() for k in ["start", "end"]]):
                bpms[bpm_name][
                    np.abs(int(float(parts[0]))) - 1, int(float(parts[1])) - 1, :
                ] = np.array(parts[2:])
    return np.array(list(bpms.keys())), np.transpose(
        np.array(list(bpms.values())), axes=[3, 0, 1, 2]
    )
