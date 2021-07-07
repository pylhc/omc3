"""
ESRF TbT Data Handler
---------------------

Data handling for tbt data from ``ESRF``.
"""
import json

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from scipy.io import loadmat

from omc3.tbt import handler

BPM_NAMES_FILE: str = "bpm_names.json"


def read_tbt(filepath: Union[str, Path]):
    """
    Reads ESRF ``Matlab`` file.

    Args:
        filepath (Union[str, Path]): path to a file

    Returns:
        `tbt.TbTData` object.
    """
    filepath = Path(filepath)
    names, matrix = load_esrf_mat_file(filepath)
    return handler.numpy_to_tbts(names, matrix)


def load_esrf_mat_file(infile: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the ESRF TbT ``Matlab`` file, checks for nans and data duplicities from consecutive kicks.

    Args:
        infile (Union[str, Path]): path to file to be read

    Returns:
        A Numpy array of BPM names and a 4D Numpy array [quantity, BPM, particle/bunch No.,
        turn No.] quantities in order [x, y]
    """
    esrf_data = loadmat(infile)  # accepts str or pathlib.Path
    hor, ver = esrf_data["allx"], esrf_data["allz"]
    if hor.shape != ver.shape:
        raise ValueError("Number of turns, BPMs or measurements in X and Y do not match")
    # TODO change for tfs file got from accelerator class
    bpm_names = json.load((Path(__file__).parent / BPM_NAMES_FILE).open("r"))  # weird?
    if hor.shape[1] != len(bpm_names):
        raise ValueError("Number of bpms does not match with accelerator class")
    tbt_data = _check_esrf_tbt_data(np.transpose(np.array([hor, ver]), axes=[0, 2, 3, 1]))
    return np.array(bpm_names), tbt_data


def _check_esrf_tbt_data(tbt_data: np.ndarray) -> np.ndarray:
    tbt_data[np.isnan(np.sum(tbt_data, axis=3)), :] = 0.0
    # check if contains the same data as in previous kick
    mask_prev = (
        np.concatenate(
            (
                np.ones((tbt_data.shape[0], tbt_data.shape[1], 1)),
                np.sum(np.abs(np.diff(tbt_data, axis=2)), axis=3),
            ),
            axis=2,
        )
        == 0.0
    )
    tbt_data[mask_prev, :] = 0.0
    return tbt_data
