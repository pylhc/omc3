"""
ESRF Turn-by-Turn Data Handler
--------------------------------

Data handling for tbt data from ESRF.

"""
from os.path import abspath, join, dirname
import json
from scipy.io import loadmat
import numpy as np
from tbt import handler


def read_tbt(filepath):
    """
    Reads ESRF matlab file.

    Args:
        filepath: path to a file

    Returns:
        tbt.TbTData object
    """
    names, matrix = load_esrf_mat_file(filepath)
    return handler.numpy_to_tbts(names, matrix)


def load_esrf_mat_file(infile):
    """
    Reads the ESRF TbT Matlab file, checks for nans and data duplicities from consecutive kicks

    Attributes:
        infile: path to file to be read
    Returns:
        Numpy array of BPM names
        4D Numpy array [quantity, BPM, particle/bunch No., turn No.]
        quantities in order [x, y]
        """
    esrf_data = loadmat(infile)
    hor, ver = esrf_data["allx"], esrf_data["allz"]
    if hor.shape != ver.shape:
        raise ValueError("Number of turns, BPMs or measurements in X and Y do not match")
    # TODO change for tfs file got from accelerator class
    bpm_names = json.load(open(abspath(join(dirname(__file__), "bpm_names.json")), "r"))
    if hor.shape[1] != len(bpm_names):
        raise ValueError("Number of bpms does not match with accelerator class")
    tbt_data = _check_esrf_tbt_data(np.transpose(np.array([hor, ver]), axes=[0, 2, 3, 1]))
    return np.array(bpm_names), tbt_data


def _check_esrf_tbt_data(tbt_data):
    tbt_data[np.isnan(np.sum(tbt_data, axis=3)), :] = 0.0
    # check if contains the same data as in previous kick
    mask_prev = np.concatenate((np.ones((tbt_data.shape[0], tbt_data.shape[1], 1)),
                                np.sum(np.abs(np.diff(tbt_data, axis=2)), axis=3)), axis=2) == 0.0
    tbt_data[mask_prev, :] = 0.0
    return tbt_data
