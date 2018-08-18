from os.path import abspath, join, dirname
import json
from collections import OrderedDict
import numpy as np
from numpy import savez_compressed as _save
from numpy import load as _load
import pandas as pd
from scipy.io import loadmat
from sdds_files import turn_by_turn_writer


# Introduce a system for lists(dicts) of TbT files, trackones ... ,
#  what is utils/dict_tools.py - Josch?
def trackone_to_sdds(nturns=0, npart=0,
                     infile='trackone', outfile="trackone.sdds"):
    names, matrix = get_structure_from_trackone(nturns, npart, infile)
    # matrix[0, 2] contains just (x, y) samples.
    turn_by_turn_writer.write_tbt_file(names, matrix[[0, 2]], outfile)


def save_dict(file_name, di):
    _save(file_name, di)


def load_dict(file_name):  # check length?
    loaded = _load(file_name)
    return loaded[loaded.files[0]].item()  # np.ndarray.item()


def save_df(file_name, df):
    save_dict(file_name, df_to_dict(df))


def load_df(file_name):
    di = load_dict(file_name)
    return dict_to_df(di)


def dict_to_df(di):  # should contain at least 'data': np.array(2D)
    return pd.DataFrame(data=di.get('data'), index=di.setdefault('index'),
                        columns=di.setdefault('columns'))


def df_to_dict(df):
    return {'data': df.as_matrix().values, 'index': df.index.values, 'columns': df.columns.values}


def save_tbt_files(file_name, *tbts):
    save_dict(file_name, *tbts)


def get_trackone_stats(infile):
    stats_string = ""
    nturns = 0
    nparticles = 0
    first_seg = True
    with open(infile, 'r') as f:
        for l in f:
            if len(l.strip()) == 0:
                continue
            if l.strip()[0] in ['@', '*', '$']:
                stats_string = stats_string + l
                continue
            parts = l.split()
            if parts[0] == '#segment':
                if first_seg:
                    nturns = int(parts[2])
                    nparticles = int(parts[3])
                    first_seg = False
                    stats_string = stats_string + l
                else:
                    break
            else:
                stats_string = stats_string + l
    stats_file = open('stats.txt', "w")
    stats_file.write(stats_string)
    stats_file.close()
    return nturns, nparticles


def get_structure_from_trackone(nturns=0, npart=0, infile='trackone'):
    """
    Reads the trackone file produced by PTC

    Attributes:
        nturns: Number of turns tracked in the trackone, i.e. obtained from get_trackone_stats()
        npart:  Number of particles tracked in the trackone, i.e. obtained from get_trackone_stats()
        infile: path to trackone file to be read
    Returns:
        Numpy array of BPM names
        4D Numpy array [quantity, BPM, particle/bunch No., turn No.]
        quantities in order [x, px, y, py, t, pt, s, E]
    """
    bpms = OrderedDict()
    with open(infile, 'r') as f:
        for l in f:
            if len(l.strip()) == 0:
                continue
            if l.strip()[0] in ['@', '*', '$']:
                continue
            parts = l.split()
            if parts[0] == '#segment':
                bpm_name = parts[-1].upper()
                if ('BPM' in bpm_name) and (bpm_name not in bpms.keys()):
                    bpms[bpm_name] = np.empty([npart, nturns, 8], dtype=float)
            elif 'BPM' in bpm_name:
                bpms[bpm_name][int(parts[0]) - 1, int(parts[1]) - 1, :] = np.array(parts[2:])
    return np.array(bpms.keys()), np.transpose(np.array(bpms.values()), axes=[3, 0, 1, 2])


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
    if hor.shape[0] != ver.shape[0]:
        raise ValueError("Number of turns in x and y do not match")
    if hor.shape[2] != ver.shape[2]:
        raise ValueError("Number of measurements in x and y do not match")
    # TODO change for tfs file got from accelerator class
    bpm_names = json.load(open(abspath(join(dirname(__file__), "bpm_names.json")), "r"))
    if hor.shape[1] == len(bpm_names) == ver.shape[1]:
        tbt_data = _check_esrf_tbt_data(np.transpose(np.array([hor, ver]), axes=[0, 2, 3, 1]))
        return np.array(bpm_names), tbt_data
    raise ValueError("Number of bpms does not match with accelerator class")


def _check_esrf_tbt_data(tbt_data):
    tbt_data[np.isnan(np.sum(tbt_data, axis=3)), :] = 0.0
    # check if contains the same data as in previous kick
    mask_prev = np.concatenate((np.ones((tbt_data.shape[0], tbt_data.shape[1], 1)),
                                np.sum(np.abs(np.diff(tbt_data, axis=2)), axis=3)), axis=2) == 0.0
    tbt_data[mask_prev, :] = 0.0
    return tbt_data
