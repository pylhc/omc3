"""
trackone
----------------------

Tbt data handling from PTC trackone.

"""
from collections import OrderedDict
import numpy as np
import pandas as pd
from tbt.handler import write_tbt, TbtData

# Introduce a system for lists(dicts) of TbT files, trackones ... ,


def trackone_to_sdds(infile, outfile, nturns=None, npart=None):
    if nturns is None or npart is None:
        nturns, npart = get_trackone_stats(infile)
    names, matrix = get_structure_from_trackone(nturns, npart, infile)
    # matrix[0, 2] contains just (x, y) samples.
    tbt_data = numpy_to_tbts(names, matrix[[0, 2]])
    write_tbt(outfile, tbt_data)


def save_dict(file_name, di):
    np.savez_compressed(file_name, di)


def load_dict(file_name):  # check length?
    loaded = np.load(file_name)
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
    return {'data': df.values, 'index': df.index.values, 'columns': df.columns.values}


def save_tbt_files(file_name, *tbts):
    save_dict(file_name, *tbts)


def get_trackone_stats(infile):
    stats_string = ""
    nturns, nparticles = 0, 0
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
                if not first_seg:
                    break
                nturns = int(parts[2])
                nparticles = int(parts[3])
                first_seg = False
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
    return np.array(list(bpms.keys())), np.transpose(np.array(list(bpms.values())), axes=[3, 0, 1, 2])


def numpy_to_tbts(names, matrix):
    """Converts turn by turn data and names into TbTData.

    Arguments:
        names: Numpy array of BPM names
        matrix: 4D Numpy array [quantity, BPM, particle/bunch No., turn No.]
            quantities in order [x, y]
    """
    # get list of TbTFile from 4D matrix ...
    _, nbpms, nbunches, nturns = matrix.shape
    matrices = []
    indices = []
    for index in range(nbunches):
        matrices.append({"X": pd.DataFrame(index=names, data=matrix[0, :, index, :]),
                         "Y": pd.DataFrame(index=names, data=matrix[1, :, index, :])})
        indices.append(index)
    return TbtData(matrices, None, np.array(indices), nturns)

