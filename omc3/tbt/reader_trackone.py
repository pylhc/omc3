"""
Trackone Turn-by-Turn Data Handler
-----------------------------------

Tbt data handling from PTC trackone.

"""
from collections import OrderedDict
import numpy as np
from tbt import handler


def read_tbt(infile):
    nturns, npart = get_trackone_stats(infile)
    names, matrix = get_structure_from_trackone(nturns, npart, infile)
    # matrix[0, 2] contains just (x, y) samples.
    return handler.numpy_to_tbts(names, matrix[[0, 2]])


def load_dict(file_name):  # check length?
    loaded = np.load(file_name)
    return loaded[loaded.files[0]].item()  # np.ndarray.item()


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
    with open('stats.txt', "w") as stats_file:
        stats_file.write(stats_string)
    return nturns - 1, nparticles


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




