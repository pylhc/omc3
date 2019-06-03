"""
 :module: twiss_to_lin

 Created on 18/02/18

 :author: Lukas Malina


This module generates the .linx/y (both on-momentum and off-momentum) from two twiss files,
for free motion and for driven motion. The twisses should contain the chromatic functions as well.

"""
from collections import OrderedDict
import numpy as np
import pandas as pd
from os.path import join
from . import context
import tfs
PLANES = ('X', 'Y')
DRIVEN = "_d"
FREE = "_f"
PLANE_TO_NUM = dict(X=1, Y=2)
COUP = dict(X="01", Y="10")
OTHER = dict(X="Y", Y="X")
M_TO_MM = 1000
NOISE=0.1
NTURNS=6600
ACTION_UM = 0.005
ERRTUNE = 3e-7
NAT_OVER_DRV = 0.01
MAGIC_NUMBER = 6   # SVD cleaning effect + main lobe size effect
COUPLING = 0.1

def optics_measurement_test_files(modeldir, dpps):
    """

    Args:
        modeldir: path to model directory
        dpps: list of required dpp values for simulated lin files

    Returns:

    """
    model, tune, nattune = get_combined_model_and_tunes(modeldir)
    lins=[]
    for dpp_value in dpps:
        lins.append(generate_lin_files(model, tune, nattune, dpp=dpp_value))
    return lins


def generate_lin_files(model, tune, nattune, dpp=0.0):
    nbpms = len(model.index.values)

    lins = {}
    for plane in PLANES:
        lin = model.loc[:, ['NAME', 'S']]
        noise_freq_domain = NOISE / np.sqrt(NTURNS) / MAGIC_NUMBER
        lin['NOISE'] = noise_freq_domain
        lin['CO'] = dpp * M_TO_MM * model.loc[:, f"D{plane}{DRIVEN}"] + np.random.randn(nbpms) * (NOISE / np.sqrt(NTURNS))
        lin['CORMS'] = np.abs(np.random.randn(nbpms) * 0.003 + 0.003)  # TODO
        lin['PK2PK'] = 2 * (np.sqrt(model.loc[:, f"BET{plane}{DRIVEN}"] * ACTION_UM) + 3 * NOISE)
        lin[f"TUNE{plane}"] = tune[plane] + ERRTUNE * np.random.randn(nbpms)
        lin[f"NATTUNE{plane}"] = nattune[plane] + (ERRTUNE / np.sqrt(NAT_OVER_DRV)) * np.random.randn(nbpms)
        lin[f"MU{plane}"] = np.remainder(model.loc[:, f"MU{plane}{DRIVEN}"]
                                         + dpp * model.loc[:, f"DMU{plane}{DRIVEN}"]
                                         + (noise_freq_domain / (2 * np.pi)) *np.random.randn(nbpms) + np.random.rand(), 1)
        lin[f"ERRMU{plane}"] = noise_freq_domain / (2 * np.pi)
        lin[f"AMP{plane}"] = np.sqrt(model.loc[:, f"BET{plane}{DRIVEN}"] * ACTION_UM *
                                     (1 + dpp * np.sin(2 * np.pi * model.loc[:, f"PHI{plane}{DRIVEN}"])
                                      * model.loc[:, f"W{plane}{DRIVEN}"])) + noise_freq_domain * np.random.randn(nbpms)

        lin[f"NATMU{plane}"] = np.remainder(model.loc[:, f"MU{plane}{FREE}"]
                                            + (NAT_OVER_DRV * noise_freq_domain / (2 * np.pi)) * np.random.randn(nbpms) + np.random.rand(), 1)
        lin[f"NATAMP{plane}"] = NAT_OVER_DRV * np.sqrt(ACTION_UM * model.loc[:, f"BET{plane}{FREE}"]) + noise_freq_domain * np.random.randn(nbpms)

        lin[f"PHASE{COUP[plane]}"] = np.remainder(model.loc[:, f"MU{OTHER[plane]}{DRIVEN}"] + dpp * model.loc[:, f"DMU{OTHER[plane]}{DRIVEN}"]
                                                  + (COUPLING * noise_freq_domain / (2 * np.pi)) * np.random.randn(nbpms) + np.random.rand(), 1)
        lin[f"AMP{COUP[plane]}"] = COUPLING * np.sqrt(ACTION_UM *model.loc[:, f"BET{OTHER[plane]}{DRIVEN}"]
                                                  * (1 + dpp * np.sin(model.loc[:, f"PHI{OTHER[plane]}{DRIVEN}"]) * model.loc[:, f"W{OTHER[plane]}{DRIVEN}"])) + COUPLING * noise_freq_domain * np.random.randn(nbpms)

        # backwards compatibility with drive  TODO remove
        lin[f"AMP{plane}"] = lin.loc[:, f"AMP{plane}"].values / 2
        lin[f"NATAMP{plane}"] = lin.loc[:, f"NATAMP{plane}"].values / 2

        lins[plane] = tfs.TfsDataFrame(lin, headers=_get_header(tune, nattune, plane)).set_index("NAME")
    return lins


def get_combined_model_and_tunes(model_dir):
    free = tfs.read(join(model_dir, 'twiss.dat'))
    driven = tfs.read(join(model_dir, 'twiss_ac.dat'))
    nattune = {"X": np.remainder(free.headers['Q1'], 1), "Y": np.remainder(free.headers['Q2'], 1)}
    tune = {"X": np.remainder(driven.headers['Q1'], 1), "Y": np.remainder(driven.headers['Q2'], 1)}
    model = pd.merge(free, driven, how='inner', on='NAME', suffixes=(FREE, DRIVEN))
    model['S'] = model.loc[:, 'S_f']
    return model, tune, nattune


def _get_header(tunes, nattunes, plane):
    header = OrderedDict()
    header[f"Q{PLANE_TO_NUM[plane]}"] = tunes[plane]
    header[f"Q{PLANE_TO_NUM[plane]}RMS"] = 1e-7
    header[f"NATQ{PLANE_TO_NUM[plane]}"] = nattunes[plane]
    header[f"NATQ{PLANE_TO_NUM[plane]}RMS"] = 1e-6
    return header
