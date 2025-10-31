"""
 :module: twiss_to_lin

 Created on 18/02/18

 :author: Lukas Malina


This module generates the .linx/y (both on-momentum and off-momentum) from two twiss files,
for free motion and for driven motion. The twisses should contain the chromatic functions as well.

"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tfs

from omc3.definitions import formats
from omc3.definitions.constants import PLANES
from omc3.harpy.constants import (
    COL_AMP,
    COL_CO,
    COL_CORMS,
    COL_ERR,
    COL_MU,
    COL_NAME,
    COL_NATAMP,
    COL_NATMU,
    COL_NATTUNE,
    COL_NOISE,
    COL_PHASE,
    COL_PK2PK,
    COL_S,
    COL_TIME,
    COL_TUNE,
    MAINLINE_UNIT,
)
from omc3.optics_measurements.constants import NAT_TUNE, TUNE

MOTION = {"free": "_f", "driven": "_d"}
PLANE_TO_NUM = {"X": 1, "Y": 2}
COUPLING_INDICES = {"X": "01", "Y": "10"}
OTHER = {"X": "Y", "Y": "X"}
NOISE = 1e-4
NTURNS = 6600
ACTION = 5e-9
ERRTUNE = 3e-7
NAT_OVER_DRV = 0.01
MAGIC_NUMBER = 6  # SVD cleaning effect + main lobe size effect
COUPLING = 0.1


def optics_measurement_test_files(modeldir, dpps, motion, beam_direction):
    """

    Args:
        modeldir: path to model directory
        dpps: list of required dpp values for simulated lin files

    Returns:

    """
    if beam_direction not in (-1, 1):
        raise ValueError("Beam direction has to be either 1 or -1")
    model, tune, nattune = get_combined_model_and_tunes(modeldir)
    lins = []
    for dpp_value in dpps:
        lins.append(
            generate_lin_files(
                model, tune, nattune, MOTION[motion], dpp=dpp_value, beam_direction=beam_direction
            )
        )
    return lins


def generate_lin_files(model, tune, nattune, motion="_d", dpp=0.0, beam_direction=1):
    nbpms = len(model.index.to_numpy())
    lins = {}
    for plane in PLANES:
        lin = model.loc[:, [COL_NAME, COL_S]]
        noise_freq_domain = NOISE / np.sqrt(NTURNS) / MAGIC_NUMBER
        lin[COL_NOISE] = noise_freq_domain
        lin[COL_CO] = dpp * model.loc[:, f"D{plane}{motion}"] + np.random.randn(nbpms) * (
            NOISE / np.sqrt(NTURNS)
        )
        lin[COL_CORMS] = np.abs(np.random.randn(nbpms) * 3e-6 + 3e-6)  # TODO
        lin[COL_PK2PK] = 2 * (np.sqrt(model.loc[:, f"BET{plane}{motion}"] * ACTION) + 3 * NOISE)
        lin[f"{COL_TUNE}{plane}"] = tune[plane] + ERRTUNE * np.random.randn(nbpms)
        lin[f"{COL_NATTUNE}{plane}"] = nattune[plane] + (
            ERRTUNE / np.sqrt(NAT_OVER_DRV)
        ) * np.random.randn(nbpms)
        lin[f"{COL_MU}{plane}"] = (
            np.remainder(
                model.loc[:, f"MU{plane}{motion}"]
                + dpp * model.loc[:, f"DMU{plane}{motion}"]
                + (noise_freq_domain / (2 * np.pi)) * np.random.randn(nbpms)
                + np.random.rand(),
                1,
            )
            * beam_direction
        )
        lin[f"{COL_ERR}{COL_MU}{plane}"] = noise_freq_domain / (2 * np.pi)

        lin[f"{COL_ERR}{COL_AMP}{plane}"] = noise_freq_domain * np.random.randn(nbpms)

        lin[f"{COL_AMP}{plane}"] = (
            np.sqrt(
                model.loc[:, f"BET{plane}{motion}"]
                * ACTION
                * (
                    1
                    + dpp
                    * np.sin(2 * np.pi * model.loc[:, f"PHI{plane}{motion}"])
                    * model.loc[:, f"W{plane}{motion}"]
                )
            )
            + lin[f"{COL_ERR}{COL_AMP}{plane}"]
        )

        lin[f"{COL_ERR}{COL_AMP}{plane}"] = np.abs(
            lin[f"{COL_ERR}{COL_AMP}{plane}"]
        )  # * 2  divided by two removed?

        lin[f"{COL_NATMU}{plane}"] = (
            np.remainder(
                model.loc[:, f"MU{plane}{MOTION['free']}"]
                + (NAT_OVER_DRV * noise_freq_domain / (2 * np.pi)) * np.random.randn(nbpms)
                + np.random.rand(),
                1,
            )
            * beam_direction
        )

        lin[f"{COL_ERR}{COL_NATAMP}{plane}"] = noise_freq_domain * np.random.randn(nbpms)

        lin[f"{COL_NATAMP}{plane}"] = (
            NAT_OVER_DRV * np.sqrt(ACTION * model.loc[:, f"BET{plane}{MOTION['free']}"])
            + lin[f"{COL_ERR}{COL_NATAMP}{plane}"]
        )

        lin[f"{COL_ERR}{COL_NATAMP}{plane}"] = np.abs(
            lin[f"{COL_ERR}{COL_NATAMP}{plane}"]
        )  # * 2  divided by two removed?

        lin[f"{COL_PHASE}{COUPLING_INDICES[plane]}"] = (
            np.remainder(
                model.loc[:, f"MU{OTHER[plane]}{motion}"]
                + dpp * model.loc[:, f"DMU{OTHER[plane]}{motion}"]
                + (COUPLING * noise_freq_domain / (2 * np.pi)) * np.random.randn(nbpms)
                + np.random.rand(),
                1,
            )
            * beam_direction
        )
        lin[f"{COL_AMP}{COUPLING_INDICES[plane]}"] = COUPLING * np.sqrt(
            ACTION
            * model.loc[:, f"BET{OTHER[plane]}{motion}"]
            * (
                1
                + dpp
                * np.sin(model.loc[:, f"PHI{OTHER[plane]}{motion}"])
                * model.loc[:, f"W{OTHER[plane]}{motion}"]
            )
        ) + COUPLING * noise_freq_domain * np.random.randn(nbpms)

        lins[plane] = tfs.TfsDataFrame(lin, headers=_get_header(tune, nattune, plane)).set_index(
            COL_NAME, drop=False
        )
    return lins


def get_combined_model_and_tunes(model_dir: Path):
    free = tfs.read(model_dir / "twiss.dat")
    driven = tfs.read(model_dir / "twiss_ac.dat")
    nattune = {"X": np.remainder(free.headers["Q1"], 1), "Y": np.remainder(free.headers["Q2"], 1)}
    tune = {"X": np.remainder(driven.headers["Q1"], 1), "Y": np.remainder(driven.headers["Q2"], 1)}
    model = pd.merge(free, driven, how="inner", on=COL_NAME, suffixes=MOTION.values())
    model[COL_S] = model.loc[:, "S_f"]
    return model, tune, nattune


def _get_header(tunes, nattunes, plane):
    return {
        f"{TUNE}{PLANE_TO_NUM[plane]}": tunes[plane],
        f"{TUNE}{PLANE_TO_NUM[plane]}RMS": 1e-7,
        f"{NAT_TUNE}{PLANE_TO_NUM[plane]}": nattunes[plane],
        f"{NAT_TUNE}{PLANE_TO_NUM[plane]}RMS": 1e-6,
        COL_TIME: datetime.now(timezone.utc).strftime(formats.TIME),
        MAINLINE_UNIT: "m",
    }
