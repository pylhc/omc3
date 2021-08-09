"""
Module coupling.nw
------------------------------------

@author: awegsche

@version: 1.0

Coupling calculations

"""

from functools import reduce
from omc3.optics_measurements.beta_from_phase import _tilt_slice_matrix
from collections import namedtuple
import numpy as np
from numpy import exp, tan, cos, sin
import os
import sys
import tfs
import pandas as pd
from numpy import conj
from omc3.utils import logging_tools, stats
from numpy import sqrt
from omc3.definitions.constants import PI2I, PI2
from omc3.harpy.constants import COL_MU
from optics_functions.coupling import coupling_via_cmatrix
from pathlib import Path


# column name constants
COL_AMPX_SEC = "AMP01_X"     # amplitude of secondary line in horizontal spectrum
COL_AMPY_SEC = "AMP10_Y"     # amplitude of secondary line in vertical spectrum
COL_FREQX_SEC = "PHASE01_X"  # frequency of secondary line in horizontal spectrum
COL_FREQY_SEC = "PHASE10_Y"  # frequency of secondary line in vertical spectrum

COLS_TO_KEEP_X = ["NAME", "S", "AMP01", "PHASE01", "MUX"]
COLS_TO_KEEP_Y = ["NAME", "S", "AMP10", "PHASE10", "MUY"]

LOG = logging_tools.get_logger(__name__)

CUTOFF = 5

# --------------------------------------------------------------------------------------------------
# ---- main part -----------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

def calculate_coupling(meas_input, input_files, phase_dict, tune_dict, header_dict):
    """
       Calculates the coupling

    Args:
      meas_input (OpticsInput): programm arguments
      input_files (TfsDataFrames): sdds input files
      phase_dict (PhaseDict): contains measured phase advances
      tune_dict (TuneDict): contains measured tunes
      header_dict (dict): dictionary of header items common for all output files

    """
    LOG.info("calculating coupling")

    # intersect measurements
    compensation = 'uncompensated' if meas_input.compensation == 'model' else 'free'
    joined = _joined_frames(input_files)
    joined_index = joined.index \
        .intersection(phase_dict['X'][compensation]["MEAS"].index) \
        .intersection(phase_dict['Y'][compensation]["MEAS"].index)
    joined = joined.loc[joined_index]

    phases_x = phase_dict['X'][compensation]["MEAS"].loc[joined_index]
    phases_y = phase_dict['Y'][compensation]["MEAS"].loc[joined_index]

    # averaging
    for col in [COL_AMPX_SEC, COL_AMPY_SEC]:
        cols = [c for c in joined if c.startswith(col)]
        joined[col] = stats.weighted_mean(joined[cols], axis=1)
    for col in [COL_FREQX_SEC, COL_FREQY_SEC]:
        cols = [x for x in joined if x.startswith(col)]
        joined[col] = stats.circular_mean(joined[cols], axis=1)

    pairs_x, deltas_x = _find_pair(phases_x)
    pairs_y, deltas_y = _find_pair(phases_y)

    A01 = .5*_get_complex(
        joined[COL_AMPX_SEC].values*exp(joined[COL_FREQX_SEC].values * PI2I), deltas_x, pairs_x
    )
    B10 = .5 * _get_complex(
        joined[COL_AMPY_SEC].values*exp(joined[COL_FREQY_SEC].values * PI2I), deltas_y, pairs_y
    )
    A0_1 = .5*_get_complex(
        joined[COL_AMPX_SEC].values*exp(-joined[COL_FREQX_SEC].values * PI2I), deltas_x, pairs_x
    )
    B_10 = .5 * _get_complex(
        joined[COL_AMPY_SEC].values*exp(-joined[COL_FREQY_SEC].values * PI2I), deltas_y, pairs_y
    )

    q1001_from_A = -np.angle(A01)  + (joined[f"{COL_MU}Y"].to_numpy() - 0.25) * PI2
    q1001_from_B = np.angle(B10) - (joined[f"{COL_MU}X"].to_numpy() - 0.25) * PI2

    q1010_from_A = -np.angle(A0_1) - (joined[f"{COL_MU}Y"].to_numpy() + 0.25) * PI2
    q1010_from_B = -np.angle(B_10) - (joined[f"{COL_MU}X"].to_numpy() + 0.25) * PI2

    f1001 = -.5 * sqrt(np.abs(A01 * B10))  *0.5*(exp(1.0j * q1001_from_A) + exp(1.0j * q1001_from_B))
    f1010 = .5 * sqrt(np.abs(A0_1 * B_10))*0.5*(exp(1.0j * q1010_from_A) + exp(1.0j * q1010_from_B))

    LOG.debug("f1001 = {}".format(f1001))
    tune_sep = np.abs(tune_dict["X"]["QFM"] % 1.0 - tune_dict["Y"]["QFM"] % 1.0)

    # old Cminus
    C_old = 4.0 * tune_sep * np.mean(np.abs(f1001))
    header_dict["OldCminus"] = C_old
    LOG.info(f"abs OldCminus = {C_old}, tune_sep = {tune_sep}")

    # new Cminus
    C_new = np.abs(4.0 * tune_sep * np.mean(f1001 * exp(1.0j * (joined[f"{COL_MU}X"] - joined[f"{COL_MU}Y"]))))
    header_dict["newCminus"] = C_new
    LOG.info(f"abs NewCminus = {C_new}")

    if meas_input.compensation == "model":
        f1001, f1010 =  compensate_model(f1001, f1010, tune_dict)
    rdt_df = pd.DataFrame(index=joined_index,
                          columns=["S", "F1001R", "F1010R", "F1001I", "F1010I", "F1001W", "F1010W"],
                          data=np.array([
                              meas_input.accelerator.model["S"].values[pairs_x],
                              np.real(f1001), np.real(f1010),
                              np.imag(f1001), np.imag(f1010),
                              np.abs(f1001), np.abs(f1010),
                          ]).transpose())

    rdt_df.sort_values(by="S", inplace=True)

    # adding model values and deltas
    model_coupling = coupling_via_cmatrix(meas_input.accelerator.model).loc[rdt_df.index]
    RDTCOLS = ["F1001", "F1010"]
    for (domain, func) in [("I", np.imag),
                           ("R", np.real),
                           ("W", np.abs)]:
        for col in RDTCOLS:
            mdlcol = func(model_coupling[col])
            rdt_df[f"{col}{domain}MDL"] = mdlcol
            rdt_df[f"DELTA{col}{domain}"] = rdt_df[f"{col}{domain}"] - mdlcol
            rdt_df[f"ERRDELTA{col}{domain}"] = 0.0

    _write_coupling_tfs(rdt_df, meas_input.outputdir, header_dict)


def _write_coupling_tfs(rdt_df, outdir, header_dict):
    common_cols = ["S"]
    cols_to_print_f1001 = common_cols + [col for col in rdt_df.columns if "1001" in col]
    cols_to_print_f1010 = common_cols + [col for col in rdt_df.columns if "1010" in col]

    tfs.write(Path(outdir) / "f1001.tfs", rdt_df[cols_to_print_f1001], header_dict, save_index="NAME")
    tfs.write(Path(outdir) / "f1010.tfs", rdt_df[cols_to_print_f1010], header_dict, save_index="NAME")


def compensate_model(f1001, f1010, tune_dict):
    """
    Compensation by model only.

    Args:
        f1001, f1010: the pre-calculated driven coupling RDTs
        tune_dict (TuneDict): the free and driven tunes
    """
    Qx = PI2 * tune_dict["X"]["QFM"]  # natural tunes
    Qy = PI2 * tune_dict["Y"]["QFM"]

    dQx = PI2 * tune_dict["X"]["QM"]  # driven tunes
    dQy = PI2 * tune_dict["Y"]["QM"]

    factor1001 = np.sqrt(np.abs(sin(dQy - Qx)*sin(dQx - Qy)))/np.abs(sin(Qx - Qy))
    factor1010 = np.abs(np.sqrt(sin(Qx + dQy)*sin(Qy + dQx))/sin(Qx + Qy))
    f1001 *= factor1001
    f1010 *= factor1010

    LOG.info("compensation by model")
    LOG.info(f"f1001 factor: {factor1001}")
    LOG.info(f"f1010 factor: {factor1010}")

    return f1001, f1010

def compensate_ryoichi():
    pass

# --------------------------------------------------------------------------------------------------
# ---- helper functions ----------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------


def _take_next(phases, shift=1):
    """
    Takes the following BPM for momentum reconstruction by a given shift
    """
    indices = np.roll(np.arange(phases.values.shape[0]), shift)
    return indices, phases.values[np.arange(phases.values.shape[0]), indices] - 0.25


def _find_pair(phases):
    """ finds the best candidate for momentum reconstruction

    Args:
      phases (matrix): phase advance matrix
    """
    slice = _tilt_slice_matrix(phases.values, 0, 2*CUTOFF) - 0.25
    indices = (np.argmin(abs(slice), axis=0))
    deltas = slice[indices, range(len(indices))]
    indices = (indices + np.arange(len(indices))) % len(indices)
    #deltas = [phases[col][indices] for col in phases.columns]

    return np.array(indices), deltas


def _get_complex(spectral_lines, deltas, pairs):
    """
    calculates the complex line from the real lines at positions i and j, where j is determined by
    taking the next BPM with a phase advance sufficiently close to pi/2

    Args:
      spectral_lines (vector): measured (real) spectral lines
      deltas (vector): phase advances minus 90deg
      pairs (vector): indices for pairing
    """
    return (1.0 - 1.0j*tan(PI2*deltas)) * spectral_lines - 1.0j/cos(PI2*deltas) * spectral_lines[pairs]


def _joined_frames(input_files):
    """
    Merges spectrum data from the two planes from all the input files.

    TODO: unify with CRDT
    """
    joined_dfs = []

    assert len(input_files['X']) == len(input_files['Y'])

    for i, (linx, liny) in enumerate(zip(input_files['X'], input_files['Y'])):
        linx = linx[COLS_TO_KEEP_X]
        liny = liny[COLS_TO_KEEP_Y]
        linx.rename(columns=rename_col("X", i), inplace=True)
        liny.rename(columns=rename_col("Y", i), inplace=True)
        merged_df = pd.merge(left=linx,
                             right=liny,
                             on=['NAME', 'S'],
                             how='inner',
                             sort=False,
                            )
        
        joined_dfs.append(merged_df)

    return reduce(lambda a,b: pd.merge(a, b, how='inner', on=['NAME', 'S'],sort=False), joined_dfs).set_index("NAME")


def rename_col(plane, index):
    def fn(column):
        if column in ["NAME", "S", "MUX", "MUY"]:
            return column
        return f"{column}_{plane}_{index}" 

    return fn
