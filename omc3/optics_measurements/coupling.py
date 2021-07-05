"""
Module coupling.nw
------------------------------------

@author: awegsche

@version: 1.0

Coupling calculations

"""

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
from omc3.definitions.constants import PLANES, PI2I, PI2
from omc3.harpy.constants import COL_AMP, COL_MU, COL_PHASE, COL_TUNE

LOG = logging_tools.get_logger(__name__)

CUTOFF = 5

# --------------------------------------------------------------------------------------------------
# ---- main part -----------------------------------------------------------------------------------


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
    # say hello
    LOG.info("calculating coupling -fffe")

    # intersect measurements
    compensation = 'uncompensated' if meas_input.compensation == 'model' else 'free'
    joined = _joined_frames(input_files)[0]
    joined_index = joined.index.intersection(phase_dict['X'][compensation]['MEAS'].index)  #shouldn't be necessary in the end
    joined = joined.loc[joined_index]

    phases_x = phase_dict['X'][compensation]["MEAS"].loc[joined_index]
    phases_y = phase_dict['Y'][compensation]["MEAS"].loc[joined_index]

    pairs_x, deltas_x = _find_pair(phases_x)
    pairs_y, deltas_y = _find_pair(phases_y)

    A01 = .5*_get_complex(
        joined["AMP01_X"].values*exp(joined["PHASE01_X"].values * PI2I), deltas_x, pairs_x
    )
    B10 = .5 * _get_complex(
        joined["AMP10_Y"].values*exp(joined["PHASE10_Y"].values * PI2I), deltas_y, pairs_y
    )
    A0_1 = .5*_get_complex(
        joined["AMP01_X"].values*exp(-joined["PHASE01_X"].values * PI2I), deltas_x, pairs_x
    )
    B_10 = .5 * _get_complex(
        joined["AMP10_Y"].values*exp(-joined["PHASE10_Y"].values * PI2I), deltas_y, pairs_y
    )

    q1001_from_A = np.angle(A01) - (joined[f"{COL_MU}Y"].to_numpy() - 0.25) * PI2
    q1001_from_B = np.angle(B10) - (joined[f"{COL_MU}X"].to_numpy() + 0.25) * PI2
    q1010_from_A = np.angle(A0_1) + (joined[f"{COL_MU}X"].to_numpy() + 0.25) * PI2
    q1010_from_B = np.angle(B_10) + (joined[f"{COL_MU}Y"].to_numpy() - 0.25) * PI2

    print(f"q1001_from_A.shape = {q1001_from_A.shape}")
    print(f"A01.shape = {A01.shape}")
    for i in range(len(q1001_from_A)):
        q1001_i = np.angle(A01[i]) - (joined[f"{COL_MU}X"].to_numpy()[i] - 0.25) * PI2
        print(f"{q1001_from_A[i]} = {q1001_i} = {np.angle(A01[i])} - {joined['MUX'].to_numpy()[i]} - 0.25")

    f1001 = .5 * sqrt(np.abs(A01 * B10))*exp(1.0j * q1001_from_A)
    f1010 = .5 * sqrt(np.abs(A0_1 * B_10))*exp(1.0j * q1010_from_A)

    LOG.debug("f1001 = {}".format(f1001))
    one_over_N = 1 / len(f1001)
    tune_sep = (tune_dict["X"]["QFM"] % 1.0 - tune_dict["Y"]["QFM"] % 1.0)

    # old Cminus
    C_old = 4.0 * tune_sep * np.mean(np.abs(f1001))
    header_dict["OldCminus"] = C_old
    LOG.info(f"abs OldCminus = {C_old}, tune_sep = {tune_sep}")

    # new Cminus
    C_new = np.abs(4.0 * tune_sep * np.mean(f1001 * exp(1.0j * (joined[f"{COL_MU}X"] - joined[f"{COL_MU}Y"]))))
    header_dict["newCminus"] = C_new
    LOG.info(f"abs NewCminus = {C_new}")

    q1001_from_A = (q1001_from_A/PI2) % 1.0
    q1001_from_B = (q1001_from_B/PI2) % 1.0
    q1010_from_A = (q1010_from_A/PI2) % 1.0
    q1010_from_B = (q1010_from_B/PI2) % 1.0

    if meas_input.compensation == "model":
        f1001, f1010 =  compensate_model(f1001, f1010, tune_dict)
    rdt_df = pd.DataFrame(index=joined_index,
                          columns=["S", "F1001R", "F1010R", "F1001I", "F1010I", "q1001"],
                          data=np.array([
                              meas_input.accelerator.model["S"].values[pairs_x],
                              np.real(f1001), np.real(f1010),
                              np.imag(f1001), np.imag(f1010),
                              q1001_from_A,
                          ]).transpose())

    tfs.write(os.path.join(meas_input.outputdir, "coupling.tfs"),
              rdt_df, header_dict, save_index="NAME")


def compensate_model(f1001, f1010, tune_dict):
    """
    Compensation by model only.

    Args:
        df (DataFrame): the pre-calculated driven coupling RDTs
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
        for df, plane in zip((linx, liny), PLANES):
            rename_cols(df, plane, ['NAME', 'S', f'{COL_TUNE}{plane}', f'{COL_MU}{plane}',
                                    f'{COL_MU}{plane}SYNC'])
        merged_df = pd.merge(left=linx,
                             right=liny,
                             on=['NAME', 'S'],
                             how='inner',
                             sort=False,
                             suffixes=(False, f"__{i}")
                            ).set_index('NAME')
        merged_df.index.name='NAME'
        joined_dfs.append(merged_df)
    return joined_dfs


def rename_cols(df, suffix, exceptions=['']):
    df.columns = [f'{col}_{suffix}' if col not in exceptions else col for col in df.columns]
    df.index.name = None
    return df
