# --------------------------------------------------------------------
# INFORMATION:    This file has been generated automatically.
#.....................................................................
# creation time:  Thu Mar  7 16:24:08 2019

# --------------------------------------------------------------------

"""
Module coupling.nw
------------------------------------

@author: awegsche

@version: 1.0

Coupling calculations

"""

from numpy import cos, tan, exp
import tfs
from utils import logging_tools, stats
from numpy import sqrt

LOGGER = logging_tools.get_logger(__name__)

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
    model = meas_input.accelerator.get_model_tfs()
    elements = meas_input.accelerator.get_elements_tfs()
    
    a_real_01 = hor_df["AMP01"]
    b_real_10 = ver_df["AMP10"]
    
    phi_01_i = hor_df["FREQ01"]
    psi_10_i = ver_df["FREQ10"]
    
    a_real_01_paired = _find_pair([a_real_01, phase_advances[0,:]], phase_advances_x["MEAS"])
    b_real_10_paired = _find_pair([b_real_10, phase_advances[0,:]], phase_advances_y["MEAS"])
    
    A01 = abs((1 - 1j*tan(Delta)) * a_real_01 * exp(1j * phi01_i)
              - 1j/cos(Delta) * a_real_01_paired * exp(1j * phi01_j))
    
    B10 = abs((1 - 1j*tan(Delta)) * b_real_10 * exp(1j * psi10_i)
              - 1j/cos(Delta) * b_real_10_paired * exp(1j * psi10_j))
    f1001 = .5 * sqrt(A01*B10)
    f1010 = .5 * sqrt(A0_1*B_10)
    
def _find_pair(columns, phases):
    """ finds the best candidate for momentum reconstruction

    The calculations will be done in a minimum number of lines. Many FLOPs are thrown out of the window

    Args:
      columns (list of columns): the columns we are interested in
      phases (matrix): phase advance matrix
    """
    masked_phases = sum(numpy.diag(numpy.diag(phases,k),k) for k in range(1,CUTOFF))
    return [column[phases.index[abs(masked_phases-0.5).idxmin()]] for column in columns]

