"""
CRDTs
-------------------

:module: optics_measurements.crdt

Computes combined resonance driving terms following the derivatons in https://arxiv.org/pdf/1402.1461.pdf.
"""

from pathlib import Path
import numpy as np
import tfs
import pandas as pd
from omc3.optics_measurements.constants import ERR, EXT, PLANES, AMPLITUDE
from omc3.utils import iotools, logging_tools

LOGGER = logging_tools.get_logger(__name__)
PHASE = 'PHASE'

# ORDER = {
#     "Coupling": {
#         "F_XY": get_Fxy,
#         "F_YX": get_Fyx,
#                 },
#     "Sextupole": {
#         "F_NS3":get_Fns3,
#         "F_NS2":get_Fns2,
#         "F_NS1":get_Fns1,
#         "F_NS0":get_Fns0,
#                  },
#     "Skew Sextupole": {
#         "F_SS3":get_Fss3,
#         "F_SS2":get_Fss2,
#         "F_SS1":get_Fss1,
#         "F_SS0":get_Fss0,
#                       },
#     "Octupole": {
#         "F_NO5":get_Fno5,
#         "F_NO4":get_Fno4,
#         "F_NO3":get_Fno3,
#         "F_NO2":get_Fno2,
#         "F_NO1":get_Fno1,
#         "F_NO0":get_Fno0,
#                 }
# }


def calculate(measure_input, input_files, header):
    """

    Args:
        measure_input:
        input_files:
        header:
    Returns:

    """
    LOGGER.info(f"Start of CRDT analysis")
    
    # CRDT relies on double plane BPMs, selection here
    bpm_names = input_files.bpms(dpp_value=0)
    
    # for order, crdts in ORDER.items():
    #     for crdt in crdts.keys():
    # print(input_files)


def write(df, header, meas_input, order, crdt):
    outputdir = Path(meas_input.outputdir)/"crdt"/order
    iotools.create_dirs(outputdir)
    tfs.write(outputdir/crdt.with_suffix(EXT), df, header, save_index='NAME')


def Aover2B(df, lineA, lineB, phaseA, phaseB):
    df[AMPLITUDE] = lineA['val']/(2*lineB['val'])
    df[f'{ERR}{AMPLITUDE}'] = np.sqrt(lineA['err']**2/(4*lineB['val']**2)
                                      + 0.25 * lineA['val']**2 * np.abs(lineB['err']/lineB['val']**2)**2)
    df[PHASE] = phaseA['val'] - phaseB['val'] - 1.5*np.pi
    df[f'{ERR}{PHASE}'] = np.sqrt(phaseA['err']**2 + phaseB['err']**2)
    return df


def Aover4B(df, lineA, lineB, phaseA, phaseB):
    df[AMPLITUDE] = lineA['val']/(4*lineB['val']**2)
    df[f'{ERR}{AMPLITUDE}'] = np.sqrt(lineA['err']**2/(16*lineB['val']**4)
                                      + 0.25 * lineA['val']**2 * np.abs(lineB['err']/lineB['val']**3)**2)
    df[PHASE] = phaseA['val'] + 2*phaseB['val'] - 1.5*np.pi
    df[f'{ERR}{PHASE}'] = np.sqrt(phaseA['err']**2 + phaseB['err']**2)
    return df


def Aover4BC(df, lineA, lineB, lineC, phaseA, phaseB, phaseC, sign=1):
    df[AMPLITUDE] = lineA['val']/(4*lineB['val']*lineC['val'])
    df[f'{ERR}{AMPLITUDE}'] = np.sqrt(lineA['err']/(16*lineB['val']**2*lineC['val']**2
                                                    + (1/16.)*lineA['val']**2*np.abs(
        np.sqrt(lineB['err']*lineC['val']+lineB['val']
                * lineC['err'])/lineB['val']*lineC['val']
    )**2))
    df[PHASE] = phaseA['val'] + (sign) * phaseB['val'] + phaseC['val'] - 1.5*np.pi
    df[f'{ERR}{PHASE}'] = np.sqrt(phaseA['err']**2 + phaseB['err']**2 + phaseC['err']**2)
    return df


def Aover8B(df, lineA, lineB, phaseA, phaseB):
    df[AMPLITUDE] = lineA['val']/(8*lineB['val']**3)
    df[f'{ERR}{AMPLITUDE}'] = np.sqrt(lineA['val']/(64.*lineB['val']**6) + (9./64)
                                      * lineA['val']*np.abs(lineB['err']/lineB['val']**4))
    df[PHASE] = phaseA['val'] - 3 * phaseB['val'] - 0.5*np.pi
    df[f'{ERR}{PHASE}'] = np.sqrt(phaseA['err']**2 + 9 * phaseB['err']**2)
    return df


def Aover8BC(df, lineA, lineB, lineC, phaseA, phaseB, phaseC, sign=1):
    df[AMPLITUDE] = lineA['val']/(8*lineB['val']*lineC['val']**2)
    df[f'{ERR}{AMPLITUDE}'] = np.sqrt(lineA['err']/(64*lineB['val']**2*lineC['val']**4) +
                                      (1./64)*lineA['val'] *
                                      np.abs(np.sqrt(lineB['err']**2*lineC['err']**4 + 4 * lineB['val']**2*np.abs(lineC['val']*lineC['err'])**2) /
                                             (lineB['val']*lineC['val']**4))**2
                                      )
    df[PHASE] = phaseA['val'] + (sign) * phaseB['val'] - 2 * phaseC['val'] - 0.5*np.pi
    df[f'{ERR}{PHASE}'] = np.sqrt(phaseA['err']**2 + phaseB['err']**2 + phaseC['err']**2)
    return df
