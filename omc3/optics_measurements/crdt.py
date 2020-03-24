"""
CRDTs
-------------------

:module: optics_measurements.crdt

Computes combined resonance driving terms
following the derivatons in https://arxiv.org/pdf/1402.1461.pdf.
"""

from pathlib import Path
import numpy as np
import tfs
import pandas as pd
from omc3.optics_measurements.constants import ERR, EXT, AMPLITUDE
from omc3.utils import iotools, logging_tools
from omc3.harpy.constant import FILE_LIN_EXT

LOGGER = logging_tools.get_logger(__name__)
PHASE = 'PHASE'

CRDTS = {
    {'order':"Coupling", 'term': "F_XY", 'func': Aover2B, 'A': 'X01', 'B': 'Y01'},
    {'order':"Coupling", 'term': "F_YX", 'func': Aover2B, 'A': 'Y10', 'B': 'X10'},

    {'order':"Sextupole", 'term': "F_NS3", 'func': Aover4B, 'A': 'X_20', 'B': 'X10'},
    {'order':"Sextupole", 'term': "F_NS2", 'func': Aover4B, 'A': 'X0_2', 'B': 'Y01'},
    {'order':"Sextupole", 'term': "F_NS1", 'func': Aover4BC, 'A': 'Y_1_1', 'B': 'X10', 'C': 'Y01'},
    {'order':"Sextupole", 'term': "F_NS0", 'func': Aover4BC, 'A': 'Y1_1', 'B': 'X10', 'C': 'Y01'},

    {'order':"SkewSextupole", 'term': "F_SS3", 'func': Aover4B, 'A': 'Y0_2', 'B': 'Y01'},
    {'order':"SkewSextupole", 'term': "F_SS2", 'func': Aover4B, 'A': 'Y_20', 'B': 'X10'},
    {'order':"SkewSextupole", 'term': "F_SS1", 'func': Aover4BC, 'A': 'X_1_1', 'B': 'X10', 'C': 'Y01'},
    {'order':"SkewSextupole", 'term': "F_SS0", 'func': Aover4BC, 'A': 'X1_1', 'B': 'X10', 'C': 'Y01'},

    {'order':"Octupole", 'term': "F_NO5", 'func': Aover8B, 'A': 'Y03', 'B': 'Y01'},
    {'order':"Octupole", 'term': "F_NO4", 'func': Aover8BC, 'A': 'X12', 'B': 'X10', 'C': 'Y01'},
    {'order':"Octupole", 'term': "F_NO3", 'func': Aover8B, 'A': 'X30', 'B': 'X10'},
    {'order':"Octupole", 'term': "F_NO2", 'func': Aover8BC, 'A': 'X_12', 'B': 'X10', 'C': 'Y01'},
    {'order':"Octupole", 'term': "F_NO1", 'func': Aover8BC, 'A': 'Y2_1', 'B': 'X10', 'C': 'Y01'},
    {'order':"Octupole", 'term': "F_NO0", 'func': Aover8BC, 'A': 'Y21', 'B': 'X10', 'C': 'Y01'},
}


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


def translate_line_to_col(line):
    plane = line[0]
    line = line[1:]

    if (plane == 'X' and line == '10') or (plane == 'Y' and line == '01'):
        return FILE_LIN_EXT.format(plane=plane.lower()), f'AMP{plane.lower()}', f'MU{plane.lower()}'
    return FILE_LIN_EXT.format(plane=plane.lower()), f'AMP{line}', f'PHASE{line}'


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
