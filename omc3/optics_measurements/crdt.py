"""
CRDTs
-------------------

:module: optics_measurements.crdt

Computes combined resonance driving terms
following the derivatons in https://arxiv.org/pdf/1402.1461.pdf.
"""

from pathlib import Path
from functools import reduce
import numpy as np
import pandas as pd
import tfs
from omc3.optics_measurements.constants import ERR, EXT, AMPLITUDE
from omc3.utils import iotools, logging_tools
from omc3.definitions.constants import PLANES
from omc3.utils import stats

LOGGER = logging_tools.get_logger(__name__)
PHASE = 'PHASE'


def Aover2B(df, lines, phases, errlines, errphases, sign=1):
    df[AMPLITUDE] = lines['A']/(2*lines['B'])
    df[f'{ERR}{AMPLITUDE}'] = np.sqrt(errlines['A']**2/(4*lines['B']**2)
                                      + 0.25 * lines['A']**2 * np.abs(errlines['B']/lines['B']**2)**2)
    df[PHASE] = phases['A'] - phases['B'] - 0.75
    df[f'{ERR}{PHASE}'] = np.sqrt(errphases['A']**2 + errphases['B']**2)
    return df


def Aover4B(df, lines, phases, errlines, errphases, sign=1):
    df[AMPLITUDE] = lines['A']/(4*lines['B']**2)
    df[f'{ERR}{AMPLITUDE}'] = np.sqrt(errlines['A']**2/(16*lines['B']**4)
                                      + 0.25 * lines['A']**2 * np.abs(errlines['B']/lines['B']**3)**2)
    df[PHASE] = phases['A'] + 2*phases['B'] - 0.75
    df[f'{ERR}{PHASE}'] = np.sqrt(errphases['A']**2 + errphases['B']**2)
    return df


def Aover4BC(df, lines, phases, errlines, errphases, sign=1):
    df[AMPLITUDE] = lines['A']/(4*lines['B']*lines['C'])
    df[f'{ERR}{AMPLITUDE}'] = np.sqrt(errlines['A']/(16*lines['B']**2*lines['C']**2
                              + (1/16.)*lines['A']**2*np.abs(
        np.sqrt(errlines['B']*lines['C']+lines['B']
                * errlines['C'])/lines['B']*lines['C']
    )**2))
    df[PHASE] = phases['A'] + (sign) * phases['B'] + phases['C'] - 0.75
    df[f'{ERR}{PHASE}'] = np.sqrt(errphases['A']**2 + errphases['B']**2 + errphases['C']**2)
    return df


def Aover8B(df, lines, phases, errlines, errphases, sign=1):
    df[AMPLITUDE] = lines['A']/(8*lines['B']**3)
    df[f'{ERR}{AMPLITUDE}'] = np.sqrt(errlines['A']/(64.*lines['B']**6) + (9./64)
                                      * lines['A']*np.abs(errlines['B']/lines['B']**4))
    df[PHASE] = phases['A'] - 3 * phases['B'] - 0.25
    df[f'{ERR}{PHASE}'] = np.sqrt(errphases['A']**2 + 9 * errphases['B']**2)
    return df


def Aover8BC(df, lines, phases, errlines, errphases, sign=1):
    df[AMPLITUDE] = lines['A']/(8*lines['B']*lines['C']**2)
    df[f'{ERR}{AMPLITUDE}'] = np.sqrt(errlines['A']/(64*lines['B']**2*lines['C']**4) +
                                      (1./64)*lines['A'] *
                                      np.abs(np.sqrt(errlines['B']**2*errlines['C']**4 + 4 * lines['B']**2*np.abs(lines['C']*errlines['C'])**2) /
                                             (lines['B']*lines['C']**4))**2
                                      )
    df[PHASE] = phases['A'] + (sign) * phases['B'] - 2 * phases['C']  + (sign) * 0.25
    df[f'{ERR}{PHASE}'] = np.sqrt(errphases['A']**2 + errphases['B']**2 + errphases['C']**2)
    return df


CRDTS = [
    {'order':"Coupling", 'term': "F_XY", 'func': Aover2B, 'lines': {'A': 'X01', 'B': 'Y01'}, 'sign':1},
    {'order':"Coupling", 'term': "F_YX", 'func': Aover2B, 'lines': {'A': 'Y10', 'B': 'X10'}, 'sign':1},

    {'order':"Sextupole", 'term': "F_NS3", 'func': Aover4B, 'lines': {'A': 'X_20', 'B': 'X10'}, 'sign':1},
    {'order':"Sextupole", 'term': "F_NS2", 'func': Aover4B, 'lines': {'A': 'X0_2', 'B': 'Y01'}, 'sign':1},
    {'order':"Sextupole", 'term': "F_NS1", 'func': Aover4BC, 'lines': {'A': 'Y_1_1', 'B': 'X10', 'C': 'Y01'}, 'sign':1},
    {'order':"Sextupole", 'term': "F_NS0", 'func': Aover4BC, 'lines': {'A': 'Y1_1', 'B': 'X10', 'C': 'Y01'}, 'sign':-1},

    {'order':"SkewSextupole", 'term': "F_SS3", 'func': Aover4B, 'lines': {'A': 'Y0_2', 'B': 'Y01'}, 'sign':1},
    {'order':"SkewSextupole", 'term': "F_SS2", 'func': Aover4B, 'lines': {'A': 'Y_20', 'B': 'X10'}, 'sign':1},
    {'order':"SkewSextupole", 'term': "F_SS1", 'func': Aover4BC, 'lines': {'A': 'X_1_1', 'B': 'X10', 'C': 'Y01'}, 'sign':1},
    {'order':"SkewSextupole", 'term': "F_SS0", 'func': Aover4BC, 'lines': {'A': 'X1_1', 'B': 'X10', 'C': 'Y01'}, 'sign':-1},

    {'order':"Octupole", 'term': "F_NO5", 'func': Aover8B, 'lines': {'A': 'Y03', 'B': 'Y01'}, 'sign':1},
    {'order':"Octupole", 'term': "F_NO4", 'func': Aover8BC, 'lines': {'A': 'X12', 'B': 'X10', 'C': 'Y01'}, 'sign':-1},
    {'order':"Octupole", 'term': "F_NO3", 'func': Aover8B, 'lines': {'A': 'X30', 'B': 'X10'}, 'sign':1},
    {'order':"Octupole", 'term': "F_NO2", 'func': Aover8BC, 'lines': {'A': 'X_12', 'B': 'X10', 'C': 'Y01'}, 'sign':1},
    {'order':"Octupole", 'term': "F_NO1", 'func': Aover8BC, 'lines': {'A': 'Y2_1', 'B': 'Y01', 'C': 'X10'}, 'sign':1},
    {'order':"Octupole", 'term': "F_NO0", 'func': Aover8BC, 'lines': {'A': 'Y21', 'B': 'Y01', 'C': 'X10'}, 'sign':-1},
]


def calculate(measure_input, input_files, header):
    """

    Args:
        measure_input:
        input_files:
        header:
    Returns:

    """
    LOGGER.info("Start of CRDT analysis")
    for plane in PLANES:
        input_files[plane] = unscale_amps(input_files[plane], plane)
        input_files[plane] = scale_amps_with_sqrtbeta(input_files[plane], measure_input, plane)

    joined_dfs = joined_planes(input_files)

    for crdt in CRDTS:
        LOGGER.debug(f"Processing CRDT {crdt['term']}")
        result_dfs = []
        for joined_df in joined_dfs:
            result_dfs.append(process_crdt(joined_df, crdt))
        df = average_results(result_dfs=result_dfs,
                             union_columns=['NAME', 'S'],
                             merge_columns=[AMPLITUDE, f'{ERR}{AMPLITUDE}', PHASE, f'{ERR}{PHASE}'],
                             merge_functions=[stats.weighted_nanmean, stats.weighted_nanrms,
                                              stats.circular_nanmean, stats.circular_nanerror],
                             index="NAME")
        write(df, header, measure_input, crdt['order'], crdt['term'])


def scale_amps_with_sqrtbeta(input_files, meas_input, plane):
    model_beta = meas_input.accelerator.model[f'BET{plane}']
    processed_files = []
    for lin_df in input_files:
        cols = [col for col in lin_df.columns.to_numpy() if col.startswith('AMP')]
        lin_df.loc[:, cols] = lin_df.loc[:, cols].div(model_beta**0.5, axis="index")
        processed_files.append(lin_df)
    return processed_files


def unscale_amps(input_files, plane):
    processed_files = []
    for lin_df in input_files:
        cols = [col for col in lin_df.columns.to_numpy() if col.startswith('AMP')]
        cols.remove(f"AMP{plane}")
        lin_df[f"AMP{plane}"] = lin_df.loc[:, f"AMP{plane}"].to_numpy()/2.
        lin_df.loc[:, cols] = lin_df.loc[:, cols].mul(lin_df.loc[:, f'AMP{plane}'], axis="index")
        processed_files.append(lin_df)
    return processed_files

def process_crdt(joined_df, crdt):
    df = pd.DataFrame(index=joined_df.index, data={'S': joined_df['S']})

    lines = {}
    phases = {}
    errlines = {}
    errphases = {}
    for data_dict, prefix in zip([lines, phases, errlines, errphases], ['AMP', 'FREQ', 'ERRAMP', 'ERRFREQ']):
        for key, line in crdt['lines'].items():
            try:
                translate_line = translate_line_to_col(line)
                data_dict[key] = joined_df[translate_line[prefix]]
            except KeyError:
                LOGGER.debug(f"No {prefix} for line {line} found in lin-files, set to 0")
                data_dict[key] = np.nan
    df = crdt['func'](df, lines, phases, errlines, errphases, crdt['sign'])
    return df


def average_results(result_dfs, union_columns, merge_columns, merge_functions, index):
    assert len(merge_columns) == len(merge_functions)

    result_df = reduce(lambda left, right: pd.merge(left.reset_index(level=0)[union_columns],
                                                    right.reset_index(level=0)[union_columns],
                                                    on=union_columns,
                                                    how='outer').set_index(index),
                       result_dfs)
    result_dfs = [df.reindex(result_df.index) for df in result_dfs]
    for column, func in zip(merge_columns, merge_functions):
        data = np.array([df[column].to_numpy() for df in result_dfs])
        result_df[column] = func(data, axis=0)
    return result_df


def translate_line_to_col(line):
    plane = line[0]
    line = line[1:]

    if (plane == 'X' and line == '10') or (plane == 'Y' and line == '01'):
        return {'AMP': f'AMP{plane}', 'FREQ': f'MU{plane}', 'ERRAMP': f'ERRAMP{plane}', 'ERRFREQ': f'ERRMU{plane}'}
    return {'AMP': f'AMP{line}_{plane}', 'FREQ': f'PHASE{line}_{plane}', 'ERRAMP': f'ERRAMP{line}_{plane}', 'ERRFREQ': f'ERRPHASE{line}_{plane}'}


def write(df, header, meas_input, order, crdt):
    outputdir = Path(meas_input.outputdir)/"crdt"/order
    iotools.create_dirs(outputdir)
    tfs.write(str(outputdir/f"{crdt}{EXT}"), df, header, save_index='NAME')

def joined_planes(input_files):
    """
    Merges DataFrame from the two planes in one df
    Parameters:
        input_files
    Returns:
        merged DataFrame 
    """
    joined_dfs = []

    assert len(input_files['X']) == len(input_files['Y'])

    for linx, liny in zip(input_files['X'], input_files['Y']):

        for df, plane in zip((linx, liny), PLANES):
            rename_cols(df, plane, ['NAME', 'S', f'TUNE{plane}', f'AMP{plane}', f'MU{plane}', f'MU{plane}SYNC'])

        joined_dfs.append(pd.merge(left=linx,
                                   right=liny,
                                   on=['NAME', 'S'],
                                   how='inner',
                                   sort=False,
                                   suffixes=(False, False)
                                   ).set_index('NAME'))

    return joined_dfs


def rename_cols(df, suffix, exceptions=['']):
    df.columns = [f'{col}_{suffix}' if col not in exceptions else col for col in df.columns]
    return df
