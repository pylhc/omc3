"""
CRDTs
-------------------

:module: optics_measurements.crdt

Computes combined resonance driving terms
following the derivations in https://arxiv.org/pdf/1402.1461.pdf.
"""

from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import tfs
import scipy.odr
from omc3.optics_measurements.constants import ERR, EXT, AMPLITUDE
from omc3.utils import iotools, logging_tools
from omc3.definitions.constants import PLANES
from omc3.harpy.constants import COL_AMP, COL_MU, COL_PHASE, COL_TUNE, COL_ERR

LOGGER = logging_tools.get_logger(__name__)
PHASE = 'PHASE'

CRDT_COLUMNS = [AMPLITUDE, f'{ERR}{AMPLITUDE}', PHASE, f'{ERR}{PHASE}']


CRDTS = [
    {'order':"coupling", 'term': "F_XY", 'plane': 'X', 'line': [0, 1]},
    {'order':"coupling", 'term': "F_YX", 'plane': 'Y', 'line': [1, 0]},

    {'order':"sextupole", 'term': "F_NS3", 'plane': 'X', 'line': [-2, 0]},
    {'order':"sextupole", 'term': "F_NS2", 'plane': 'X', 'line': [0, -2]},
    {'order':"sextupole", 'term': "F_NS1", 'plane': 'Y', 'line': [-1, -1]},
    {'order':"sextupole", 'term': "F_NS0", 'plane': 'Y', 'line': [1, -1]},

    {'order':"skewsextupole", 'term': "F_SS3", 'plane': 'Y', 'line': [0, -2]},
    {'order':"skewsextupole", 'term': "F_SS2", 'plane': 'Y', 'line': [-2, 0]},
    {'order':"skewsextupole", 'term': "F_SS1", 'plane': 'X', 'line': [-1, -1]},
    {'order':"skewsextupole", 'term': "F_SS0", 'plane': 'X', 'line': [1, -1]},

    {'order':"octupole", 'term': "F_NO5", 'plane': 'Y', 'line': [0, 3]},
    {'order':"octupole", 'term': "F_NO4", 'plane': 'X', 'line': [1, 2]},
    {'order':"octupole", 'term': "F_NO3", 'plane': 'X', 'line': [3, 0]},
    {'order':"octupole", 'term': "F_NO2", 'plane': 'X', 'line': [-1, 2]},
    {'order':"octupole", 'term': "F_NO1", 'plane': 'Y', 'line': [2, -1]},
    {'order':"octupole", 'term': "F_NO0", 'plane': 'Y', 'line': [2, 1]},
]


def calculate(measure_input, input_files, invariants, header):
    """

    Args:
        measure_input:
        input_files:
        header:
    Returns:

    """
    LOGGER.info("Start of CRDT analysis")

    joined_dfs = joined_planes(input_files)
    for crdt in CRDTS:
        LOGGER.debug(f"Processing CRDT {crdt['term']}")
        bpm_names = input_files.bpms(dpp_value=0)
        result_df = pd.DataFrame(measure_input.accelerator.model).loc[bpm_names, ["S"]]
        lines_and_phases, phase_sign = get_line_and_phase(crdt, joined_dfs)

        result_df[AMPLITUDE], result_df[f'{ERR}{AMPLITUDE}'] = get_crdt_amplitude(joined_dfs,
                                                                                  crdt,
                                                                                  invariants,
                                                                                  lines_and_phases)
        result_df[PHASE], result_df[f'{ERR}{PHASE}'] = get_crdt_phases(joined_dfs,
                                                                       crdt,
                                                                       lines_and_phases,
                                                                       phase_sign)

        write(result_df, header, measure_input, crdt['order'], crdt['term'])


def write(df, header, meas_input, order, crdt):
    outputdir = Path(meas_input.outputdir)/"crdt"/order
    iotools.create_dirs(outputdir)
    tfs.write(str(outputdir/f"{crdt}{EXT}"), df, header, save_index='NAME')


def get_line_and_phase(crdt, joined_df):
    line = deepcopy(crdt['line'])
    translate_line = translate_line_to_col(line, crdt['plane'])
    line[0], line[1] = -crdt['line'][0], -crdt['line'][1]
    conj_translate_line = translate_line_to_col(line, crdt['plane'])

    if all(elem in joined_df[0].columns for elem in [translate_line[AMPLITUDE], translate_line[PHASE]]):
        return translate_line, 1
    if all(elem in joined_df[0].columns for elem in [conj_translate_line[AMPLITUDE], conj_translate_line[PHASE]]):
        return conj_translate_line, -1
    raise ValueError(f"No data for line {crdt['line']} in plane {crdt['plane']} found.")


def translate_line_to_col(line, plane):
    
    line = f"{line[0]}{line[1]}".replace('-', '_')

    return dict(zip(CRDT_COLUMNS, [f'{COL_AMP}{line}_{plane}',
                                   f'{COL_ERR}{COL_AMP}{line}_{plane}',
                                   f'{COL_PHASE}{line}_{plane}',
                                   f'{COL_ERR}{COL_PHASE}{line}_{plane}']))


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
            rename_cols(df, plane, ['NAME', 'S',
                                    f'{COL_TUNE}{plane}', f'{COL_AMP}{plane}',
                                    f'{COL_MU}{plane}', f'{COL_MU}{plane}SYNC'])

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


def get_crdt_phases(joined_dfs, crdt, lines_and_phases, phase_sign):
    phases, err_phases = np.zeros((len(joined_dfs), len(joined_dfs[0]))), np.zeros((len(joined_dfs), len(joined_dfs[0])))

    for idx, joined_df in enumerate(joined_dfs):
        phases[idx, :] = (phase_sign*joined_df[lines_and_phases[PHASE]] -
                          crdt['line'][0]*joined_df['MUX'] -
                          crdt['line'][1]*joined_df['MUY'] -
                          (3 - (np.abs(crdt['line'][0])+np.abs(crdt['line'][1])-1)//2)/2*np.pi)
        try:
            err_phases[idx, :] = np.sqrt(joined_df[lines_and_phases[f'{ERR}{PHASE}']]**2+
                                         (crdt['line'][0]*joined_df['ERRMUX'])**2+
                                         (crdt['line'][1]*joined_df['ERRMUY'])**2)
        except KeyError:
            err_phases[idx, :] = np.NaN

    return np.mean(phases, axis=0), np.nanmean(err_phases, axis=0)

def get_crdt_invariant(crdt, invariants):
    exp = {'X': np.abs(crdt['line'][0]), 'Y': np.abs(crdt['line'][1])}
    exp[crdt['plane']] = exp[crdt['plane']] - 1 # to compensate for the normalization with tune line

    crdt_invariant = invariants['X'].T[0]**exp['X']*invariants['Y'].T[0]**exp['Y']
    err_crdt_invariant = np.sqrt((exp['X']*crdt_invariant/invariants['X'].T[0])**2*
                                 invariants['X'].T[1]**2 +
                                 (exp['Y']*crdt_invariant/invariants['Y'].T[0])**2*
                                 invariants['Y'].T[1]**2)
    return crdt_invariant, err_crdt_invariant


def get_crdt_amplitude(joined_dfs, crdt, invariants, lines_and_phases):
    crdt_invariants, err_crdt_invariants = get_crdt_invariant(crdt, invariants)

    assert len(crdt_invariants) == len(joined_dfs)

    amps, err_amps = np.zeros(len(joined_dfs[0])), np.zeros(len(joined_dfs[0]))

    for idx, bpm in enumerate(joined_dfs[0].index):
        lineamplitudes, err_lineamplitudes = [], []
        for joined_df in joined_dfs:
            lineamplitudes.append(joined_df.loc[bpm, lines_and_phases[AMPLITUDE]])
            try:
                err_lineamplitudes.append(joined_df.loc[bpm, lines_and_phases[f'{ERR}{AMPLITUDE}']])
            except KeyError:
                err_lineamplitudes.append(0.)

        amps[idx], err_amps[idx] = fit_amplitude(lineamplitudes,
                                                 err_lineamplitudes,
                                                 crdt_invariants,
                                                 err_crdt_invariants)

    return amps, err_amps


def fit_amplitude(lineamplitudes, err_lineamplitudes, crdt_invariant, err_crdt_invariant):

    def fun(p, x):
        return p * x * 2 # factor 2 to get to complex amplitudes again
    fit_model = scipy.odr.Model(fun)
    data_model = scipy.odr.RealData(
        x=crdt_invariant,
        y=lineamplitudes,
        sx=err_crdt_invariant,
        sy=None if all(err == 0. for err in err_lineamplitudes) else err_lineamplitudes,
    )
    odr = scipy.odr.ODR(data_model, fit_model,
                        beta0=[0.5*lineamplitudes[0]/crdt_invariant[0]])
    
    odr_output = odr.run()

    return odr_output.beta[0], odr_output.sd_beta[0]
