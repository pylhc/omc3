"""
Combined RDTs
-------------

This module contains combined resonance driving terms calculations functionality of
``optics_measurements``.
It provides functions to compute combined resonance driving terms following the derivations in
https://arxiv.org/pdf/1402.1461.pdf.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.odr
import tfs

from omc3.definitions.constants import PI2, PLANES
from omc3.harpy.constants import COL_AMP, COL_ERR, COL_MU, COL_PHASE
from omc3.optics_measurements.constants import (
    AMPLITUDE,
    CRDT_FOLDER,
    ERR,
    EXT,
    IMAG,
    MDL,
    NAME,
    PHASE,
    REAL,
)
from omc3.optics_measurements.data_models import (
    InputFiles,
    check_and_warn_about_offmomentum_data,
    filter_for_dpp,
)
from omc3.optics_measurements.rdt import get_line_sign_and_suffix
from omc3.utils import iotools, logging_tools
from omc3.utils.stats import circular_nanerror, circular_nanmean

if TYPE_CHECKING: 
    from generic_parser import DotDict

LOGGER = logging_tools.get_logger(__name__)

CRDT_COLUMNS = [AMPLITUDE, f'{ERR}{AMPLITUDE}', PHASE, f'{ERR}{PHASE}']

CRDTS = [
    {'order': "skew_quadrupole", 'term': "F_XY", 'plane': 'X', 'line': [0, 1]},
    {'order': "skew_quadrupole", 'term': "F_YX", 'plane': 'Y', 'line': [1, 0]},

    {'order': "normal_sextupole", 'term': "F_NS3", 'plane': 'X', 'line': [-2, 0]},
    {'order': "normal_sextupole", 'term': "F_NS2", 'plane': 'X', 'line': [0, -2]},
    {'order': "normal_sextupole", 'term': "F_NS1", 'plane': 'Y', 'line': [-1, -1]},
    {'order': "normal_sextupole", 'term': "F_NS0", 'plane': 'Y', 'line': [1, -1]},

    {'order': "skew_sextupole", 'term': "F_SS3", 'plane': 'Y', 'line': [0, -2]},
    {'order': "skew_sextupole", 'term': "F_SS2", 'plane': 'Y', 'line': [-2, 0]},
    {'order': "skew_sextupole", 'term': "F_SS1", 'plane': 'X', 'line': [-1, -1]},
    {'order': "skew_sextupole", 'term': "F_SS0", 'plane': 'X', 'line': [1, -1]},

    {'order': "normal_octupole", 'term': "F_NO5", 'plane': 'Y', 'line': [0, 3]},
    {'order': "normal_octupole", 'term': "F_NO4", 'plane': 'X', 'line': [1, 2]},
    {'order': "normal_octupole", 'term': "F_NO3", 'plane': 'X', 'line': [3, 0]},
    {'order': "normal_octupole", 'term': "F_NO2", 'plane': 'X', 'line': [-1, 2]},
    {'order': "normal_octupole", 'term': "F_NO1", 'plane': 'Y', 'line': [2, -1]},
    {'order': "normal_octupole", 'term': "F_NO0", 'plane': 'Y', 'line': [2, 1]},
]


def calculate(measure_input: DotDict, input_files: InputFiles, invariants, header):
    """ Calculate the CRDT values. """
    LOGGER.info("Start of CRDT analysis")
    dpp_value = measure_input.analyse_dpp
    if dpp_value is None:
        for plane in PLANES:
            check_and_warn_about_offmomentum_data(input_files, plane, id_="CRDT calculation")
    else:
        invariants = filter_for_dpp(invariants, input_files, dpp_value)

    assert len(input_files['X']) == len(input_files['Y'])
    bpm_names = input_files.bpms(dpp_value=0)
    for crdt in CRDTS:
        LOGGER.debug(f"Processing CRDT {crdt['term']}")
        result_df = generic_dataframe(input_files, measure_input, bpm_names, dpp_value=dpp_value)
        phase_sign, line_suffix = get_line_sign_and_suffix(crdt["line"], input_files, crdt["plane"])
        lines_and_phases = get_column_names(line_suffix)
        nqx, nqy = crdt['line']
        crdt_order = abs(nqx) + abs(nqy)
        # don't know the exact way to get to this via line indices so for now only via hacky way
        signflip = -1 if crdt['term'] in ("F_NO2", "F_NO1") else 1

        result_df = pd.merge(
            result_df, 
            input_files.joined_frame(crdt["plane"], lines_and_phases.values(), dpp_value=dpp_value),
            how='inner', left_index=True, right_index=True
        )

        amplitudes = input_files.get_data(result_df, lines_and_phases[AMPLITUDE])
        err_amplitudes = input_files.get_data(result_df, lines_and_phases[f"{ERR}{AMPLITUDE}"])
        result_df[AMPLITUDE], result_df[f'{ERR}{AMPLITUDE}'] = get_crdt_amplitude(crdt, invariants, amplitudes, err_amplitudes)

        phases = ((phase_sign * input_files.get_data(result_df, lines_and_phases[PHASE]) -
                  nqx * input_files.get_data(result_df, "MUX") -
                  nqy * input_files.get_data(result_df, "MUY") -
                  signflip * 0.75 - 0.5 * ((crdt_order)//3)) % 1)
        err_phases = np.sqrt(input_files.get_data(result_df, lines_and_phases[f'{ERR}{PHASE}'])**2 +
                             (nqx * input_files.get_data(result_df, "ERRMUX"))**2 +
                             (nqy * input_files.get_data(result_df, "ERRMUY"))**2)

        result_df[PHASE] = circular_nanmean(phases, axis=1, errors=err_phases)
        result_df[f'{ERR}{PHASE}'] = circular_nanerror(phases, axis=1, errors=err_phases)

        result_df[REAL] = np.cos(PI2 * result_df[PHASE].to_numpy()) * result_df[AMPLITUDE].to_numpy()
        result_df[IMAG] = np.sin(PI2 * result_df[PHASE].to_numpy()) * result_df[AMPLITUDE].to_numpy()

        write(result_df.loc[:, ["S", AMPLITUDE, f'{ERR}{AMPLITUDE}', PHASE, f'{ERR}{PHASE}', REAL, IMAG]],
              add_line_and_freq_to_header(header, crdt), measure_input, crdt['order'], crdt['term'])


def generic_dataframe(input_files: InputFiles, measure_input: DotDict, bpm_names: Sequence[str], dpp_value: int = 0):
    """ Generate a dataframe based on the MU-MDL columns from each measuement. """
    result_df = pd.DataFrame(measure_input.accelerator.model).loc[bpm_names, ["S", "MUX", "MUY"]]
    result_df.rename(columns={f"{COL_MU}X": f"{COL_MU}X{MDL}", f"{COL_MU}Y": f"{COL_MU}Y{MDL}"}, inplace=True)
    for plane in PLANES:
        result_df = pd.merge(
            result_df, input_files.joined_frame(plane, [f"MU{plane}", f"{ERR}MU{plane}"], dpp_value=dpp_value),
            how='inner', left_index=True, right_index=True)
    return result_df


def add_line_and_freq_to_header(header, crdt):
    mod_header = header.copy()
    mod_header["LINE"] = f"{crdt['plane']}({crdt['line'][0]}, {crdt['line'][1]})"
    freq = np.mod(crdt['line']@np.array([header['Q1'], header['Q2']]), 1)
    mod_header["FREQ"] = freq if freq <= 0.5 else 1 - freq
    return mod_header


def write(df, header, meas_input, order, crdt):
    outputdir = Path(meas_input.outputdir) / CRDT_FOLDER / order
    iotools.create_dirs(outputdir)
    tfs.write(str(outputdir/f"{crdt}{EXT}"), df, header, save_index=NAME)


def get_column_names(line):
    return dict(zip(CRDT_COLUMNS, [f'{COL_AMP}{line}',
                                   f'{COL_ERR}{COL_AMP}{line}',
                                   f'{COL_PHASE}{line}',
                                   f'{COL_ERR}{COL_PHASE}{line}']))


def get_crdt_invariant(crdt, invariants):
    exp = {'X': np.abs(crdt['line'][0]), 'Y': np.abs(crdt['line'][1])}
    exp[crdt['plane']] = exp[crdt['plane']] - 1 # to compensate for the normalization with tune line

    crdt_invariant = invariants['X'].T[0]**exp['X'] * invariants['Y'].T[0]**exp['Y']
    err_crdt_invariant = np.sqrt((exp['X'] * crdt_invariant / invariants['X'].T[0])**2 *
                                 invariants['X'].T[1]**2 +
                                 (exp['Y'] * crdt_invariant / invariants['Y'].T[0])**2 *
                                 invariants['Y'].T[1]**2)
    return crdt_invariant, err_crdt_invariant


def get_crdt_amplitude(crdt, invariants, line_amps, line_amp_errors):
    crdt_invariants, err_crdt_invariants = get_crdt_invariant(crdt, invariants)
    amps, err_amps = np.zeros(line_amps.shape[0]), np.zeros(line_amps.shape[0])

    for idx in range(line_amps.shape[0]):
        amps[idx], err_amps[idx] = fit_amplitude(line_amps[idx],
                                                 line_amp_errors[idx],
                                                 crdt_invariants,
                                                 err_crdt_invariants)

    return amps, err_amps


def fit_amplitude(lineamplitudes, err_lineamplitudes, crdt_invariant, err_crdt_invariant):

    def fun(p, x):
        return p * x * 2  # factor 2 to get to complex amplitudes again

    fit_model = scipy.odr.Model(fun)
    data_model = scipy.odr.RealData(
        x=crdt_invariant,
        y=lineamplitudes,
        sx=err_crdt_invariant,
        sy=None if all(err == 0. for err in err_lineamplitudes) else err_lineamplitudes,
    )
    odr = scipy.odr.ODR(data_model, fit_model, beta0=[0.5 * lineamplitudes[0] / crdt_invariant[0]])
    odr_output = odr.run()
    return odr_output.beta[0], odr_output.sd_beta[0]
