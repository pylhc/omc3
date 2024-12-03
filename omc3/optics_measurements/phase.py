"""
Phase Advance
-------------

This module contains phase calculation functionality of ``optics_measurements``.
It provides functions to compute betatron phase advances and structures to store them.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import pandas as pd
import tfs

from omc3.optics_measurements.constants import (
    DELTA,
    DRIVEN_PHASE_NAME,
    DRIVEN_TOTAL_PHASE_NAME,
    ERR,
    EXT,
    MDL,
    MEASUREMENT,
    MODEL,
    NAME,
    PHASE,
    PHASE_ADV,
    PHASE_NAME,
    SPECIAL_PHASE_NAME,
    TOTAL_PHASE_NAME,
    S,
)
from omc3.optics_measurements.data_models import (
    InputFiles,
    check_and_warn_about_offmomentum_data,
)
from omc3.optics_measurements.toolbox import ang_sum, df_ang_diff, df_diff
from omc3.utils import logging_tools, stats

if TYPE_CHECKING: 
    from collections.abc import Sequence

    from generic_parser import DotDict
    from numpy.typing import ArrayLike

    from omc3.model.accelerators.accelerator import Accelerator
    from omc3.optics_measurements.tune import TuneDict

PhaseDict = TypedDict(
    'PhaseDict', {
        MODEL: pd.DataFrame, 
        MEASUREMENT: pd.DataFrame, 
        f'{ERR}{MEASUREMENT}': pd.DataFrame
    }
)
LOGGER = logging_tools.get_logger(__name__)

class CompensationMode:
    NONE: str = "none"
    MODEL: str = "model"
    EQUATION: str = "equation"
    
    @classmethod
    def all(cls) -> list[str]:
        return [cls.NONE, cls.MODEL, cls.EQUATION]

COMPENSATED: str = "compensated"  # former 'free'
UNCOMPENSATED: str = "uncompensated"  # former 'driven'

def calculate(
    meas_input: DotDict,
    input_files: InputFiles,
    tunes: TuneDict,
    plane: str,
    no_errors: bool = False,
) -> tuple[dict[str, PhaseDict], list[pd.DataFrame]]:
    """
    Calculate phases for 'compensated' (aka 'free') and 'uncompensated' (aka 'driven') cases from the measurement files, 
    and return a dictionary combining the results for each transverse plane.

    Args:
        meas_input (DotDict): `OpticsInput` object containing analysis settings from the command-line.
        input_files (InputFiles): `InputFiles` object containing frequency spectra files (linx/y).
        tunes (TuneDict): `TuneDict` contains measured tunes.
        plane (str): marking the horizontal or vertical plane, **X** or **Y**.
        no_errors (bool): if ``True``, measured errors shall not be propagated (only their spread).

    Returns:
        A tuple of a dictionary and a list of corresponding phase DataFrames.
        The dictionary contains the compensated and uncompensated results in `PhaseDict` form, 
        i.e. a dictionary of DataFrames with the phase models, measured and errors.
        The list contains the DataFrames with the phase advances between BPMs and the total phase advances,
        for both compensated and uncompensated cases.
    """
    # Clean up compensation mode
    if meas_input.compensation is None:
        meas_input.compensation = CompensationMode.NONE
    else:
        meas_input.compensation = meas_input.compensation.lower()

    LOGGER.info("-- Run phase calculation with requested compensation")
    phase_advances, dfs = _calculate_phase_advances(
        meas_input=meas_input,
        input_files=input_files,
        tunes=tunes,
        plane=plane,
        model_df=meas_input.accelerator.model,
        compensation=meas_input.compensation,
        no_errors=no_errors,
    )

    if meas_input.compensation == CompensationMode.NONE:
        uncompensated_phase_advances = None
    else:
        LOGGER.info("-- Run phase calculation without compensation")

        uncompensated_phase_advances, drv_dfs = _calculate_phase_advances(
            meas_input=meas_input,
            input_files=input_files,
            tunes=tunes,
            plane=plane,
            model_df=meas_input.accelerator.model_driven,
            compensation=CompensationMode.NONE, 
            no_errors=no_errors,
        )
        dfs = dfs + drv_dfs

    if len(phase_advances[MEASUREMENT].index) < 3:
        LOGGER.warning(
            "Less than 3 non-NaN phase-advances found. "
            "This will most likely lead to errors later on in the N-BPM or 3-BPM methods.\n"
            "Common issues to check:\n"
            "- Did you pass the correct tunes to harpy? Possibly also too small tolerance window?\n"
            "- Did excitation trigger in both planes? BPMs might be cleaned if only found in one plane.\n"
            "- Are cleaning settings (peak-to-peak, singular-value-cut) too agressive?\n"
            "- Are you using a machine with less than 3 BPMs? Oh dear."
        )

    return {COMPENSATED: phase_advances, UNCOMPENSATED: uncompensated_phase_advances}, dfs


def _calculate_phase_advances(
    meas_input: DotDict,
    input_files: InputFiles,
    tunes: TuneDict,
    plane: str,
    model_df: pd.DataFrame,
    compensation: str,
    no_errors: bool,
) -> tuple[PhaseDict, list[pd.DataFrame, pd.DataFrame]]:
    """
    Calculates phase advances.

    Args:
        meas_input: the input object including settings and the accelerator class.
        input_files: includes measurement tfs.
        tunes: `TunesDict` object containing measured and model tunes and ac2bpm object
        plane: marking the horizontal or vertical plane, **X** or **Y**.
        compensation: compensation mode to use
        no_errors: if ``True``, measured errors shall not be propagated (only their spread).

    Returns:
        A tuple of 
        - a `dictionary` of `TfsDataFrames` indexed (BPMi x BPMj) yielding phase advance `phi_ij`.

            - "MEAS": measured phase advances,
            - "ERRMEAS": errors of measured phase advances,
            - "MODEL": model phase advances.

            +------++--------+--------+--------+--------+
            |      ||  BPM1  |  BPM2  |  BPM3  |  BPM4  |
            +======++========+========+========+========+
            | BPM1 ||   0    | phi_12 | phi_13 | phi_14 |
            +------++--------+--------+--------+--------+
            | BPM2 || phi_21 |    0   | phi_23 | phi_24 |
            +------++--------+--------+--------+--------+
            | BPM3 || phi_31 | phi_32 |   0    | phi_34 |
            +------++--------+--------+--------+--------+
            | BPM4 || phi_41 | phi_42 | phi_43 |    0   |
            +------++--------+--------+--------+--------+

            The phase advance between BPM_i and BPM_j can be obtained via:
            phase_advances["MEAS"].loc[BPMi,BPMj]
        - a list of output data frames (for tfs-files to be written out);
          the first one contains the phase advances between the BPMs, the
          second one contains the total phase advances.
    """
    LOGGER.info("Calculating phase advances")
    LOGGER.info(f"Measured tune in plane {plane} = {tunes[plane]['Q']}")

    df = model_df.loc[:, [S, f"{PHASE_ADV}{plane}"]]
    how = 'outer' if meas_input.union else 'inner'

    dpp_value = meas_input.analyse_dpp
    if dpp_value is None:
        check_and_warn_about_offmomentum_data(input_files, plane, id_="Phase calculations")

    joined_df = input_files.joined_frame(
        plane, [f"{PHASE_ADV}{plane}", f"{ERR}{PHASE_ADV}{plane}"], dpp_value=dpp_value, how=how
    )
    df = pd.merge(df, joined_df, how='inner', left_index=True, right_index=True)

    direction = meas_input.accelerator.beam_direction
    df[input_files.get_columns(df, f"{PHASE_ADV}{plane}")] = direction * input_files.get_data(df, f"{PHASE_ADV}{plane}")     
    
    phases_mdl = df.loc[:, f"{PHASE_ADV}{plane}"].to_numpy()
    phase_advances = {MODEL: _get_square_data_frame(_get_all_phase_diff(phases_mdl), df.index)}
    
    if compensation == CompensationMode.MODEL:
        df = _compensate_by_model(input_files, meas_input, df, plane)

    phases_meas = input_files.get_data(df, f"{PHASE_ADV}{plane}")

    if compensation == CompensationMode.EQUATION:
        phases_meas = _compensate_by_equation(phases_meas, plane, tunes)

    phases_errors = input_files.get_data(df, f"{ERR}{PHASE_ADV}{plane}")
    if phases_meas.ndim < 2:
        phase_advances[MEASUREMENT] = _get_square_data_frame(_get_all_phase_diff(phases_meas), df.index)
        phase_advances[f"{ERR}{MEASUREMENT}"] = _get_square_data_frame(
                np.zeros((len(phases_meas), len(phases_meas))), df.index)
        return phase_advances
    
    if meas_input.union:
        mask = np.isnan(phases_meas)
        phases_meas[mask], phases_errors[mask] = 0.0, np.inf
        if no_errors:
            phases_errors[~mask] = 1e-10
    elif no_errors:
        phases_errors = None
    phases_3d = phases_meas[np.newaxis, :, :] - phases_meas[:, np.newaxis, :]
    
    if phases_errors is not None:
        errors_3d = phases_errors[np.newaxis, :, :] + phases_errors[:, np.newaxis, :]
    else:
        errors_3d = None
    phase_advances[MEASUREMENT] = _get_square_data_frame(stats.circular_mean(
            phases_3d, period=1, errors=errors_3d, axis=2) % 1.0, df.index)
    phase_advances[f"{ERR}{MEASUREMENT}"] = _get_square_data_frame(stats.circular_error(
            phases_3d, period=1, errors=errors_3d, axis=2), df.index)
    return phase_advances, [_create_output_df(phase_advances, df, plane),
                            _create_output_df(phase_advances, df, plane, tot=True)]


def _compensate_by_equation(phases_meas: ArrayLike, plane: str, tunes: TuneDict) -> ArrayLike:
    """ Compensate the measured phases by equation.
    
    TODO: Reference!
     """
    driven_tune, free_tune, ac2bpmac = tunes[plane]["Q"], tunes[plane]["QF"], tunes[plane]["ac2bpm"]
    k_bpmac = ac2bpmac[2]
    phase_corr = ac2bpmac[1] - phases_meas[k_bpmac] + (0.5 * driven_tune)
    phases_meas = phases_meas + phase_corr[np.newaxis, :]
    r = tunes.get_lambda(plane)
    phases_meas[k_bpmac:, :] = phases_meas[k_bpmac:, :] - driven_tune
    psi = (np.arctan((1 - r) / (1 + r) * np.tan(2 * np.pi * phases_meas)) / (2 * np.pi)) % 0.5
    phases_meas = np.where(phases_meas % 1.0 > 0.5, psi + .5, psi)
    phases_meas[k_bpmac:, :] = phases_meas[k_bpmac:, :] + free_tune
    return phases_meas


def _compensate_by_model(input_files: InputFiles, meas_input: DotDict, df: pd.DataFrame, plane: str) -> pd.DataFrame:
    """ Compensate the measured phase advances by comparing the driven model with the used model."""
    # add column from driven model to df (and merge indices)
    df_driven = pd.DataFrame(meas_input.accelerator.model_driven.loc[:, [f"{PHASE_ADV}{plane}"]])
    df = pd.merge(df, df_driven, how='inner', left_index=True, right_index=True, suffixes=("", "comp"))

    # phase compensation = difference between driven model and used model
    phase_compensation = df_diff(df, f"{PHASE_ADV}{plane}", f"{PHASE_ADV}{plane}comp")

    # add the compensated phase to the measured data
    df[input_files.get_columns(df, f"{PHASE_ADV}{plane}")] = ang_sum(
        input_files.get_data(df, f"{PHASE_ADV}{plane}"), phase_compensation[:, np.newaxis])
    return df


def write(
    dfs: Sequence[pd.DataFrame], 
    headers: Sequence[dict[str, Any]] | dict[str, Any], 
    output: str | Path, plane: str
    ):
    """ Write out the phase advance data into TFS-files."""
    LOGGER.info(f"Writing phases: {len(dfs)}")
    if isinstance(headers, dict):
        headers = [headers] * len(dfs)

    for head, df, name in zip(headers, dfs, (PHASE_NAME, TOTAL_PHASE_NAME, DRIVEN_PHASE_NAME, DRIVEN_TOTAL_PHASE_NAME)):
        tfs.write(Path(output) / f"{name}{plane.lower()}{EXT}", df, head)
        LOGGER.info(f"Phase advance beating in {name}{plane.lower()}{EXT} = "
                    f"{stats.weighted_rms(df.loc[:, f'{DELTA}{PHASE}{plane}'])}")


def _create_output_df(phase_advances: PhaseDict, model: pd.DataFrame, plane: str, tot: bool = False) -> pd.DataFrame:
    """ Create the DataFrames to be written out into TFS-files.

    Args:
        phase_advances (PhaseDict): Dictionary of phase-advance DataFrames 
                                    as described in :func:`calculate_with_compensation`. 
        model (pd.DataFrame): Model DataFrame (i.e. columns from twiss_elements + compensation). 
        plane (str): Plane we are using ("X" or "Y").
        tot (bool, optional): Create DataFrame for the total phase advance or BPM phase advances. 
                              Defaults to False.

    Returns:
        pd.DataFrame: Frame as would be expected in the output files (the .tfs to be written out).
    """
    meas = phase_advances[MEASUREMENT]
    mod = phase_advances[MODEL]
    err = phase_advances[f"{ERR}{MEASUREMENT}"]
    if tot:
        output_data = model.loc[:, [S, f"{PHASE_ADV}{plane}"]].iloc[:, :]
        output_data[NAME] = output_data.index
        output_data = output_data.assign(S2=model.at[model.index[0], S], NAME2=model.index[0])
        output_data[f"{PHASE}{plane}"] = meas.to_numpy()[0, :]
        output_data[f"{ERR}{PHASE}{plane}"] = err.to_numpy()[0, :]
        output_data[f"{PHASE}{plane}{MDL}"] = mod.to_numpy()[0, :]
    else:
        output_data = model.loc[:, [S, f"{PHASE_ADV}{plane}"]].iloc[:-1, :]
        output_data[NAME] = output_data.index
        output_data = output_data.assign(S2=model.loc[:, S].to_numpy()[1:], NAME2=model.index[1:].to_numpy())
        output_data[f"{PHASE}{plane}"] = np.diag(meas.to_numpy(), k=1)
        output_data[f"{ERR}{PHASE}{plane}"] = np.diag(err.to_numpy(), k=1)
        output_data[f"{PHASE}{plane}{MDL}"] = np.diag(mod.to_numpy(), k=1)
    output_data.rename(columns={f"{PHASE_ADV}{plane}": f"{PHASE_ADV}{plane}{MDL}"}, inplace=True)
    output_data[f"{DELTA}{PHASE}{plane}"] = df_ang_diff(output_data, f"{PHASE}{plane}", f"{PHASE}{plane}{MDL}")
    output_data[f"{ERR}{DELTA}{PHASE}{plane}"] = output_data.loc[:, f"{ERR}{PHASE}{plane}"].to_numpy()
    return output_data


def _get_all_phase_diff(phases_a: ArrayLike, phases_b: ArrayLike = None) -> ArrayLike:
    if phases_b is None:
        phases_b = phases_a
    return (phases_a[np.newaxis, :] - phases_b[:, np.newaxis]) % 1.0


def _get_square_data_frame(data, index) -> pd.DataFrame:
    return pd.DataFrame(data=data, index=index, columns=index)


def write_special(meas_input: DotDict, phase_advances: pd.DataFrame, plane_tune: float, plane: str):
    """ Writes out the special phase advances, if any given by the accelerator class. """
    # TODO REFACTOR AND SIMPLIFY
    accel: Accelerator = meas_input.accelerator

    important_phase_advances: list = accel.important_phase_advances()
    if not important_phase_advances:
        return

    meas = phase_advances[MEASUREMENT]
    beam_direction = accel.beam_direction
    elements = accel.elements
    special_phase_columns = ['ELEMENT1',
                             'ELEMENT2',
                             f'{PHASE}{plane}',
                             f'{ERR}{PHASE}{plane}',
                             f'{PHASE}{plane}_DEG',
                             f'{ERR}{PHASE}{plane}_DEG',
                             f'{PHASE}{plane}{MDL}',
                             f'{PHASE}{plane}{MDL}_DEG',
                             'BPM1',
                             'BPM2',
                             f'BPM_PHASE{plane}',
                             f'BPM_{ERR}{PHASE}{plane}',]
    to_concat_rows = []
    for elem1, elem2 in accel.important_phase_advances():
        mus1 = elements.loc[elem1, f"{PHASE_ADV}{plane}"] - elements.loc[:, f"{PHASE_ADV}{plane}"]
        minmu1 = abs(mus1.loc[meas.index]).idxmin()
        mus2 = elements.loc[:, f"{PHASE_ADV}{plane}"] - elements.loc[elem2, f"{PHASE_ADV}{plane}"]
        minmu2 = abs(mus2.loc[meas.index]).idxmin()
        bpm_phase_advance = meas.loc[minmu1, minmu2]
        model_value = elements.loc[elem2, f"{PHASE_ADV}{plane}"] - elements.loc[elem1, f"{PHASE_ADV}{plane}"]
        if (elements.loc[elem1, S] - elements.loc[elem2, S]) * beam_direction > 0.0:
            bpm_phase_advance += plane_tune
            model_value += plane_tune
        bpm_err = phase_advances[f"{ERR}{MEASUREMENT}"].loc[minmu1, minmu2]
        elems_to_bpms = -mus1.loc[minmu1] - mus2.loc[minmu2]
        ph_result = ((bpm_phase_advance + elems_to_bpms) * beam_direction)
        model_value = (model_value * beam_direction) % 1
        new_row = pd.DataFrame([[
                elem1,
                elem2,
                ph_result % 1,
                bpm_err,
                _to_deg(ph_result),
                bpm_err * 360,
                model_value,
                _to_deg(model_value),
                minmu1,
                minmu2,
                bpm_phase_advance,
                elems_to_bpms,
            ]], 
            columns=special_phase_columns,
        )
        to_concat_rows.append(new_row)

    special_phase_df = pd.concat(to_concat_rows, axis="index", ignore_index=True)
    tfs.write(Path(meas_input.outputdir) / f"{SPECIAL_PHASE_NAME}{plane.lower()}{EXT}", special_phase_df)


def _to_deg(phase: ArrayLike | float) -> ArrayLike | float:
    """ Convert from tune units (radians [2pi]) to -90 to +90 degrees. """
    phase = phase % 0.5 * 360
    if phase < 90:
        return phase
    return phase - 180
