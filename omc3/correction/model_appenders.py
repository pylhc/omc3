"""
Model Appenders
---------------

Utilities to append new columns to measurement and model dataframes.
E.g. get differences between measurement and model and append those to
the measurement data (for corrections).
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from optics_functions.coupling import coupling_via_cmatrix

from omc3.definitions.constants import PI2
from omc3.optics_measurements.constants import (BETA, NAME2, NORM_DISPERSION, PHASE_ADV,
                                       TUNE, F1001, F1010, PHASE)
from omc3.correction.constants import DIFF, MODEL, VALUE
from omc3.utils import logging_tools
from omc3.optics_measurements.toolbox import df_diff, df_rel_diff


if TYPE_CHECKING:   
    from collections.abc import Callable, Sequence


LOG = logging_tools.get_logger(__name__)


def add_differences_to_model_to_measurements(
    model: pd.DataFrame,
    measurement: dict[str, pd.DataFrame],
    keys: Sequence[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Provided with DataFrames from a model and a measurement, and a number of keys to be found in both,
    returns a dictionary with the variation from measurement to model for each key.

    Args:
        model (pd.DataFrame): DataFrame of the model.
        measurement (Dict[str, pd.DataFrame]): DataFrames of the measurement.
        keys (Sequence[str]): Parameters to get variation to model for. Optional.
                              If omitted, all entries in `measurement` are used.

    Returns:
        A dictionary of optics parameters and the resulting DataFrames.
    """
    appenders: dict[str, Callable] = _get_model_appenders()
    if keys is None:
        keys = measurement.keys()

    res_dict = {}

    for key in keys:
        res_dict[key] = appenders[key](model, measurement[key].copy(), key)
    return res_dict


def _get_model_appenders() -> dict[str, Callable]:
    return defaultdict(lambda:  _get_model_generic, {
        f"{PHASE}X": _get_model_phases, f"{PHASE}Y": _get_model_phases,
        f"{BETA}X": _get_model_betabeat, f"{BETA}Y": _get_model_betabeat,
        f"{NORM_DISPERSION}X": _get_model_norm_disp, f"{TUNE}": _get_model_tunes, })


def _get_model_generic(model: pd.DataFrame, meas: pd.DataFrame, key: str) -> pd.DataFrame:
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.to_numpy(), key].to_numpy()
        meas[DIFF] = df_diff(meas, VALUE, MODEL)
    return meas


def _get_model_phases(model: pd.DataFrame, meas: pd.DataFrame, key: str) -> pd.DataFrame:
    plane = key[-1]
    tunes = {'X':model.headers[f'{TUNE}1'],
             'Y':model.headers[f'{TUNE}2'],
             }
    model_column = f"{PHASE_ADV}{plane}"
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        model_phases_advances = (model.loc[meas[NAME2].to_numpy(), model_column].to_numpy() -
                                 model.loc[meas.index.to_numpy(), model_column].to_numpy())
        model_phases_advances[model_phases_advances < 0] += tunes[plane]
        meas[MODEL] = model_phases_advances
        meas[DIFF] = df_diff(meas, VALUE, MODEL)
    return meas


def _get_model_betabeat(model: pd.DataFrame, meas: pd.DataFrame, key: str) -> pd.DataFrame:
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.to_numpy(), key].to_numpy()
        meas[DIFF] = df_rel_diff(meas, VALUE, MODEL)
    return meas


def _get_model_norm_disp(model: pd.DataFrame, meas: pd.DataFrame, key: str) -> pd.DataFrame:
    col = key[1:]
    beta = f"{BETA}{key[-1]}"
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.to_numpy(), col].to_numpy() / np.sqrt(model.loc[meas.index.to_numpy(), beta].to_numpy())
        meas[DIFF] = meas.loc[:, VALUE].to_numpy() - meas.loc[:, MODEL].to_numpy()
    return meas


def _get_model_tunes(model: pd.DataFrame, meas: pd.DataFrame, key: str) -> pd.DataFrame:
    # We want just fractional tunes
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = np.remainder([model[f"{TUNE}1"], model[f"{TUNE}2"]], [1, 1])
        meas[DIFF] = meas.loc[:, VALUE].to_numpy() - meas.loc[:, MODEL].to_numpy()
    return meas


# ----

def add_coupling_to_model(model: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the coupling RDTs from the input model TfsDataFrame and returns a copy of said TfsDataFrame with
    columns for the real and imaginary parts of the computed coupling RDTs.

    Args:
        model (tfs.TfsDataFrame): Twiss dataframe.

    Returns:
        A TfsDataFrame with the added columns.
    """
    LOG.debug("Adding coupling columns to model.")
    result_tfs_df = model.copy()
    coupling_rdts_df = coupling_via_cmatrix(result_tfs_df)

    function_map = {"R": np.real, "I": np.imag, "A": np.abs, "P": lambda x: (np.angle(x) / PI2) % 1}

    for rdt in (F1001, F1010):
        for suffix, func in function_map.items():
            result_tfs_df[f"{rdt}{suffix}"] = func(coupling_rdts_df[rdt]).astype(np.float64)
    return result_tfs_df

