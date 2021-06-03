"""
Model Appenders
---------------

Utilities to append new columns to measurement and model dataframes.
E.g. get differences between measurement and model and append those to
the measurement data (for corrections).
"""
from collections import defaultdict
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
from optics_functions.coupling import coupling_via_cmatrix

from omc3.correction.constants import (BETA, DIFF, MODEL, NORM_DISP, PHASE_ADV,
                                       TUNE, VALUE, F1001, F1010, PHASE)
from omc3.utils import logging_tools
from omc3.optics_measurements.toolbox import df_diff, df_rel_diff

LOG = logging_tools.get_logger(__name__)


def add_differences_to_model_to_measurements(
    model: pd.DataFrame,
    measurement: Dict[str, pd.DataFrame],
    keys: Sequence[str] = None,
) -> Dict[str, pd.DataFrame]:
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
    appenders: Dict[str, Callable] = _get_model_appenders()
    if keys is None:
        keys = measurement.keys()

    res_dict = {}

    for key in keys:
        res_dict[key] = appenders[key](model, measurement[key].copy(), key)
    return res_dict


def _get_model_appenders() -> Dict[str, Callable]:
    return defaultdict(lambda:  _get_model_generic, {
        f"{PHASE}X": _get_model_phases, f"{PHASE}Y": _get_model_phases,
        f"{BETA}X": _get_model_betabeat, f"{BETA}Y": _get_model_betabeat,
        f"{NORM_DISP}X": _get_model_norm_disp, f"{TUNE}": _get_model_tunes, })


def _get_model_generic(model: pd.DataFrame, meas: pd.DataFrame, key: str) -> pd.DataFrame:
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.to_numpy(), key].to_numpy()
        meas[DIFF] = df_diff(meas, VALUE, MODEL)
    return meas


def _get_model_phases(model: pd.DataFrame, meas: pd.DataFrame, key: str) -> pd.DataFrame:
    model_column = f"{PHASE_ADV}{key[-1]}"
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = (model.loc[meas["NAME2"].to_numpy(), model_column].to_numpy() -
                       model.loc[meas.index.to_numpy(), model_column].to_numpy())
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
    result_tfs_df = model.copy()
    coupling_rdts_df = coupling_via_cmatrix(result_tfs_df)
    result_tfs_df[f"{F1001}R"] = np.real(coupling_rdts_df[f"{F1001}"]).astype(np.float64)
    result_tfs_df[f"{F1001}I"] = np.imag(coupling_rdts_df[f"{F1001}"]).astype(np.float64)
    result_tfs_df[f"{F1010}R"] = np.real(coupling_rdts_df[f"{F1010}"]).astype(np.float64)
    result_tfs_df[f"{F1010}I"] = np.imag(coupling_rdts_df[f"{F1010}"]).astype(np.float64)
    return result_tfs_df

