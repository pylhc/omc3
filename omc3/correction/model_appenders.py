from collections import defaultdict
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd

from omc3.correction.constants import DIFF, MODEL, VALUE
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


def append_model_to_measurement(
    model: pd.DataFrame,
    measurement: pd.DataFrame,
    keys: Sequence[str]
) -> Dict[str, pd.DataFrame]:
    """
    Provided with DataFrames from a model and a measurement, and a number of keys to be found in both,
    returns a dictionary with the variation from measurement to model for each key.

    Args:
        model (pd.DataFrame): DataFrame of the model values.
        measurement (pd.DataFrame): DataFrame of the measurement values.
        keys (Sequence[str]): keys to get variation to model for.

    Returns:
        A dictionary
    """
    appenders: Dict[str, Callable] = _get_model_appenders()
    res_dict = {}
    for key in keys:
        res_dict[key] = appenders[key](model, measurement[key], key)
    return res_dict


def _get_model_appenders() -> Dict[str, Callable]:
    return defaultdict(lambda:  _get_model_generic, {
        'MUX': _get_model_phases, 'MUY': _get_model_phases,
        'BETX': _get_model_betabeat, 'BETY': _get_model_betabeat,
        'NDX': _get_model_norm_disp, 'Q': _get_model_tunes, })


def _get_model_generic(model: pd.DataFrame, meas: pd.DataFrame, key: str) -> pd.DataFrame:
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.values, key].values
        meas[DIFF] = meas.loc[:, VALUE].values - meas.loc[:, MODEL].values
    return meas


def _get_model_phases(model: pd.DataFrame, meas: pd.DataFrame, key: str) -> pd.DataFrame:
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = (model.loc[meas['NAME2'].values, key].values -
                       model.loc[meas.index.values, key].values)
        meas[DIFF] = meas.loc[:, VALUE].values - meas.loc[:, MODEL].values
    return meas


def _get_model_betabeat(model: pd.DataFrame, meas: pd.DataFrame, key: str) -> pd.DataFrame:
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.values, key].values
        meas[DIFF] = (meas.loc[:, VALUE].values - meas.loc[:, MODEL].values) / meas.loc[:, MODEL].values
    return meas


def _get_model_norm_disp(model: pd.DataFrame, meas: pd.DataFrame, key: str) -> pd.DataFrame:
    col = key[1:]
    beta = f"BET{key[-1]}"
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.values, col].to_numpy() / np.sqrt(model.loc[meas.index.values, beta].to_numpy())
        meas[DIFF] = meas.loc[:, VALUE].values - meas.loc[:, MODEL].values
    return meas


def _get_model_tunes(model: pd.DataFrame, meas: pd.DataFrame, key: str) -> pd.DataFrame:
    # We want just fractional tunes
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = np.remainder([model['Q1'], model['Q2']], [1, 1])
        meas[DIFF] = meas.loc[:, VALUE].values - meas.loc[:, MODEL].values
    return meas
