from collections import defaultdict
import numpy as np
from correction.constants import MODEL, DIFF, VALUE
from utils import logging_tools
LOG = logging_tools.get_logger(__name__)


def append_model_to_measurement(model, measurement, keys):
    """

    Args:
        model:
        measurement:
        keys:

    Returns:

    """
    appenders = _get_model_appenders()
    meas = {}
    for key in keys:
        meas[key] = appenders[key](model, measurement[key], key)
    return meas


def _get_model_appenders():
    return defaultdict(lambda:  _get_model_generic, {
        'MUX': _get_model_phases, 'MUY': _get_model_phases,
        'BETX': _get_model_betabeat, 'BETY': _get_model_betabeat,
        'NDX': _get_model_norm_disp, 'Q': _get_model_tunes, })


def _get_model_generic(model, meas, key):
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.values, key].values
        meas[DIFF] = meas.loc[:, VALUE].values - meas.loc[:, MODEL].values
    return meas


def _get_model_phases(model, meas, key):
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = (model.loc[meas['NAME2'].values, key].values -
                       model.loc[meas.index.values, key].values)
        meas[DIFF] = meas.loc[:, VALUE].values - meas.loc[:, MODEL].values
    return meas


def _get_model_betabeat(model, meas, key):
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.values, key].values
        meas[DIFF] = (meas.loc[:, VALUE].values - meas.loc[:, MODEL].values) / meas.loc[:, MODEL].values
    return meas


def _get_model_norm_disp(model, meas, key):
    col = key[1:]
    beta = f"BET{key[-1]}"
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = model.loc[meas.index.values, col].values / np.sqrt(model.loc[meas.index.values, beta].values)
        meas[DIFF] = meas.loc[:, VALUE].values - meas.loc[:, MODEL].values
    return meas


def _get_model_tunes(model, meas, key):
    # We want just fractional tunes
    with logging_tools.log_pandas_settings_with_copy(LOG.debug):
        meas[MODEL] = np.remainder([model['Q1'], model['Q2']], [1, 1])
        meas[DIFF] = meas.loc[:, VALUE].values - meas.loc[:, MODEL].values
    return meas