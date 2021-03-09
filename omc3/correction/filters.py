from collections import defaultdict
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
import tfs

from omc3.correction.constants import DELTA, ERR, ERROR, VALUE, WEIGHT
from omc3.utils import logging_tools, stats

LOG = logging_tools.get_logger(__name__)


def filter_measurement(keys: Sequence[str], meas: pd.DataFrame, model: pd.DataFrame, opt: dict) -> dict:
    """ Filters measurements and renames columns to VALUE, ERROR, WEIGHT"""
    filters = _get_measurement_filters()
    new = dict.fromkeys(keys)
    for key in keys:
        new[key] = filters[key](key, meas[key], model, opt)
    return new


def _get_measurement_filters() -> defaultdict:
    return defaultdict(lambda: _get_filtered_generic, {'Q': _get_tunes})


def _get_filtered_generic(key: str, meas: pd.DataFrame, model: pd.DataFrame, opt: dict) -> tfs.TfsDataFrame:
    common_bpms = meas.index.intersection(model.index)
    meas = meas.loc[common_bpms, :]
    new = tfs.TfsDataFrame(index=common_bpms)
    col = key if "MU" not in key else f"PHASE{key[-1]}"
    new[VALUE] = meas.loc[:, col].values
    new[ERROR] = meas.loc[:, f"{ERR}{col}"].values
    new[WEIGHT] = (_get_errorbased_weights(key, opt.weights[key], meas.loc[:, f"{ERR}{DELTA}{col}"])
                   if opt.use_errorbars else opt.weights[key])
    # filter cuts
    error_filter = meas.loc[:, f"{ERR}{DELTA}{col}"].values < opt.errorcut[key]
    model_filter = np.abs(meas.loc[:, f"{DELTA}{col}"].values) < opt.modelcut[key]
    # if opt.automatic_model_cut:  # TODO automated model cut
    #     model_filter = _get_smallest_data_mask(np.abs(meas.loc[:, f"{DELTA}{col}"].values), portion=0.95)
    if "MU" in key:
        new['NAME2'] = meas.loc[:, 'NAME2'].values
        second_bpm_in = np.in1d(new.loc[:, 'NAME2'].values, new.index.values)
        good_bpms = error_filter & model_filter & second_bpm_in
        good_bpms[-1] = False
    else:
        good_bpms = error_filter & model_filter
    LOG.debug(f"Number of BPMs with {key}: {np.sum(good_bpms)}")
    return new.loc[good_bpms, :]


def _get_tunes(key: str, meas: pd.DataFrame, model, opt: dict):
    meas[WEIGHT] = opt.weights[key]
    if opt.use_errorbars:
        meas[WEIGHT] = _get_errorbased_weights(key, meas[WEIGHT], meas[ERROR])
    LOG.debug(f"Number of tune measurements: {len(meas.index.values)}")
    return meas


def _get_errorbased_weights(key: str, weights, errors):
    # TODO case without errors used may corrupt the correction (typical error != 1)
    w2 = stats.weights_from_errors(errors)
    if w2 is None:
        LOG.warn(f"Weights will not be based on errors for '{key}'"
                 f", zeros of NaNs were found. Maybe don't use --errorbars.")
        return weights
    return weights * np.sqrt(w2)


def filter_response_index(response, measurement, keys: Sequence[str]):
    not_in_response = [key for key in keys if key not in response]
    if len(not_in_response) > 0:
        raise KeyError(f"The following optical parameters are not present in current"
                       f"response matrix: {not_in_response}")

    filters = _get_response_filters()
    new_response = {}
    for key in keys:
        new_response[key] = filters[key](response[key], measurement[key])
    return new_response


def _get_response_filters() -> Dict[str, Callable]:
    return defaultdict(lambda:  _get_generic_response, {
        'MUX': _get_phase_response, 'MUY': _get_phase_response,
        'Q': _get_tune_response})


def _get_generic_response(resp: pd.DataFrame, meas: pd.DataFrame) -> pd.DataFrame:
    return resp.loc[meas.index.values, :]


def _get_phase_response(resp: pd.DataFrame, meas: pd.DataFrame) -> pd.DataFrame:
    phase1 = resp.loc[meas.index.values, :]
    phase2 = resp.loc[meas.loc[:, 'NAME2'].values, :]
    return -phase1.sub(phase2.values)  # phs2-phs1 but with idx of phs1


def _get_tune_response(resp, meas):  # ???
    return resp


def _get_smallest_data_mask(data, portion: float = 0.95) -> np.ndarray:
    if not 0 <= portion <= 1:
        raise ValueError("Portion of data has to be between 0 and 1")
    b = int(len(data) * portion)
    mask = np.ones_like(data, dtype=bool)
    mask[np.argpartition(data, b)[b:]] = False
    return mask
