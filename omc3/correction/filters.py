"""
Filters
-------

Filters for different kind of measurement data, and to filter the entries
in the response matrix based on (the presumably then filtered) measurement data.
Measurement filters extract valid (or trustworthy) data, e.g. to be used in
corrections.
The main function is :meth:`omc3.correction.filters.filter_measurement`
which decides on which filter to use for the given keys.

In earlier implementations there was a split between all kinds of measures,
i.e. beta, phase etc. In this implementation most of it is handled by
the `_get_filtered_generic` function.
"""
from collections import defaultdict
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
import tfs
from generic_parser import DotDict

from omc3.correction.constants import DELTA, ERR, ERROR, NAME2, PHASE, PHASE_ADV, TUNE, VALUE, WEIGHT
from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import AMPLITUDE, F1001, F1010, IMAG, REAL
from omc3.utils import logging_tools, stats

LOG = logging_tools.get_logger(__name__)


# Measurement Filter -----------------------------------------------------------


def filter_measurement(
    keys: Sequence[str], meas: Dict[str, pd.DataFrame], model: pd.DataFrame, opt: DotDict
) -> dict:
    """Filters measurements in `keys` based on the dict-entries (keys as in `keys`)
    in `opt.errorcut`, `opt.modelcut` and `opt.weights` and unifies the
    data-column names to VALUE, ERROR, WEIGHT.
    If `opt.use_errorbars` is `True` the weights will be also based on the errors."""
    filters = _get_measurement_filters()
    new = dict.fromkeys(keys)
    for key in keys:
        new[key] = filters[key](key, meas[key], model, opt)
    return new


def _get_measurement_filters() -> defaultdict:
    """Returns a dict with the respective `_get_*` filter-functions that defaults
    to `~_get_filtered_generic`. Some columns might need to have extra steps for
    filtering, or simply a different filtering process."""
    return defaultdict(
        lambda: _get_filtered_generic,
        {
            f"{TUNE}": _get_tunes,
            f"{F1010}I": _get_coupling,
            f"{F1010}R": _get_coupling,
            f"{F1001}I": _get_coupling,
            f"{F1001}R": _get_coupling,
        },
    )


def _get_filtered_generic(col: str, meas: pd.DataFrame, model: pd.DataFrame, opt: DotDict) -> tfs.TfsDataFrame:
    """
    Filters the provided column *col* of the measurement dataframe *meas*, based on the model values
    (from the *model* dataframe) and the filtering options given at the command line (for instance,
    the ``errorcut`` and ``modelcut`` values).

    Args:
        col (str): The column name to filter.
        meas (pd.DataFrame): The measurement dataframe
        model (pd.DataFrame): The model dataframe, which we get from the ``model_creator``.
        opt (DotDict): The command line options dictionary.

    Returns:
        The filtered dataframe as a `~tfs.TfsDataFrame`.
    """
    common_bpms = meas.index.intersection(model.index)
    meas = meas.loc[common_bpms, :]

    new = tfs.TfsDataFrame(index=common_bpms)
    new[VALUE] = meas.loc[:, col].to_numpy()
    new[ERROR] = meas.loc[:, f"{ERR}{col}"].to_numpy()
    new[WEIGHT] = (
        _get_errorbased_weights(col, opt.weights[col], meas.loc[:, f"{ERR}{DELTA}{col}"])
        if opt.use_errorbars
        else opt.weights[col]
    )

    # Applying filtering cuts
    error_filter = meas.loc[:, f"{ERR}{DELTA}{col}"].to_numpy() < opt.errorcut[col]
    model_filter = np.abs(meas.loc[:, f"{DELTA}{col}"].to_numpy()) < opt.modelcut[col]
    # if opt.automatic_model_cut:  # TODO automated model cut
    #     model_filter = _get_smallest_data_mask(np.abs(meas.loc[:, f"{DELTA}{col}"].to_numpy()), portion=0.95)
    if f"{PHASE}" in col:
        new[NAME2] = meas.loc[:, NAME2].to_numpy()
        second_bpm_in = np.in1d(new.loc[:, NAME2].to_numpy(), new.index.to_numpy())
        good_bpms = error_filter & model_filter & second_bpm_in
        good_bpms[-1] = False
    else:
        good_bpms = error_filter & model_filter
    LOG.debug(f"Number of BPMs kept for column '{col}' after filtering: {np.sum(good_bpms)}")
    return new.loc[good_bpms, :]


def _get_tunes(key: str, meas: pd.DataFrame, model, opt: DotDict):
    meas[WEIGHT] = opt.weights[key]
    if opt.use_errorbars:
        meas[WEIGHT] = _get_errorbased_weights(key, meas[WEIGHT], meas[ERROR])
    LOG.debug(f"Number of tune measurements: {len(meas.index.to_numpy())}")
    return meas


def _get_coupling(col: str, meas: pd.DataFrame, model: pd.DataFrame, opt: DotDict) -> tfs.TfsDataFrame:
    """
    Applies filters to the coupling dataframe *meas*. This is a bit hacky. Takes the measurement and
    model dataframes for one of the coupling RDTs (*meas* comes from **f1001.tfs** or **f1010.tfs**)
    maps the column *col* name to the "old" naming (for the F1001 for instance, REAL -> F1001R)
    before passing to `~_get_filtered_generic` which will do the filtering. This is because
    `~_get_filtered_generic` does a comparison to the model which has old names.

    Args:
        col (str): The column name to filter.
        meas (pd.DataFrame): The measurement dataframe, which here will be the loaded **f1001.tfs**
            or **f1010.tfs** file.
        model (pd.DataFrame): The model dataframe, which we get from the ``model_creator``.
        opt (DotDict): The command line options dictionary.

    Returns:
        The filtered dataframe as a `~tfs.TfsDataFrame`.
    """
    # rename measurement column to key
    column_map = {c[0]: c for c in [REAL, IMAG, AMPLITUDE, PHASE]}  # only REAL and IMAG implemented in responses so far
    meas_col = column_map[col[-1]]
    meas.columns = meas.columns.str.replace(meas_col, col)
    return _get_filtered_generic(col, meas, model, opt)


def _get_errorbased_weights(key: str, weights, errors):
    # TODO case without errors used may corrupt the correction (typical error != 1)
    w2 = stats.weights_from_errors(errors)
    if w2 is None:
        LOG.warn(
            f"Weights will not be based on errors for '{key}'"
            f", zeros of NaNs were found. Maybe don't use --errorbars."
        )
        return weights
    return weights * np.sqrt(w2)


# Response Matrix Filter -------------------------------------------------------


def filter_response_index(response: Dict, measurement: Dict, keys: Sequence[str]):
    """Filters the index of the response matrices `response` by the respective entries in `measurement`."""
    # rename MU to PHASE as we create a PHASE-Response afterwards
    # easier to do here, than to check eveywhere below. (jdilly)
    _rename_phase_advance(response)

    not_in_response = [key for key in keys if key not in response]
    if len(not_in_response) > 0:
        raise KeyError(
            f"The following optical parameters are not present in current"
            f"response matrix: {not_in_response}"
        )

    filters = _get_response_filters()
    new_response = {}
    for key in keys:
        new_response[key] = filters[key](response[key], measurement[key])
    return new_response


def _get_response_filters() -> Dict[str, Callable]:
    """
    Returns a dict with the respective `_get_*_response` functions that defaults
    to `_get_generic_response`.
    """
    return defaultdict(
        lambda: _get_generic_response,
        {f"{PHASE}X": _get_phase_response, f"{PHASE}Y": _get_phase_response},
    )


def _get_generic_response(resp: pd.DataFrame, meas: pd.DataFrame) -> pd.DataFrame:
    return resp.loc[meas.index.to_numpy(), :]


def _get_phase_response(resp: pd.DataFrame, meas: pd.DataFrame) -> pd.DataFrame:
    """Creates response for PHASE, not MU."""
    phase1 = resp.loc[meas.index.to_numpy(), :]
    phase2 = resp.loc[meas.loc[:, NAME2].to_numpy(), :]
    return -phase1.sub(phase2.to_numpy())  # phs2-phs1 but with idx of phs1


def _get_smallest_data_mask(data, portion: float = 0.95) -> np.ndarray:
    if not 0 <= portion <= 1:
        raise ValueError("Portion of data has to be between 0 and 1")
    b = int(len(data) * portion)
    mask = np.ones_like(data, dtype=bool)
    mask[np.argpartition(data, b)[b:]] = False
    return mask


def _rename_phase_advance(response):
    """Renames MU to PHASE inplace."""
    for plane in PLANES:
        try:
            response[f"{PHASE}{plane}"] = response.pop(f"{PHASE_ADV}{plane}")
        except KeyError:
            pass
