"""
Model Diff
----------

Calculate the differences in optics parameters between twiss-models.
Similar to :func:`omc3.correction.model_appenders.add_differences_to_model_to_measurements`,
yet operates on two twiss files instead.

"""
from collections import defaultdict
from typing import Sequence

import numpy as np

import tfs
from omc3.optics_measurements.constants import DISPERSION, BETA, PHASE_ADV, TUNE, PHASE, NORM_DISPERSION, DELTA
from omc3.optics_measurements.toolbox import ang_diff


def diff_twiss_parameters(model_a: tfs.TfsDataFrame,
                          model_b: tfs.TfsDataFrame,
                          parameters: Sequence[str] = None) -> tfs.TfsDataFrame:
    """Create a TfsDataFrame containing of the given parameters between
    model_a and model_b."""
    # preparation ---
    if parameters is None:
        parameters = model_a.columns.intersection(model_b.columns)

    index = model_a.index.intersection(model_b.index)
    model_a, model_b = model_a.loc[index, :], model_b.loc[index, :]

    # get diff dataframe ---
    with_tune = TUNE in parameters
    if with_tune:
        parameters = [p for p in parameters if p != TUNE]

    diff_df = tfs.TfsDataFrame(index=index, columns=[f"{DELTA}{p}" for p in parameters])
    fun_map = _get_mapping()
    for parameter in parameters:
        diff_function = fun_map[parameter[:-1]]
        diff_df.loc[:, f"{DELTA}{parameter}"] = diff_function(model_a, model_b, parameter)

    if with_tune:
        for tune in (f"{TUNE}1", f"{TUNE}2"):
            diff_df.headers[f"{DELTA}{tune}"] = _tune_diff(model_a, model_b, tune)

    return diff_df


# Helper -----------------------------------------------------------------------


def _get_mapping():
    return defaultdict(
        lambda: _diff,
        {
            PHASE_ADV: _phase_advance_diff,
            PHASE: _phase_advance_diff,
            BETA: _beta_beating,
            NORM_DISPERSION: _normalized_dispersion_diff,
        }
    )


def _diff(model_a, model_b, column):
    return model_a.loc[:, column] - model_b.loc[:, column]


def _tune_diff(model_a, model_b, column):
    return model_a.headers[column] % 1 - model_b.headers[column] % 1


def _phase_advance_diff(model_a, model_b, column):
    column = f"{PHASE_ADV}{column[-1]}"  # in case it's "PHASE", which is a parameter of the measurement
    name = model_a.index[:-1]
    name2 = model_a.index[1:]
    return np.append(
        [0],
        ang_diff(
            ang_diff(model_a.loc[name2, column].to_numpy(), model_a.loc[name, column].to_numpy()),
            ang_diff(model_b.loc[name2, column].to_numpy(), model_b.loc[name, column].to_numpy())
        )
    )


def _beta_beating(model_a, model_b, column):
    return np.divide(model_a.loc[:, column], model_b.loc[:, column]) - 1


def _normalized_dispersion_diff(model_a, model_b, column):
    plane = column[-1]
    return -np.diff([
        np.divide(model.loc[:, f"{DISPERSION}{plane}"], np.sqrt(model.loc[:, f"{BETA}{plane}"])).to_numpy()
        for model in (model_a, model_b)
    ], axis=0).T
