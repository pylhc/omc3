"""
Isolation Forest
----------------

This module contains the isolation forest functionality of ``optics_measurements``.
It provides functions to detect and exclude BPMs with anomalies.

TODO:
After discussion with lmalina: he thinks that we should not run Isolation Forest in
its current form, as the binning of the frequencies in Harpy automatically leads to
some clustering, which might then trigger wrong tune-filtering in here.
This should be tested and possibly mitigated. (jdilly, 2024)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs
from sklearn.ensemble import IsolationForest

from omc3.definitions.constants import PLANE_TO_NUM
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
    from logging import Logger

    from generic_parser import DotDict

LOGGER: Logger = logging_tools.get_logger(__name__)
ARCS_CONT = 0.01
IRS_CONT = 0.025

# Columns ---
FEATURE: str = "FEATURE"


def clean_with_isolation_forest(
    input_files: Sequence[tfs.TfsDataFrame], meas_input: DotDict, plane: str,
) -> Sequence[tfs.TfsDataFrame]:
    """
    Runs isolation forest anomaly detection and removes flagged BPMs from input files.
    For every file where a cleaning is performed, a TfsDataFrame with information on
    the identified and cleaned BPMs is written to disk in an adjacent location in the
    corresponding output dir.

    Args:
        input_files: list of measurement DataFrames.
        meas_input: `OpticsInput` object containing analysis settings.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        The input files with anomalous BPMs removed.
    """
    bad_bpms = identify_bad_bpms(meas_input, input_files, plane)
    input_files = remove_bad_bpms(input_files, list(set(bad_bpms.NAME)), plane)
    LOGGER.info(str(list(set(bad_bpms.NAME))))
    if meas_input.outputdir is not None:
        tfs.write(Path(meas_input.outputdir)/ f"bad_bpms_iforest_{plane.lower()}.tfs", bad_bpms)
    return input_files


def identify_bad_bpms(
    meas_input: DotDict,
    input_files: Sequence[tfs.TfsDataFrame],
    plane: str,
) -> pd.DataFrame:
    """
    Identifies anomalous BPMs across arc and IR regions using isolation forest.

    Args:
        meas_input: `OpticsInput` object containing analysis settings.
        input_files: list of measurement DataFrames.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        A `DataFrame` listing the detected bad BPMs with their significant features and scores.
    """
    bpm_data = pd.concat(
                            [tfs_df[["NAME", f"TUNE{plane}", "NOISE_SCALED", f"AMP{plane}"]]
                            for tfs_df in input_files]
                        )
    arc_bpm_data, ir_bpm_data = get_data_for_clustering(bpm_data, plane, meas_input.accelerator)
    return pd.concat([identify_single_cluster_bad_bpms(bpm_data, ARCS_CONT, arc_bpm_data, plane),
                      identify_single_cluster_bad_bpms(bpm_data, IRS_CONT, ir_bpm_data, plane)])


def identify_single_cluster_bad_bpms(
    bpm_tfs_data: pd.DataFrame,
    cont: float,
    data_for_clustering: pd.DataFrame,
    plane: str,
) -> pd.DataFrame:
    """Runs isolation forest on a single cluster and returns the significant features of detected anomalies.

    Args:
        bpm_tfs_data: concatenated BPM data from all input files.
        cont: contamination parameter for the isolation forest.
        data_for_clustering: normalized feature data for clustering.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        A `DataFrame` with anomalous BPM names, their most significant feature, and anomaly scores.
    """
    bad_bpms, good_bpms, bad_bpms_scores = detect_anomalies(cont, data_for_clustering, plane)
    bpm_tfs_data, data_for_clustering, bad_bpms, good_bpms = \
        [reassign_index(data) for data in (bpm_tfs_data, data_for_clustering, bad_bpms, good_bpms)]
    signif_feature = get_significant_features(bpm_tfs_data, data_for_clustering, bad_bpms,
                                              good_bpms, plane)
    signif_feature.loc[:, "SCORE"] = bad_bpms_scores
    return signif_feature


def reassign_index(data: pd.DataFrame) -> pd.DataFrame:
    """Resets the index of the DataFrame to a sequential integer range."""
    data["NEW_INDEX"] = range(len(data.NAME))
    return data.set_index("NEW_INDEX")


def get_significant_features(
    bpm_tfs_data: pd.DataFrame,
    data_for_clustering: pd.DataFrame,
    bad_bpms: pd.DataFrame,
    good_bpms: pd.DataFrame,
    plane: str,
) -> pd.DataFrame:
    """Determines the most significant feature for each anomalous BPM.

    For each bad BPM, finds the feature (tune, noise, or amplitude) with the
    largest deviation from the mean of good BPMs.

    Args:
        bpm_tfs_data: concatenated BPM data from all input files.
        data_for_clustering: normalized feature data used for clustering.
        bad_bpms: DataFrame of BPMs flagged as anomalous.
        good_bpms: DataFrame of BPMs considered normal.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        A `DataFrame` indexed like ``bad_bpms`` with columns NAME, FEATURE, VALUE, and AVG.
    """
    features_df = pd.DataFrame(index=bad_bpms.index)
    for index in bad_bpms.index:
        max_dist = max([(abs(data_for_clustering.loc[index, col] -
                             good_bpms.loc[:, col].mean()), col)
                        for col in [f"TUNE{plane}", "NOISE_SCALED", f"AMP{plane}"]])
        max_dist, sig_col = max_dist
        features_df.loc[index, "NAME"] = bad_bpms.loc[index, "NAME"]
        features_df.loc[index, FEATURE] = sig_col
        features_df.loc[index, "VALUE"] = bpm_tfs_data.loc[index, sig_col]
        features_df.loc[index, "AVG"] = np.mean(bpm_tfs_data.loc[good_bpms.index][sig_col])
    return features_df


def detect_anomalies(
    contamination: float, data: pd.DataFrame, plane: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Fits an isolation forest and separates data into anomalous and normal BPMs.

    Args:
        contamination: expected proportion of outliers in the data.
        data: normalized BPM feature data.
        plane: marking the horizontal or vertical plane, **X** or **Y**.

    Returns:
        A tuple of (bad_bpms, good_bpms, bad_bpms_scores).
    """
    iforest = IsolationForest(n_estimators=100, max_samples='auto',
                              contamination=contamination, max_features=1.0,
                              bootstrap=False)
    features = data[[f"TUNE{plane}", "NOISE_SCALED", f"AMP{plane}"]]
    iforest.fit(features)
    labels = iforest.predict(features)
    bad_bpms = data.iloc[np.where(labels == -1)].copy()
    good_bpms = data.iloc[np.where(labels != -1)].copy()
    bad_bpms_scores = iforest.decision_function(features.iloc[np.where(labels == -1)])
    return bad_bpms, good_bpms, bad_bpms_scores


def get_data_for_clustering(bpm_tfs_data, plane, accelerator):
    arc_bpm_mask = accelerator.get_element_types_mask(bpm_tfs_data.NAME, types=["arc_bpm"])
    ir_bpm_data_for_clustering = bpm_tfs_data.iloc[~arc_bpm_mask].copy()
    arc_bpm_data_for_clustering = bpm_tfs_data.iloc[arc_bpm_mask].copy()
    for col in [f"TUNE{plane}", "NOISE_SCALED", f"AMP{plane}"]:
        ir_bpm_data_for_clustering.loc[:, col] = _normalize_parameter(ir_bpm_data_for_clustering.loc[:, col])
        arc_bpm_data_for_clustering.loc[:, col] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, col])
    return arc_bpm_data_for_clustering, ir_bpm_data_for_clustering


def _normalize_parameter(column_data):
    return (column_data - column_data.min()) / (column_data.max() - column_data.min())


def remove_bad_bpms(tfs_dfs, bad_bpm_names, plane):
    for i in range(len(tfs_dfs)):
        tfs_dfs[i] = tfs_dfs[i].loc[~tfs_dfs[i].index.isin(bad_bpm_names)]
        tfs_dfs[i].headers[f"Q{PLANE_TO_NUM[plane]}"] = np.mean(tfs_dfs[i][f"TUNE{plane}"])
        tfs_dfs[i].headers[f"Q{PLANE_TO_NUM[plane]}RMS"] = np.std(tfs_dfs[i][f"TUNE{plane}"])
    return tfs_dfs
