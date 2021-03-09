"""
Isolation Forest
----------------

This module contains the isolation forest functionality of ``optics_measurements``.
It provides functions to detect and exclude BPMs with anomalies.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from omc3.definitions.constants import PLANE_TO_NUM
from omc3.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)
ARCS_CONT = 0.01
IRS_CONT = 0.025


def clean_with_isolation_forest(input_files, meas_input, plane):
    bad_bpms = identify_bad_bpms(meas_input, input_files, plane)
    input_files = remove_bad_bpms(input_files, list(set(bad_bpms.NAME)), plane)
    LOGGER.info(str(list(set(bad_bpms.NAME))))
    # TODO potentially write output files ... currently not unique indices!
    #  tfs.write(os.path.join(meas_input.outputdir, f"bad_bpms_iforest_{plane.lower()}.tfs"), bad_bpms)
    return input_files


def identify_bad_bpms(meas_input, input_files, plane):
    bpm_data = pd.concat([tfs_df[["NAME", f"TUNE{plane}", "NOISE_SCALED", f"AMP{plane}"]]
                          for tfs_df in input_files])
    arc_bpm_data, ir_bpm_data = get_data_for_clustering(bpm_data, plane, meas_input.accelerator)
    return pd.concat([identify_single_cluster_bad_bpms(bpm_data, ARCS_CONT, arc_bpm_data, plane),
                      identify_single_cluster_bad_bpms(bpm_data, IRS_CONT, ir_bpm_data, plane)])


def identify_single_cluster_bad_bpms(bpm_tfs_data, cont, data_for_clustering, plane):
    bad_bpms, good_bpms, bad_bpms_scores = detect_anomalies(cont, data_for_clustering, plane)
    bpm_tfs_data, data_for_clustering, bad_bpms, good_bpms = \
        [reassign_index(data) for data in (bpm_tfs_data, data_for_clustering, bad_bpms, good_bpms)]
    signif_feature = get_significant_features(bpm_tfs_data, data_for_clustering, bad_bpms,
                                              good_bpms, plane)
    signif_feature.loc[:, "SCORE"] = bad_bpms_scores
    return signif_feature


def reassign_index(data):
    data["NEW_INDEX"] = range(len(data.NAME))
    return data.set_index("NEW_INDEX")


def get_significant_features(bpm_tfs_data, data_for_clustering, bad_bpms, good_bpms, plane):
    features_df = pd.DataFrame(index=bad_bpms.index)
    for index in bad_bpms.index:
        max_dist = max([(abs(data_for_clustering.loc[index, col] -
                             good_bpms.loc[:, col].mean()), col)
                        for col in [f"TUNE{plane}", "NOISE_SCALED", f"AMP{plane}"]])
        max_dist, sig_col = max_dist
        features_df.loc[index, "NAME"] = bad_bpms.loc[index, "NAME"]
        features_df.loc[index, "FEATURE"] = sig_col
        features_df.loc[index, "VALUE"] = bpm_tfs_data.loc[index, sig_col]
        features_df.loc[index, "AVG"] = np.mean(bpm_tfs_data.loc[good_bpms.index][sig_col])
    return features_df


def detect_anomalies(contamination, data, plane):
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
