"""
Chromatic
----------------

This module contains chromatic calculations functionality of ``optics_measurements``.
It provides functions to compute various chromatic beam properties.
"""
import numpy as np
import pandas as pd

from omc3.optics_measurements.constants import DELTA, ERR, MDL
from omc3.optics_measurements.toolbox import df_prod, df_ratio


def calculate_w_and_phi(betas, dpps, input_files, measure_input, plane):
    columns = [f"{pref}{DELTA}{col}{plane}" for pref in ("", ERR) for col in ("BET", "ALF")]
    joined = betas[0].loc[:, columns]
    for i, beta in enumerate(betas[1:]):
        joined = pd.merge(joined, beta.loc[:, columns], how="inner", left_index=True,
                          right_index=True, suffixes=('', '__' + str(i + 1)))
    for column in columns:
        joined.rename(columns={column: column + '__0'}, inplace=True)
    joined = pd.merge(joined,
                      betas[np.argmin(np.abs(dpps))].loc[:, [f"ALF{plane}", f"{ERR}ALF{plane}"]],
                      how="inner",
                      left_index=True, right_index=True)
    for col in ("BET", "ALF"):
        fit = np.polyfit(np.repeat(dpps, 2),
                         np.repeat(input_files.get_data(joined, f"{DELTA}{col}{plane}").T, 2,
                                   axis=0), 1, cov=True)
        joined[f"D{col}{plane}"] = fit[0][-2, :].T
        joined[f"{ERR}D{col}{plane}"] = np.sqrt(fit[1][-2, -2, :].T)
    a = joined.loc[:, f"DBET{plane}"].to_numpy()
    aerr = joined.loc[:, f"{ERR}DBET{plane}"].to_numpy()
    b = joined.loc[:, f"DALF{plane}"].to_numpy() - joined.loc[:, f"ALF{plane}"].to_numpy() * joined.loc[:,
                                                                                     f"DBET{plane}"].to_numpy()
    berr = np.sqrt(df_prod(joined, f"{ERR}DALF{plane}", f"{ERR}DALF{plane}") +
                   np.square(df_prod(joined, f"{ERR}ALF{plane}", f"DBET{plane}")) +
                   np.square(df_prod(joined, f"ALF{plane}", f"{ERR}DBET{plane}")))
    w = np.sqrt(np.square(a) + np.square(b))
    joined[f"W{plane}"] = w
    joined[f"{ERR}W{plane}"] = np.sqrt(np.square(a * aerr / w) + np.square(b * berr / w))
    joined[f"PHI{plane}"] = np.arctan2(b, a) / (2 * np.pi)
    joined[f"{ERR}PHI{plane}"] = 1 / (1 + np.square(a / b)) * np.sqrt(
        np.square(aerr / b) + np.square(berr * a / np.square(b))) / (2 * np.pi)
    output_df = pd.merge(measure_input.accelerator.model.loc[:,
                         ["S", f"MU{plane}", f"BET{plane}", f"ALF{plane}", f"W{plane}",
                          f"PHI{plane}"]],
                         joined.loc[:,
                         [f"{pref}{col}{plane}" for pref in ("", ERR) for col in ("W", "PHI")]],
                         how="inner", left_index=True,
                         right_index=True, suffixes=(MDL, ''))
    output_df.rename(columns={"SMDL": "S"}, inplace=True)
    return output_df


def calculate_chromatic_coupling(couplings, dpps, input_files, measure_input):
    # TODO how to treat the model values?
    columns = [f"{pref}{col}{part}" for pref in ("", ERR) for col in ("F1001", "F1010") for part in ("RE", "IM")]
    joined = couplings[0].loc[:, columns]
    for i, coup in enumerate(couplings[1:]):
        joined = pd.merge(joined, coup.loc[:, columns], how="inner", left_index=True,
                          right_index=True, suffixes=('', '__' + str(i + 1)))
    for column in columns:
        joined.rename(columns={column: column + '__0'}, inplace=True)

    for col in ("F1001", "F1010"):
        for part in ("RE", "IM"):
            fit = np.polyfit(np.repeat(dpps, 2),
                             np.repeat(input_files.get_data(joined, f"{col}{part}").T, 2,
                                       axis=0), 1, cov=True)
            joined[f"D{col}{part}"] = fit[0][-2, :].T
            joined[f"{ERR}D{col}{part}"] = np.sqrt(fit[1][-2, -2, :].T)
        joined[f"D{col}"] = np.sqrt(np.square(joined.loc[:, f"D{col}RE"].to_numpy()) + np.square(joined.loc[:, f"D{col}IM"].to_numpy()))
        joined[f"{ERR}D{col}"] = np.sqrt(np.square(joined.loc[:, f"D{col}RE"].to_numpy() * df_ratio(joined, f"{ERR}D{col}RE", f"D{col}")) +
                                         np.square(joined.loc[:, f"D{col}IM"].to_numpy() * df_ratio(joined, f"{ERR}D{col}IM", f"D{col}")))
    output_df = pd.merge(measure_input.accelerator.model.loc[:, ["S"]], joined.loc[:,
                         [f"{pref}{col}{part}" for pref in ("", ERR) for col in ("F1001", "F1010") for part in ("", "RE", "IM")]],
                         how="inner", left_index=True,
                         right_index=True)
    return output_df
