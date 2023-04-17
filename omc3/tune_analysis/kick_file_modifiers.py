"""
Kick File Modifiers
-------------------

Functions to add data to or extract data from **kick_ac** files.
"""
from pathlib import Path
from typing import Tuple, Union, Sequence

import numpy as np
import pandas as pd
import tfs
from scipy import odr
from tfs import TfsDataFrame

from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import KICK_NAME, TIME, ERR, NAT_TUNE, ACTION
from omc3.tune_analysis import bbq_tools
from omc3.tune_analysis.bbq_tools import OutlierFilterOpt, FilterOpts, MinMaxFilterOpt
from omc3.tune_analysis.constants import (get_odr_header_coeff_corrected,
                                          get_odr_header_err_coeff_corrected,
                                          get_odr_header_err_coeff, get_odr_header_coeff,
                                          get_natq_corr_col, get_corr_natq_err_col,
                                          get_natq_err_col, get_natq_col, get_bbq_col,
                                          get_mav_col, get_mav_err_col, get_used_in_mav_col,
                                          get_time_col, get_action_col, get_action_err_col,
                                          get_mav_window_header, get_outlier_limit_header,
                                          get_fine_window_header, get_fine_cut_header,
                                          get_min_tune_header, get_max_tune_header,
                                          AmpDetData, FakeOdrOutput, get_kick_out_name, COEFFICIENT, CORRECTED,
                                          get_odr_header_default,
                                          )
from omc3.utils import logging_tools
from omc3.utils.time_tools import CERNDatetime, get_cern_time_format

LOG = logging_tools.get_logger(__name__)


def _get_odr_headers(corrected):
    """Return Headers needed for ODR."""
    if corrected:
        return get_odr_header_coeff_corrected, get_odr_header_err_coeff_corrected
    return get_odr_header_coeff, get_odr_header_err_coeff


def _get_ampdet_columns(corrected):
    """Get columns needed for amplitude detuning."""
    if corrected:
        return get_natq_corr_col, get_corr_natq_err_col
    return get_natq_col, get_natq_err_col


# Data Addition ################################################################


def add_bbq_data(kick_df: pd.DataFrame, bbq_df: pd.DataFrame, column: str, bbq_column: str = None):
    """
    Add BBQ values from column to kickac dataframe into same column.

    Args:
        kick_df: kick `DataFrame`, which needs to have time as index, best load it with
            ``read_timed_dataframe()``.
        bbq_df: `DataFrame` of bbq data with time as index.
        column: column name to add the data into.
        bbq_column: column name to get the data from (same as `column` if not given).

    Returns:
        Modified kick `Dataframe`.
    """
    if bbq_column is None:
        bbq_column = column
    kick_indx = get_timestamp_index(kick_df.index)
    bbq_indx = get_timestamp_index(bbq_df.index)

    values = []
    for time in kick_indx:
        nearest_bbq_index = bbq_indx.get_indexer([time], method="nearest")[0]
        values.append(bbq_df[bbq_column].iloc[nearest_bbq_index])
    kick_df[column] = values
    return kick_df


def add_moving_average(kickac_df: TfsDataFrame, bbq_df: TfsDataFrame, filter_opt: FilterOpts,
                       ) -> Tuple[TfsDataFrame, TfsDataFrame]:
    """Adds the moving average of the bbq data to kickac_df and bbq_df."""
    LOG.debug("Calculating moving average.")

    if isinstance(filter_opt, OutlierFilterOpt):
        bbq_df = _filter_bbq_outliers(bbq_df, filter_opt)
    else:
        bbq_df = _filter_bbq_cut(bbq_df, filter_opt)

    for plane in PLANES:
        kickac_df = add_bbq_data(kickac_df, bbq_df, get_mav_col(plane))
        kickac_df = add_bbq_data(kickac_df, bbq_df, get_mav_err_col(plane))
    return kickac_df, bbq_df


def _filter_bbq_outliers(bbq_df: tfs.TfsDataFrame, filter_opt: OutlierFilterOpt) -> tfs.TfsDataFrame:
    header_limit = get_outlier_limit_header()
    header_window = get_mav_window_header()

    # check if we need to recompute (as this might take quite some time)
    if (
            (header_limit in bbq_df.headers)
            and (bbq_df.headers[header_limit] == filter_opt.limit)
            and (bbq_df.headers[header_window] == filter_opt.window)
            and all(col in bbq_df.columns for p in PLANES for col in (
                    get_mav_col(p), get_mav_err_col(p), get_used_in_mav_col(p)))
    ):
        LOG.info("BBQ data has already been filtered with the same parameters. Using data from file.")
        return bbq_df

    bbq_df.headers[header_window] = filter_opt.window
    bbq_df.headers[header_limit] = filter_opt.limit
    for plane in PLANES:
        bbq_mav, bbq_err, mask = bbq_tools.clean_outliers_moving_average(bbq_df[get_bbq_col(plane)],
                                                                         filter_opt=filter_opt)
        bbq_df[get_mav_col(plane)] = bbq_mav
        bbq_df[get_mav_err_col(plane)] = bbq_err  # TODO to be discussed with Ewen and Tobias (jdilly, 2022-05-23)
        bbq_df[get_used_in_mav_col(plane)] = mask
    return bbq_df


def _filter_bbq_cut(bbq_df: tfs.TfsDataFrame, filter_opts: Sequence[MinMaxFilterOpt]) -> tfs.TfsDataFrame:
    bbq_df.headers[get_mav_window_header()] = filter_opts[0].window
    bbq_df.headers[get_fine_window_header()] = filter_opts[0].fine_window
    bbq_df.headers[get_fine_cut_header()] = filter_opts[0].fine_cut

    for idx, plane in enumerate(PLANES):
        bbq_mav, bbq_err, mask = bbq_tools.get_moving_average(bbq_df[get_bbq_col(plane)], filter_opts[idx])

        bbq_df.headers[get_min_tune_header(plane)] = filter_opts[idx].min
        bbq_df.headers[get_max_tune_header(plane)] = filter_opts[idx].max

        bbq_df[get_mav_col(plane)] = bbq_mav
        bbq_df[get_mav_err_col(plane)] = bbq_err  # TODO to be discussed with Ewen and Tobias (jdilly, 2022-05-23)
        bbq_df[get_used_in_mav_col(plane)] = mask
    return bbq_df


def add_corrected_natural_tunes(kickac_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the corrected natural tunes to ``kickac_df``.
    Add also the total standard deviation of the natural tune to the kickac dataframe.
    The total standard deviation is here defined as the standard deviation of
    the measurement plus the standard deviation of the moving average.

    Args:
        kickac_df: `Dataframe` containing the data.

    Returns:
        Modified kick_ac.
    """
    for plane in PLANES:
        # NATQ
        kickac_df[get_natq_corr_col(plane)] = (
            kickac_df[get_natq_col(plane)] - kickac_df[get_mav_col(plane)]
        )

        # ERROR
        kickac_df[get_corr_natq_err_col(plane)] = np.sqrt(
            np.power(kickac_df[get_natq_err_col(plane)], 2) +
            np.power(kickac_df[get_mav_err_col(plane)], 2)
        )
    return kickac_df


def add_odr(kickac_df: pd.DataFrame, odr_fit: odr.Output, 
            action_plane: str, tune_plane: str, corrected: bool = False):
    """
    Adds the odr fit of the (un)corrected data to the header of the ``kickac_df``.

    Args:
        kickac_df: `Dataframe` containing the data.
        odr_fit: odr-fit data (definitions see ``detuning_tools.py``).
        action_plane: Plane of the action.
        tune_plane: Plane of the tune.
        corrected: (BBQ) corrected data or uncorrected fit?.

    Returns:
        Modified kick_ac.
    """
    header_val, header_err = _get_odr_headers(corrected)
    for idx in range(len(odr_fit.beta)):
        kickac_df.headers[header_val(j_plane=action_plane, q_plane=tune_plane, order=idx)] = odr_fit.beta[idx]
        kickac_df.headers[header_err(j_plane=action_plane, q_plane=tune_plane, order=idx)] = odr_fit.sd_beta[idx]
    return kickac_df


# Data Extraction --------------------------------------------------------------

def get_odr_data(kickac_df: pd.DataFrame, action_plane: str, tune_plane: str,
                 order: int, corrected: bool = False) -> FakeOdrOutput:
    """
    Extract the data from kickac.

    Args:
        kickac_df: `Dataframe` containing the data.
        action_plane: Plane of the action.
        tune_plane: Plane of the tune.
        order: Order of the odr fit.
        corrected: (BBQ) corrected data or uncorrected fit?

    Returns:
        `Dictionary` containing fit data from odr.
    """

    header_val, header_err = _get_odr_headers(corrected)
    odr_data = FakeOdrOutput(beta=[0] * (order+1), sd_beta=[0] * (order+1))
    for idx in range(order+1):
        try:
            odr_data.beta[idx] = kickac_df.headers[header_val(q_plane=tune_plane, j_plane=action_plane, order=idx)]
            odr_data.sd_beta[idx] = kickac_df.headers[header_err(q_plane=tune_plane, j_plane=action_plane, order=idx)]
        except KeyError as e:
            LOG.debug(f"Fit data for order {order} not found. ({str(e)})")
    return odr_data


def get_ampdet_data(kickac_df: pd.DataFrame, action_plane: str, tune_plane: str,
                    corrected: bool = False, dropna: bool = True) -> AmpDetData:
    """
    Extract the data needed for the (un)corrected amplitude detuning from ``kickac_df``.

    Args:
        kickac_df (DataFrame): Containing the data
                              (either `kick_x`, `kick_y` from optics
                              or `kick_ampdet_xy` from ampdet analysis)..
        action_plane (str): Plane of the action.
        tune_plane (str): Plane of the tune.
        corrected (bool): if the BBQ-corrected columns should be taken
        dropna (bool): drop columns containing NaNs

    Returns:
        `Dataframe` containing `action`, `tune`, `action_err` and `tune_err` columns.
    """
    col_natq, col_natq_std = _get_ampdet_columns(corrected)

    columns = {"action": get_action_col(action_plane),
               "action_err": get_action_err_col(action_plane),
               "tune": col_natq(tune_plane),
               "tune_err": col_natq_std(tune_plane),
               }

    not_found = [cv for cv in columns.values() if cv not in kickac_df.columns]
    if any(not_found):
        raise KeyError(f"The following columns were not found in kick-file: '{str(not_found)}'")

    data = kickac_df.loc[:, list(columns.values())]
    data.columns = columns.keys()

    if data.isna().any().any():
        # could be on purpose,
        # for manually filtering kick points in one plane, but not in the other
        LOG.warning(
            f"Amplitude Detuning data for Q{tune_plane} and J{action_plane} contains NaNs"
        )
        if dropna:
            LOG.debug("NaN's are being dropped.")
            data = data.dropna(axis=0)

    return AmpDetData(
        action_plane=action_plane,
        tune_plane=tune_plane,
        **{column: data[column] for column in data.columns}
    )


# Timed DataFrames -------------------------------------------------------------


def get_timestamp_index(index: pd.Index) -> pd.Index:
    return pd.Index([i.timestamp() for i in index])


def read_timed_dataframe(path: Union[str, Path]) -> tfs.TfsDataFrame:
    df = tfs.read(path, index=get_time_col())
    df.index = pd.Index([CERNDatetime.from_cern_utc_string(i) for i in df.index], dtype=object)
    return df


def write_timed_dataframe(path: Union[str, Path], df: tfs.TfsDataFrame):
    df = df.copy()
    df.index = pd.Index([i.strftime(get_cern_time_format()) for i in df.index], dtype=str)
    tfs.write(path, df, save_index=get_time_col(), headerswidth=max(len(k) for k in df.headers.keys()))


def read_two_kick_files_from_folder(folder: Path) -> tfs.TfsDataFrame:
    return merge_two_plane_kick_dfs(
        *[read_timed_dataframe(folder / f'{KICK_NAME}{p.lower()}.tfs') for p in PLANES]
    )


def merge_two_plane_kick_dfs(df_x: tfs.TfsDataFrame, df_y: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    df_xy = TfsDataFrame(pd.merge(df_x, df_y, how='outer', left_index=True, right_index=True, suffixes=PLANES))
    if len(df_xy.index) != len(df_x.index) or len(df_xy.index) != len(df_y.index):
        LOG.warning("The kick-files for each plane seem to have different indices.")
    df_xy.headers = df_x.headers
    df_xy.headers.update(df_y.headers)
    return df_xy


# BBS Converter ----------------------------------------------------------------

def convert_bbs_kickfile(file: Path):
    """ Wrapper for :meth:`omc3.tune_analysis.kick_file_modifiers.convert_old_kickdataframe`
    with read/write capabilities."""
    df = tfs.read(file)
    df = convert_bbs_kickdataframe(df)
    tfs.write(file.with_name(get_kick_out_name()), df)


def convert_bbs_kickdataframe(df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """Converts a kick-dataframe created with the python2 amplitude detuning
    analysis into a dataframe compatible with omc3's detuning analysis.
    Beware that the old files may not contain the error coefficients for
    the corrected data in the header (the uncorrected errors are used here),
    as well as no corrected natural tune error columns.
    And that the offset (coeffient 0) has no error, which is here assumed
    to be zero (it is not used anyway, but required when reading the data).
    The action unit is converted from um to m.
    !!! Be also aware, that due to a bug the Action and Tune planes might be switched !!!
    This is not taken into account here.
    """
    # Header ---
    df.headers = {_rename_old_header(key): value for key, value in df.headers.items()}
    for key in list(df.headers.keys()):
        # convert coeffients (only first order was implemented in python2)
        if COEFFICIENT.format(order=1) in key:
            df.headers[key] = df.headers[key] * 1e6  # inverse um to inverse m

        # # use uncorrected error columns for corrected data
        # if ERR in key:
        #     new_key = key.replace(ERR, f"{ERR}{CORRECTED}")
        #     df.headers[new_key] = df.headers[key]

    # add err headers for coefficient 0 (offset)
    for tune in PLANES:
        for action in PLANES:
            name = get_odr_header_err_coeff(q_plane=tune, j_plane=action, order=0)
            df.headers[name] = df.headers.get(name, 0)

            name = get_odr_header_err_coeff_corrected(q_plane=tune, j_plane=action, order=0)
            df.headers[name] = df.headers.get(name, 0)

    # Columns ---
    df[TIME] = [CERNDatetime.from_timestamp(ts).cern_utc_string() for ts in df[TIME]]
    df.columns = [_rename_old_column(c) for c in df.columns]
    for column in df.columns:
        if ACTION in column:
            df[column] = df[column] * 1e-6  # um to m

    for plane in PLANES:
        col = f"{ERR}{NAT_TUNE}{plane}{CORRECTED}"
        if col not in df.columns:
            df[col] = df[f"{ERR}{NAT_TUNE}{plane}"]
    return df


def _rename_old_header(key: str):
    for tune in PLANES:
        for action in PLANES:
            old = f"ODR_J{action}Q{tune}"
            new = get_odr_header_default(q_plane=tune, j_plane=action)
            key = key.replace(old, new)
    key = key.replace("OFFSET", COEFFICIENT.format(order=0)).replace("SLOPE", COEFFICIENT.format(order=1))

    # CORR was the last prefix in the old file, so replace this first
    if "_CORR" in key:
        parts = key.split("_")
        key = "_".join(parts[:2] + [f"{CORRECTED}{parts[2]}"] + parts[3:-1])

    # and then remove also STD and add ERR
    if "_STD" in key:
        parts = key.split("_")
        key = "_".join(parts[:-2] + [f"{ERR}{parts[-2]}"])
    return key


def _rename_old_column(column: str):
    if "STD" in column or "RMS" in column:
        return f"{ERR}{column.replace('STD', '').replace('RMS', '')}"
    return column
