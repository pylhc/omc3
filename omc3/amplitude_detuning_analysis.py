"""
Entrypoint Amplitude Detuning Analysis
------------------------------------------------

Entrypoint for amplitude detuning analysis.

This module provides functionality to run amplitude detuning analysis with
additionally getting BBQ data from timber, averaging and filtering this data and
subtracting it from the measurement data.

Furthermore, the orthogonal distance regression is utilized to get a
linear fit from the measurements.


:author: Joschua Dilly
"""
import os
import pandas as pd

from tune_analysis import timber_extract, detuning_tools, kick_file_modifiers
import tune_analysis.constants as ta_const

from tune_analysis.kick_file_modifiers import (read_timed_dataframe,
                                               write_timed_dataframe,
                                               read_two_kick_files_from_folder
                                               )
from utils import logging_tools
from utils.time_tools import CERNDatetime
from generic_parser.entrypoint_parser import entrypoint, EntryPointParameters, save_options_to_config

# Globals ####################################################################

# Column Names
COL_TIME = ta_const.get_time_col
COL_BBQ = ta_const.get_bbq_col
COL_MAV = ta_const.get_mav_col
COL_MAV_STD = ta_const.get_mav_std_col
COL_IN_MAV = ta_const.get_used_in_mav_col
COL_NATQ = ta_const.get_natq_col
COL_CORRECTED = ta_const.get_natq_corr_col

PLANES = ta_const.get_planes()
TIMBER_KEY = ta_const.get_timber_bbq_key

DTIME = 60  # extra seconds to add to kick times when extracting from timber

LOG = logging_tools.get_logger(__name__)


# Get Parameters #############################################################


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(
        flags="--label",
        help="Label to identify this run.",
        name="label",
        type=str,
    )
    params.add_parameter(
        flags="--beam",
        help="Which beam to use.",
        name="beam",
        required=True,
        type=int,
    )
    params.add_parameter(
        flags="--plane",
        help="Plane of the kicks. 'X' or 'Y'.",
        name="plane",
        required=True,
        choices=PLANES,
        type=str,
    )
    params.add_parameter(
        flags="--timberin",
        help="Fill number of desired data or path to presaved tfs-file",
        name="timber_in",
    )
    params.add_parameter(
        flags="--timberout",
        help="Output location to save fill as tfs-file",
        name="timber_out",
        type=str,
    )
    params.add_parameter(
        flags="--kick",
        help="Location of the kick files (parent folder).",
        name="kick",
        type=str,
        required=True,
    )
    params.add_parameter(
        flags="--out",
        help="Output directory for the modified kickfile and bbq data.",
        name="output",
        type=str,
    )
    params.add_parameter(
        flags="--window",
        help="Length of the moving average window. (# data points)",
        name="window_length",
        type=int,
        default=20,
    )

    # cleaning method one:
    params.add_parameter(
        flags="--tunex",
        help="Horizontal Tune. For BBQ cleaning.",
        name="tune_x",
        type=float,
    )
    params.add_parameter(
        flags="--tuney",
        help="Vertical Tune. For BBQ cleaning.",
        name="tune_y",
        type=float,
    )
    params.add_parameter(
        flags="--tunecut",
        help="Cuts for the tune. For BBQ cleaning.",
        name="tune_cut",
        type=float,
    )
    # cleaning method two:
    params.add_parameter(
        flags="--tunexmin",
        help="Horizontal Tune minimum. For BBQ cleaning.",
        name="tune_x_min",
        type=float,
    )
    params.add_parameter(
        flags="--tunexmax",
        help="Horizontal Tune minimum. For BBQ cleaning.",
        name="tune_x_max",
        type=float,
    )
    params.add_parameter(
        flags="--tuneymin",
        help="Vertical  Tune minimum. For BBQ cleaning.",
        name="tune_y_min",
        type=float,
    )
    params.add_parameter(
        flags="--tuneymax",
        help="Vertical Tune minimum. For BBQ cleaning.",
        name="tune_y_max",
        type=float,
    )

    # fine cleaning
    params.add_parameter(
        flags="--finewindow",
        help="Length of the moving average window. (# data points)",
        name="fine_window",
        type=int,
    )
    params.add_parameter(
        flags="--finecut",
        help="Cut (i.e. tolerance) of the tune for the fine cleaning.",
        name="fine_cut",
        type=float,
    )

    # Debug
    params.add_parameter(
        flags="--debug",
        help="Activates Debug mode",
        name="debug",
        action="store_true",
    )
    params.add_parameter(
        flags="--logfile",
        help="Logfile if debug mode is active.",
        name="logfile",
        type=str,
    )

    return params


# Main #########################################################################


@entrypoint(_get_params(), strict=True)
def analyse_with_bbq_corrections(opt):
    """ Create amplitude detuning analysis with BBQ correction from timber data.

     """
    LOG.info("Starting Amplitude Detuning Analysis")
    _save_options(opt)

    with logging_tools.DebugMode(active=opt.debug, log_file=opt.logfile):
        opt = _check_analyse_opt(opt)

        # get data
        kick_df = read_two_kick_files_from_folder(opt.kick)
        bbq_df = _get_timber_data(opt.beam, opt.timber_in, kick_df)
        x_interval = get_approx_bbq_interval(bbq_df, kick_df.index, opt.window_length)

        # add moving average to kick
        kick_df, bbq_df = kick_file_modifiers.add_moving_average(kick_df, bbq_df,
                                                              **opt.get_subdict([
                                                                  "window_length",
                                                                  "tune_x_min", "tune_x_max",
                                                                  "tune_y_min", "tune_y_max",
                                                                  "fine_cut", "fine_window"]
                                                              )
                                                              )

        # add corrected values to kick
        kick_df = kick_file_modifiers.add_corrected_natural_tunes(kick_df)
        kick_df = kick_file_modifiers.add_total_natq_std(kick_df)

        # amplitude detuning odr
        for tune_plane in PLANES:
            for corr in [False, True]:
                # get the proper data
                data = kick_file_modifiers.get_ampdet_data(kick_df, opt.plane, tune_plane, corrected=corr)

                # make the odr
                odr_fit = detuning_tools.do_linear_odr(**data)
                kick_df = kick_file_modifiers.add_odr(kick_df, odr_fit, opt.plane, tune_plane, corrected=corr)

    # output kick and bbq data
    if opt.output:
        os.makedirs(os.path.dirname(opt.output), exist_ok=True)
        write_timed_dataframe(os.path.join(opt.output, ta_const.get_kick_out_name()),
                              kick_df)
        write_timed_dataframe(os.path.join(opt.output, ta_const.get_bbq_out_name()),
                              bbq_df.loc[x_interval[0]:x_interval[1]])

    return kick_df, bbq_df


def get_approx_bbq_interval(bbq_df, time_array, window_length):
    """ Get data in approximate time interval,
    for averaging based on window length and kick interval """
    bbq_tmp = bbq_df.dropna()

    # convert to float to use math-comparisons
    ts_index = kick_file_modifiers.get_timestamp_index(bbq_df.index)
    ts_start, ts_end = time_array[0].timestamp(), time_array[-1].timestamp()

    i_start = max(ts_index.get_loc(ts_start, method='nearest') - int(window_length/2.), 0)
    i_end = min(ts_index.get_loc(ts_end, method='nearest') + int(window_length/2.), len(ts_index)-1)

    return bbq_tmp.index[i_start], bbq_tmp.index[i_end]


# Private Functions ############################################################


def _check_analyse_opt(opt):
    """ Perform manual checks on opt-sturcture """
    LOG.debug("Checking Options.")

    # for label
    if opt.label is None:
        opt.label = f"Amplitude Detuning for Beam {opt.beam:d}"

    # check if cleaning is properly specified
    if (any([opt.tune_x, opt.tune_y, opt.tune_cut])
            and any([opt.tune_x_min, opt.tune_x_max, opt.tune_y_min, opt.tune_y_max])
    ):
        raise KeyError("Choose either the method of cleaning BBQ"
                             "with tunes and cut or with min and max values")

    for plane in PLANES:
        tune = f"tune_{plane.lower()}"
        if opt[tune]:
            if opt.tune_cut is None:
                raise KeyError("Tune cut is needed for cleaning tune.")
            opt[f"{tune}_min"] = opt[tune] - opt.tune_cut
            opt[f"{tune}_max"] = opt[tune] + opt.tune_cut

    if bool(opt.fine_cut) != bool(opt.fine_window):
        raise KeyError("To activate fine cleaning, both fine cut and fine window need"
                             "to be specified")
    return opt


def _get_timber_data(beam, input, kick_df):
    """ Return Timber data from input """

    try:
        fill_number = int(input)
    except ValueError:
        # input is a string
        LOG.debug(f"Getting timber data from file '{input:s}'")
        data = read_timed_dataframe(input)
        data.drop([COL_MAV(p) for p in PLANES if COL_MAV(p) in data.columns],
                  axis='columns')
    except TypeError:
        # input is None
        LOG.debug("Getting timber data from kick-times.")
        timber_keys, bbq_cols = _get_timber_keys_and_bbq_columns(beam)
        t_start = min(kick_df.index.values)
        t_end = max(kick_df.index.values)
        data = timber_extract.extract_between_times(t_start-DTIME, t_end+DTIME,
                                                    keys=timber_keys,
                                                    names=dict(zip(timber_keys, bbq_cols)))
    else:
        # input is a number
        LOG.debug(f"Getting timber data from fill '{input:d}'")
        timber_keys, bbq_cols = _get_timber_keys_and_bbq_columns(beam)
        data = timber_extract.lhc_fill_to_tfs(fill_number,
                                              keys=timber_keys,
                                              names=dict(zip(timber_keys, bbq_cols)))
    return data


def _get_timber_keys_and_bbq_columns(beam):
    keys = [TIMBER_KEY(plane, beam) for plane in PLANES]
    cols = [COL_BBQ(plane) for plane in PLANES]
    return keys, cols


def _save_options(opt):
    if opt.output:
        os.makedirs(opt.output, exist_ok=True)
        save_options_to_config(
            os.path.join(opt.output, f'ampdet_analysis_{CERNDatetime.now().cern_utc_string()}.ini'),
            opt
        )


# Script Mode ##################################################################


if __name__ == '__main__':
    analyse_with_bbq_corrections()
