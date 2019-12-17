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
    return EntryPointParameters(
        beam=dict(
            help="Which beam to use.",
            required=True,
            type=int,
        ),
        kick=dict(
            help="Location of the kick files (parent folder).",
            type=str,
            required=True,
        ),
        plane=dict(
            help="Plane of the kicks. 'X' or 'Y'.",
            required=True,
            choices=PLANES,
            type=str,
        ),
        label=dict(
            help="Label to identify this run.",
            type=str,
        ),
        timber_in=dict(
            help="Fill number of desired data or path to presaved tfs-file",
        ),
        output=dict(
            help="Output directory for the modified kickfile and bbq data.",
            type=str,
        ),
        window_length=dict(
            help="Length of the moving average window. (# data points)",
            type=int,
            default=20,
        ),
        # Cleaning Method A
        tune_x=dict(
            help="Horizontal Tune. For BBQ cleaning (Method A).",
            type=float,
        ),
        tune_y=dict(
            help="Vertical Tune. For BBQ cleaning (Method A).",
            type=float,
        ),
        tune_cut=dict(
            help="Cuts for the tune. For BBQ cleaning (Method A).",
            type=float,
        ),
        # Cleaning Method B:
        tune_x_min=dict(
            help="Horizontal Tune minimum. For BBQ cleaning (Method B).",
            type=float,
        ),
        tune_x_max=dict(
            help="Horizontal Tune maximum. For BBQ cleaning (Method B).",
            type=float,
        ),
        tune_y_min=dict(
            help="Vertical  Tune minimum. For BBQ cleaning (Method B).",
            type=float,
        ),
        tune_y_max=dict(
            help="Vertical Tune maximum. For BBQ cleaning (Method B).",
            type=float,
        ),
        # Fine Cleaning
        fine_window=dict(
            help="Length of the moving average window, i.e # data points (fine cleaning).",
            type=int,
        ),
        fine_cut=dict(
            help="Cut, i.e. tolerance, of the tune (fine cleaning).",
            type=float,
        ),
        # Debug
        debug=dict(
            help="Activates Debug mode",
            action="store_true",
        ),
        logfile=dict(
            help="Logfile if debug mode is active.",
            type=str,
        ),
    )


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
        kick_df, bbq_df = kick_file_modifiers.add_moving_average(
            kick_df, bbq_df, **opt.get_subdict(["tune_x_min", "tune_x_max", "tune_y_min", "tune_y_max",
                                                "window_length", "fine_cut", "fine_window"]))

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
    method_a = [opt.tune_x, opt.tune_y, opt.tune_cut]
    method_b = [opt.tune_x_min, opt.tune_x_max, opt.tune_y_min, opt.tune_y_max]
    if any(method_a) and any(method_b):
        raise KeyError("Choose either the method of cleaning BBQ with tunes and cut or with min and max values")

    if any(method_a):
        if opt.tune_cut is None:
            raise KeyError("Tune cut is needed for cleaning tune.")

        # set min and max for specified tune cut
        for plane in PLANES:
            qstr = f"tune_{plane.lower()}"
            tune = opt.pop(qstr)
            if tune:
                opt[f"{qstr}_min"] = tune - opt.tune_cut
                opt[f"{qstr}_max"] = tune + opt.tune_cut
        opt.pop('tune_cut')

    # check fine cleaning
    if bool(opt.fine_cut) != bool(opt.fine_window):
        raise KeyError("To activate fine cleaning, both fine cut and fine window need to be specified")
    return opt


def _get_timber_data(beam, input, kick_df):
    """ Return Timber data from input """

    try:
        fill_number = int(input)
    except ValueError:
        # input is a string
        LOG.debug(f"Getting timber data from file '{input:s}'")
        data = read_timed_dataframe(input)
        data.drop([COL_MAV(p) for p in PLANES if COL_MAV(p) in data.columns], axis='columns')
    except TypeError:
        # input is None
        LOG.debug("Getting timber data from kick-times.")
        timber_keys, bbq_cols = _get_timber_keys_and_bbq_columns(beam)
        t_start = min(kick_df.index.values)
        t_end = max(kick_df.index.values)
        data = timber_extract.extract_between_times(t_start-DTIME, t_end+DTIME, keys=timber_keys,
                                                    names=dict(zip(timber_keys, bbq_cols)))
    else:
        # input is a number
        LOG.debug(f"Getting timber data from fill '{input:d}'")
        timber_keys, bbq_cols = _get_timber_keys_and_bbq_columns(beam)
        data = timber_extract.lhc_fill_to_tfs(fill_number, keys=timber_keys, names=dict(zip(timber_keys, bbq_cols)))
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
