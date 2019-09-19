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

import datetime
import os

import matplotlib.pyplot as plt

from tune_analysis import bbq_tools, timber_extract, detuning_tools, kickac_modifiers
import tune_analysis.constants as ta_const
from utils import logging_tools
import tfs
from generic_parser.entrypoint import entrypoint, EntryPointParameters

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

TIMEZONE = ta_const.get_experiment_timezone()

DTIME = 60  # extra seconds to add to kickac times when extracting from timber

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
        flags="--bbqout",
        help="Output location to save bbq data as tfs-file",
        name="bbq_out",
        type=str,
    )
    params.add_parameter(
        flags="--kickac",
        help="Location of the kickac file",
        name="kickac_path",
        type=str,
        required=True,
    )
    params.add_parameter(
        flags="--kickacout",
        help="If given, writes out the modified kickac file",
        name="kickac_out",
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

    Keyword Args:
        Required
        beam (int): Which beam to use.
                    **Flags**: --beam
        kickac_path (str): Location of the kickac file
                           **Flags**: --kickac
        plane (str): Plane of the kicks. 'X' or 'Y'.
                           **Flags**: --plane
                           **Choices**: XY
        Optional
        ampdet_plot_out (str): Save the amplitude detuning plot here.
                          **Flags**: --ampdetplot
        ampdet_plot_show: Show the amplitude detuning plot.
                          **Flags**: --ampdetplotshow
                          **Action**: ``store_true``
        ampdet_plot_xmax (float): Maximum action (x-axis) in amplitude detuning plot.
                          **Flags**: --ampdetplotxmax
        ampdet_plot_xmin (float): Minimum action (x-axis) in amplitude detuning plot.
                                  **Flags**: --ampdetplotxmin
        ampdet_plot_ymax (float): Maximum tune (y-axis) in amplitude detuning plot.
                                  **Flags**: --ampdetplotymax
        ampdet_plot_ymin (float): Minimum tune (y-axis) in amplitude detuning plot.
                                  **Flags**: --ampdetplotymin
        bbq_out (str): Output location to save bbq data as tfs-file
                       **Flags**: --bbqout
        bbq_plot_full: Plot the full bqq data with interval as lines.
                       **Flags**: --bbqplotfull
                       **Action**: ``store_true``
        bbq_plot_out (str): Save the bbq plot here.
                            **Flags**: --bbqplot
        bbq_plot_show: Show the bbq plot.
                       **Flags**: --bbqplotshow
                       **Action**: ``store_true``
        bbq_plot_two: Two plots for the bbq plot.
                      **Flags**: --bbqplottwo
                      **Action**: ``store_true``
        debug: Activates Debug mode
               **Flags**: --debug
               **Action**: ``store_true``
        fine_cut (float): Cut (i.e. tolerance) of the tune for the fine cleaning.
                          **Flags**: --finecut
        fine_window (int): Length of the moving average window. (# data points)
                           **Flags**: --finewindow
        kickac_out (str): If given, writes out the modified kickac file
                          **Flags**: --kickacout
        label (str): Label to identify this run.
                     **Flags**: --label
        logfile (str): Logfile if debug mode is active.
                       **Flags**: --logfile
        timber_in: Fill number of desired data or path to presaved tfs-file
                   **Flags**: --timberin
        timber_out (str): Output location to save fill as tfs-file
                          **Flags**: --timberout
        tune_cut (float): Cuts for the tune. For BBQ cleaning.
                          **Flags**: --tunecut
        tune_x (float): Horizontal Tune. For BBQ cleaning.
                        **Flags**: --tunex
        tune_x_max (float): Horizontal Tune minimum. For BBQ cleaning.
                            **Flags**: --tunexmax
        tune_x_min (float): Horizontal Tune minimum. For BBQ cleaning.
                            **Flags**: --tunexmin
        tune_y (float): Vertical Tune. For BBQ cleaning.
                        **Flags**: --tuney
        tune_y_max (float): Vertical Tune minimum. For BBQ cleaning.
                            **Flags**: --tuneymax
        tune_y_min (float): Vertical  Tune minimum. For BBQ cleaning.
                            **Flags**: --tuneymin
        window_length (int): Length of the moving average window. (# data points)
                             **Flags**: --window
                             **Default**: ``20``
     """
    LOG.info("Starting Amplitude Detuning Analysis")
    with logging_tools.DebugMode(active=opt.debug, log_file=opt.logfile):
        opt = _check_analyse_opt(opt)
        figs = {}

        # get data
        kickac_df = tfs.read_tfs(opt.kickac_path, index=COL_TIME())
        bbq_df = _get_timber_data(opt.beam, opt.timber_in, opt.timber_out, kickac_df)
        x_interval = get_approx_bbq_interval(bbq_df, kickac_df.index, opt.window_length)

        # add moving average to kickac
        kickac_df, bbq_df = kickac_modifiers.add_moving_average(kickac_df, bbq_df,
                                                                **opt.get_subdict([
                                                                    "window_length",
                                                                    "tune_x_min", "tune_x_max",
                                                                    "tune_y_min", "tune_y_max",
                                                                    "fine_cut", "fine_window"]
                                                                )
                                                                )

        # add corrected values to kickac
        kickac_df = kickac_modifiers.add_corrected_natural_tunes(kickac_df)
        kickac_df = kickac_modifiers.add_total_natq_std(kickac_df)

        # amplitude detuning odr and plotting
        for tune_plane in PLANES:
            for corr in [False, True]:
                # get the proper data
                data = kickac_modifiers.get_ampdet_data(kickac_df, opt.plane, tune_plane,
                                                        corrected=corr)

                # make the odr
                odr_fit = detuning_tools.do_linear_odr(**data)
                kickac_df = kickac_modifiers.add_odr(kickac_df, odr_fit, opt.plane, tune_plane,
                                                     corrected=corr)

    # output kickac and bbq data
    tfs.write_tfs(opt.kickac_out, kickac_df, save_index=COL_TIME())
    tfs.write_tfs(opt.bbq_out, bbq_df.loc[x_interval[0]:x_interval[1]],
                  save_index=COL_TIME())

    return figs


def get_approx_bbq_interval(bbq_df, time_array, window_length):
    """ Get data in approximate time interval,
    for averaging based on window length and kickac interval """
    bbq_tmp = bbq_df.dropna()

    i_start = max(bbq_tmp.index.get_loc(time_array[0], method='nearest') - int(window_length/2.),
                  0
                  )
    i_end = min(bbq_tmp.index.get_loc(time_array[-1], method='nearest') + int(window_length/2.),
                len(bbq_tmp.index)-1
                )

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


def _get_timber_data(beam, input, output, kickac_df):
    """ Return Timber data from input """

    try:
        fill_number = int(input)
    except ValueError:
        # input is a string
        LOG.debug(f"Getting timber data from file '{input:s}'")
        data = tfs.read_tfs(input, index=COL_TIME())
        data.drop([COL_MAV(p) for p in PLANES if COL_MAV(p) in data.columns],
                  axis='columns')
    except TypeError:
        # input is None
        LOG.debug("Getting timber data from kickac-times.")
        timber_keys, bbq_cols = _get_timber_keys_and_bbq_columns(beam)
        t_start = min(kickac_df.index.values)
        t_end = max(kickac_df.index.values)
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

    if output:
        tfs.write_tfs(output, data, save_index=COL_TIME())

    return data


def _get_timber_keys_and_bbq_columns(beam):
    keys = [TIMBER_KEY(plane, beam) for plane in PLANES]
    cols = [COL_BBQ(plane) for plane in PLANES]
    return keys, cols


# Script Mode ##################################################################


if __name__ == '__main__':
    analyse_with_bbq_corrections()
