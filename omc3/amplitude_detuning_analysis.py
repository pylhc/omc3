"""
Amplitude Detuning Analysis
---------------------------

Entrypoint for amplitude detuning analysis.

This module provides functionality to run amplitude detuning analysis with additionally getting
BBQ data from timber, averaging and filtering this data and subtracting it from the measurement
data.

Furthermore, the orthogonal distance regression is utilized to get a linear or quadratic fit from
the measurements.


**Arguments:**

*--Required--*

- **beam** *(int)*: Which beam to use.

- **kick** *(str)*: Location of the kick files (parent folder).

- **plane** *(str)*: Plane of the kicks. 'X' or 'Y'.

  Choices: ``('X', 'Y')``

*--Optional--*

- **bbq_filtering_method** *(str)*:Filtering method for the bbq to use. 'cut' cuts around a given tune,
  'minmax' lets you specify the limits and 'outliers' uses the outlier filtering from utils.

  Choices: ``['cut', 'minmax', 'outliers']``
  Default: ``outliers``
- **bbq_in**: Fill number of desired data to extract from timber
  or path to presaved bbq-tfs-file. Use the string 'kick' to use the timestamps
  in the kickfile for timber extraction. Not giving this parameter skips bbq compensation.

- **debug**: Activates Debug mode

  Action: ``store_true``
- **detuning_order** *(int)*: Order of the detuning as int. Basically just the order of the applied fit.

  Default: ``1``
- **fine_cut** *(float)*: Cut, i.e. tolerance, of the tune (fine cleaning for 'minmax' or 'cut').

- **fine_window** *(int)*: Length of the moving average window, i.e # data points (fine cleaning for 'minmax' or 'cut').

- **label** *(str)*: Label to identify this run.

- **logfile** *(str)*: Logfile if debug mode is active.

- **outlier_limit** *(float)*: Limit, i.e. cut, on outliers (Method 'outliers')

  Default: ``0.0002``
- **output** *(str)*: Output directory for the modified kickfile and bbq data.

- **tune_cut** *(float)*: Cuts for the tune. For BBQ cleaning (Method 'cut').

- **tunes** *(float)*: Tunes for BBQ cleaning (Method 'cut').

- **tunes_minmax** *(float)*: Tunes minima and maxima in the order x_min, x_max, y_min, y_max. For BBQ cleaning (Method 'minmax').

- **window_length** *(int)*: Length of the moving average window. (# data points)

  Default: ``20``
"""
import os
from collections import OrderedDict
from pathlib import Path

from generic_parser.entrypoint_parser import entrypoint, EntryPointParameters, save_options_to_config

from omc3.definitions import formats
from omc3.definitions.constants import PLANES
from omc3.tune_analysis import fitting_tools, kick_file_modifiers
from omc3.tune_analysis.constants import (get_kick_out_name, get_bbq_out_name,
                                          get_mav_col, get_timber_bbq_key,
                                          get_bbq_col)
from omc3.tune_analysis.kick_file_modifiers import (read_timed_dataframe,
                                                    write_timed_dataframe,
                                                    read_two_kick_files_from_folder
                                                    )
from omc3.utils.logging_tools import get_logger, list2str, DebugMode

try:
    from omc3.tune_analysis import timber_extract
except ImportError:
    timber_extract = None

# Globals ----------------------------------------------------------------------

DTIME = 60  # extra seconds to add to kick times when extracting from timber

LOG = get_logger(__name__)


# Get Parameters ---------------------------------------------------------------


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
        bbq_in=dict(
            help=("Fill number of desired data to extract from timber "
                  "or path to presaved bbq-tfs-file. Use the string 'kick' to "
                  "use the timestamps in the kickfile for timber extraction. "
                  "Not giving this parameter skips bbq compensation."),
        ),
        detuning_order=dict(
            help="Order of the detuning as int. Basically just the order of the applied fit.",
            type=int,
            default=1,
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
        bbq_filtering_method=dict(
            help="Filtering method for the bbq to use. 'cut' cuts around a given tune, "
                 "'minmax' lets you specify the limits and 'outliers' uses the outlier filtering "
                 "from utils.",
            type=str,
            choices=["cut", "minmax", "outliers"],
            default="outliers",
        ),
        # Filtering method outliers
        outlier_limit=dict(
            help="Limit, i.e. cut, on outliers (Method 'outliers')",
            type=float,
            default=2e-4,
        ),
        # Filtering method tune-cut
        tunes=dict(
            help="Tunes for BBQ cleaning (Method 'cut').",
            type=float,
            nargs=2,
        ),
        tune_cut=dict(
            help="Cuts for the tune. For BBQ cleaning (Method 'cut').",
            type=float,
        ),
        # Filtering method tune-minmax
        tunes_minmax=dict(
            help=("Tunes minima and maxima in the order x_min, x_max, y_min, y_max. "
                  "For BBQ cleaning (Method 'minmax')."),
            type=float,
            nargs=4,
        ),
        # Fine Cleaning
        fine_window=dict(
            help="Length of the moving average window, i.e # data points (fine cleaning for 'minmax' or 'cut').",
            type=int,
        ),
        fine_cut=dict(
            help="Cut, i.e. tolerance, of the tune (fine cleaning for 'minmax' or 'cut').",
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


# Main -------------------------------------------------------------------------


@entrypoint(_get_params(), strict=True)
def analyse_with_bbq_corrections(opt):
    """Create amplitude detuning analysis with BBQ correction from timber data."""
    LOG.info("Starting Amplitude Detuning Analysis")
    _save_options(opt)

    with DebugMode(active=opt.debug, log_file=opt.logfile):
        opt, filter_opt = _check_analyse_opt(opt)

        # get data
        kick_df = read_two_kick_files_from_folder(opt.kick)
        bbq_df = None

        if opt.bbq_in is not None:
            bbq_df = _get_bbq_data(opt.beam, opt.bbq_in, kick_df)
            x_interval = get_approx_bbq_interval(bbq_df, kick_df.index, opt.window_length)

            # add moving average to kick
            kick_df, bbq_df = kick_file_modifiers.add_moving_average(kick_df, bbq_df, filter_opt)

            # add corrected values to kick
            kick_df = kick_file_modifiers.add_corrected_natural_tunes(kick_df)
            kick_df = kick_file_modifiers.add_total_natq_std(kick_df)

        kick_plane = opt.plane

        # amplitude detuning odr
        for tune_plane in PLANES:
            for corrected in [False, True]:
                if corrected and opt.bbq_in is None:
                    continue

                # get the proper data
                data_df = kick_file_modifiers.get_ampdet_data(kick_df, kick_plane, tune_plane, corrected=corrected)

                # make the odr
                odr_fit = fitting_tools.do_odr(x=data_df['action'], y=data_df['tune'],
                                               xerr=data_df['action_err'], yerr=data_df['tune_err'],
                                               order=opt.detuning_order)
                kick_df = kick_file_modifiers.add_odr(kick_df, odr_fit, kick_plane, tune_plane, corrected=corrected)

    # output kick and bbq data
    if opt.output:
        opt.output.mkdir(parents=True, exist_ok=True)
        write_timed_dataframe(opt.output / get_kick_out_name(),
                              kick_df)
        if bbq_df is not None:
            write_timed_dataframe(opt.output / get_bbq_out_name(),
                                  bbq_df.loc[x_interval[0]:x_interval[1]])

    return kick_df, bbq_df


def get_approx_bbq_interval(bbq_df, time_array, window_length):
    """
    Get data in approximate time interval, for averaging based on window length and kick interval.
    """
    bbq_tmp = bbq_df.dropna()

    # convert to float to use math-comparisons
    ts_index = kick_file_modifiers.get_timestamp_index(bbq_df.index)
    ts_start, ts_end = time_array[0].timestamp(), time_array[-1].timestamp()

    i_start = max(ts_index.get_loc(ts_start, method='nearest') - int(window_length/2.), 0)
    i_end = min(ts_index.get_loc(ts_end, method='nearest') + int(window_length/2.), len(ts_index)-1)

    return bbq_tmp.index[i_start], bbq_tmp.index[i_end]


# Private Functions ------------------------------------------------------------


def _check_analyse_opt(opt):
    """Perform manual checks on opt-sturcture."""
    LOG.debug("Checking Options.")

    # for label
    if opt.label is None:
        opt.label = f"Amplitude Detuning for Beam {opt.beam:d}"

    filter_opt = None
    if opt.bbq_in is not None:
        # check if cleaning is properly specified
        all_filter_opt = dict(
            cut=opt.get_subdict(['bbq_filtering_method', 'window_length',
                                 'tunes', 'tune_cut',
                                 'fine_window', 'fine_cut']
                                ),
            minmax=opt.get_subdict(['bbq_filtering_method', 'window_length',
                                    'tunes_minmax', 'fine_window', 'fine_cut']
                                   ),
            outliers=opt.get_subdict(['bbq_filtering_method',
                                      'window_length', 'outlier_limit']
                                     ),
        )

        for method, params in all_filter_opt.items():
            if opt.bbq_filtering_method == method:
                missing_params = [k for k, v in params.items() if v is None]
                if any(missing_params):
                    raise KeyError("Missing parameters for chosen cleaning method "
                                   f"{method}: '{list2str(missing_params)}'")
                filter_opt = params

        if filter_opt.bbq_filtering_method == 'cut':
            filter_opt[f"tunes_minmax"] = [minmax for t in opt.tunes
                                           for minmax in (t - opt.tune_cut, t + opt.tune_cut)]
            filter_opt.pop('tune_cut')
            filter_opt.pop('tunes')

        if filter_opt.bbq_filtering_method != 'outliers':
            # check fine cleaning
            if bool(filter_opt.fine_cut) != bool(filter_opt.fine_window):
                raise KeyError("To activate fine cleaning, "
                               "both fine cut and fine window need to be specified")

    if opt.output is not None:
        opt.output = Path(opt.output)

    return opt, filter_opt


def _get_bbq_data(beam, input_, kick_df):
    """Return BBQ data from input, either file or timber fill."""
    try:
        fill_number = int(input_)
    except ValueError:
        # input is a string
        if input_ == "kick":
            if not timber_extract:
                LOG.error(
                    "The pytimber package does not seem to be installed but is needed for this function. "
                    "Install it with the 'tech' dependency of omc3, which requires to be on the CERN "
                    "technical network.")
                raise ImportError("The pytimber package is needed for this operation but can't be found.")
            LOG.debug("Getting timber data from kick-times.")
            timber_keys, bbq_cols = _get_timber_keys_and_bbq_columns(beam)
            t_start = min(kick_df.index.to_numpy())
            t_end = max(kick_df.index.to_numpy())
            data = timber_extract.extract_between_times(t_start-DTIME, t_end+DTIME,
                                                        keys=timber_keys,
                                                        names=dict(zip(timber_keys, bbq_cols)))
        else:
            LOG.debug(f"Getting bbq data from file '{input_:s}'")
            data = read_timed_dataframe(input_)
            data.drop([get_mav_col(p) for p in PLANES
                       if get_mav_col(p) in data.columns], axis='columns')

    else:
        # input is a number
        LOG.debug(f"Getting timber data from fill '{input_:d}'")
        if not timber_extract:
            LOG.error("The pytimber package does not seem to be installed but is needed for this function. "
                      "Install it with the 'tech' dependency of omc3, which requires to be on the CERN "
                      "technical network.")
            raise ImportError("The pytimber package is needed for this operation but can't be found.")
        timber_keys, bbq_cols = _get_timber_keys_and_bbq_columns(beam)
        data = timber_extract.lhc_fill_to_tfs(fill_number,
                                              keys=timber_keys,
                                              names=dict(zip(timber_keys, bbq_cols)))
    return data


def _get_timber_keys_and_bbq_columns(beam):
    keys = [get_timber_bbq_key(plane, beam) for plane in PLANES]
    cols = [get_bbq_col(plane) for plane in PLANES]
    return keys, cols


def _save_options(opt):
    if opt.output:
        os.makedirs(opt.output, exist_ok=True)
        save_options_to_config(os.path.join(opt.output, formats.get_config_filename(__file__)),
                               OrderedDict(sorted(opt.items()))
                               )


# Script Mode ##################################################################


if __name__ == '__main__':
    analyse_with_bbq_corrections()
