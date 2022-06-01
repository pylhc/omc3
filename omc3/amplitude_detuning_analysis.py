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

- **beam** *(int)*:

    Which beam to use.


- **kick** *(PathOrStr)*:

    Location of the kick files (parent folder).


- **plane** *(str)*:

    Plane of the kicks. 'X' or 'Y' or 'XY'.

    choices: ``['X', 'Y', 'XY']``


*--Optional--*

- **bbq_filtering_method** *(str)*:

    Filtering method for the bbq to use. 'cut' cuts around a given tune,
    'minmax' lets you specify the limits and 'outliers' uses the outlier
    filtering from utils.

    choices: ``['cut', 'minmax', 'outliers']``

    default: ``outliers``


- **bbq_in** *(UnionPathStrInt)*:

    Fill number of desired data to extract from timber or path to presaved
    bbq-tfs-file. Use the string 'kick' to use the timestamps in the
    kickfile for timber extraction. Not giving this parameter skips bbq
    compensation.


- **detuning_order** *(int)*:

    Order of the detuning as int. Basically just the order of the applied
    fit.

    default: ``1``


- **fine_cut** *(float)*:

    Cut, i.e. tolerance, of the tune (fine cleaning for 'minmax' or
    'cut').


- **fine_window** *(int)*:

    Length of the moving average window, i.e the number of data points
    (fine cleaning for 'minmax' or 'cut').


- **label** *(str)*:

    Label to identify this run.


- **outlier_limit** *(float)*:

    Limit, i.e. cut, on outliers (Method 'outliers')

    default: ``0.0002``


- **output** *(PathOrStr)*:

    Output directory for the modified kickfile and bbq data.


- **tune_cut** *(float)*:

    Cuts for the tune. For BBQ cleaning (Method 'cut').


- **tunes** *(float)*:

    Tunes for BBQ cleaning (Method 'cut').


- **tunes_minmax** *(float)*:

    Tunes minima and maxima in the order x_min, x_max, y_min, y_max. For
    BBQ cleaning (Method 'minmax').


- **window_length** *(int)*:

    Length of the moving average window. (# data points)

    default: ``20``


"""
import os
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Any, Union

import numpy as np
import tfs
from generic_parser import DotDict
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint, save_options_to_config
from numpy.typing import ArrayLike
from tfs.frame import TfsDataFrame

from omc3.definitions import formats
from omc3.definitions.constants import PLANES
from omc3.tune_analysis import fitting_tools, kick_file_modifiers, timber_extract
from omc3.tune_analysis.constants import (
    get_bbq_col,
    get_bbq_out_name,
    get_kick_out_name,
    get_mav_col,
    get_timber_bbq_key, INPUT_KICK, INPUT_PREVIOUS, CORRECTED,
)
from omc3.tune_analysis.kick_file_modifiers import (
    read_timed_dataframe,
    read_two_kick_files_from_folder,
    write_timed_dataframe, AmpDetData,
)
from omc3.utils.iotools import PathOrStr, UnionPathStrInt, save_config
from omc3.utils.logging_tools import get_logger, list2str
from omc3.utils.time_tools import CERNDatetime

# Globals ----------------------------------------------------------------------

DTIME: int = 120  # extra seconds to add to kick times window when extracting from timber

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
            type=PathOrStr,
            required=True,
        ),
        plane=dict(
            help="Plane of the kicks. 'X' or 'Y' or 'XY'.",
            required=True,
            choices=list(PLANES) + ["".join(PLANES)],
            type=str,
        ),
        label=dict(
            help="Label to identify this run.",
            type=str,
        ),
        bbq_in=dict(
            help="Fill number of desired data to extract from timber or path to presaved bbq-tfs-file. "
            f"Use the string '{INPUT_KICK}' to use the timestamps in the kickfile for timber extraction. "
            f"Use the string '{INPUT_PREVIOUS}' to look for the modified ampdet kick-file from a previous run. "
            "Not giving this parameter skips bbq compensation.",
            type=UnionPathStrInt
        ),

        detuning_order=dict(
            help="Order of the detuning as int. Basically just the order of the applied fit.",
            type=int,
            default=1,
        ),
        output=dict(
            help="Output directory for the modified kickfile and bbq data.",
            type=PathOrStr,
        ),
        window_length=dict(
            help="Length of the moving average window. (# data points)",
            type=int,
            default=20,
        ),
        bbq_filtering_method=dict(
            help="Filtering method for the bbq to use. 'cut' cuts around a given tune, 'minmax' lets you "
            "specify the limits and 'outliers' uses the outlier filtering from utils.",
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
            help="Tunes minima and maxima in the order x_min, x_max, y_min, y_max. "
            "For BBQ cleaning (Method 'minmax').",
            type=float,
            nargs=4,
        ),
        # Fine Cleaning
        fine_window=dict(
            help="Length of the moving average window, i.e the number of data points "
            "(fine cleaning for 'minmax' or 'cut').",
            type=int,
        ),
        fine_cut=dict(
            help="Cut, i.e. tolerance, of the tune (fine cleaning for 'minmax' or 'cut').",
            type=float,
        ),
    )


# Main -------------------------------------------------------------------------


@entrypoint(_get_params(), strict=True)
def analyse_with_bbq_corrections(opt: DotDict) -> Tuple[TfsDataFrame, TfsDataFrame]:
    """
    Create amplitude detuning analysis with BBQ correction from timber data.

    Returns:
        The amplitude detuning analysis results as a TfsDataFrame and the BBQ data as a TfsDataFrame.
    """
    LOG.info("Starting Amplitude Detuning Analysis")
    _save_options(opt)

    opt, filter_opt = _check_analyse_opt(opt)
    kick_df, bbq_df = get_kick_and_bbq_df(kick=opt.kick, bbq_in=opt.bbq_in, beam=opt.beam, filter_opt=filter_opt)

    kick_plane = opt.plane

    for corrected in [False] + _should_do_corrected(kick_df, opt.bbq_in):
        if kick_plane in PLANES:
            kick_df = single_action_analysis(kick_df, kick_plane, opt.detuning_order, corrected)
        else:
            kick_df = double_action_analysis(kick_df, opt.detunig_order, corrected)

    # output kick and bbq data
    if opt.output:
        LOG.info(f"Writing kick data to file in directory '{opt.output.absolute()}'")
        opt.output.mkdir(parents=True, exist_ok=True)
        write_timed_dataframe(opt.output / get_kick_out_name(), kick_df)

    return kick_df, bbq_df


def get_kick_and_bbq_df(kick: Union[Path, str], bbq_in: Union[Path, str],
                        beam: int = None, filter_opt = None, output: Path = None
                        ) -> Tuple[tfs.TfsDataFrame, tfs.TfsDataFrame]:
    """Load the input data."""
    bbq_df = None
    if bbq_in is not None and bbq_in == INPUT_PREVIOUS:
        LOG.debug("Getting data from previous ampdet kick file")
        kick_df = read_timed_dataframe(Path(kick) / get_kick_out_name())
        kick_df.headers = {k: v for k, v in kick_df.headers.items() if not k.startswith("ODR_")}
    else:
        LOG.debug("Getting data from kick files")
        kick_df = read_two_kick_files_from_folder(kick)

        if bbq_in is not None:
            bbq_df = _get_bbq_data(beam, bbq_in, kick_df)

            LOG.debug("Adding moving average data to kick data")
            kick_df, bbq_df = kick_file_modifiers.add_moving_average(kick_df, bbq_df, filter_opt)

            LOG.debug("Adding corrected natural tunes and stdev to kick data")
            kick_df = kick_file_modifiers.add_corrected_natural_tunes(kick_df)

            if output:
                LOG.info(f"Writing BBQ data to file in directory '{output.absolute()}'")
                x_interval = get_approx_bbq_interval(bbq_df, kick_df.index, filter_opt.window_length)
                write_timed_dataframe(output / get_bbq_out_name(), bbq_df.loc[x_interval[0]: x_interval[1]])
    return kick_df, bbq_df


def single_action_analysis(kick_df: tfs.TfsDataFrame, kick_plane: str, detuning_order: int = 1, corrected: bool = False
                           ) -> tfs.TfsDataFrame:
    """Performs the fit one action and tune pane at a time."""
    LOG.info(f"Performing amplitude detuning ODR for single-plane kicks in {kick_plane}.")
    for tune_plane in PLANES:
        LOG.debug("Getting ampdet data")
        data = kick_file_modifiers.get_ampdet_data(
            kickac_df=kick_df,
            action_plane=kick_plane,
            tune_plane=tune_plane,
            corrected=corrected
        )

        LOG.debug("Fitting ODR to kick data")
        odr_fit = fitting_tools.do_odr(
            x=data.action,
            y=data.tune,
            xerr=data.action_err,
            yerr=data.tune_err,
            order=detuning_order,
        )

        kick_df = kick_file_modifiers.add_odr(
            kickac_df=kick_df,
            odr_fit=odr_fit,
            action_plane=kick_plane,
            tune_plane=tune_plane,
            corrected=corrected
        )
    return kick_df


def double_action_analysis(kick_df: tfs.TfsDataFrame, detuning_order: int = 1, corrected: bool = False):
    """Performs the full 2D/4D fitting of the data."""
    if detuning_order > 1:
        raise NotImplementedError(f"2D Analysis for detuning order {detuning_order:d} is not implemented "
                                  f"(only first order so far).")
    LOG.info("Performing amplitude detuning ODR for diagonal kicks.")
    data = {}

    # get all action arrays and all tune arrays, unfolded below
    for plane in PLANES:
        LOG.debug(f"Getting action and tune data for plane {plane}.")
        data[plane] = kick_file_modifiers.get_ampdet_data(
            kickac_df=kick_df,
            action_plane=plane,
            tune_plane=plane,
            corrected=corrected,
            dropna=False,  # so that they still have the same lengths
        )

    LOG.debug("Fitting ODR to kick data")
    odr_fits = fitting_tools.do_2d_kicks_odr(
        x=_get_ampdet_data_as_array(data, "action"),  # gets [2Jx, 2Jy]
        y=_get_ampdet_data_as_array(data, "tune"),  # gets [Qx, Qy]
        xerr=_get_ampdet_data_as_array(data, "action_err"),
        yerr=_get_ampdet_data_as_array(data, "tune_err"),
    )

    # add the fits to the kick header
    for t_plane in PLANES:
        for k_plane in PLANES:
            kick_df = kick_file_modifiers.add_odr(
                kickac_df=kick_df,
                odr_fit=odr_fits[t_plane][k_plane],
                action_plane=k_plane,
                tune_plane=t_plane,
                corrected=corrected
            )
    return kick_df


def get_approx_bbq_interval(
        bbq_df: TfsDataFrame, time_array: Sequence[CERNDatetime], window_length: int
) -> Tuple[CERNDatetime, CERNDatetime]:
    """Get approximate start and end times for averaging, based on window length and kick interval."""
    bbq_tmp = bbq_df.dropna()

    # convert to float to use math-comparisons
    ts_bbq_index = kick_file_modifiers.get_timestamp_index(bbq_tmp.index)
    ts_kick_index = kick_file_modifiers.get_timestamp_index(time_array)
    ts_start, ts_end = min(ts_kick_index), max(ts_kick_index)

    ts_bbq_min, ts_bbq_max = min(ts_bbq_index), max(ts_bbq_index)

    if not (ts_bbq_min <= ts_start <= ts_bbq_max):
        raise ValueError("The starting time of the kicks lies outside of the given BBQ times.")

    if not (ts_bbq_min <= ts_end <= ts_bbq_max):
        raise ValueError("The end time of the kicks lies outside of the given BBQ times.")

    i_start = max(ts_bbq_index.get_indexer([ts_start], method="nearest")[0] - window_length, 0)
    i_end = min(ts_bbq_index.get_indexer([ts_end], method="nearest")[0] + window_length, len(ts_bbq_index) - 1)

    return bbq_tmp.index[i_start], bbq_tmp.index[i_end]


# Private Functions ------------------------------------------------------------


def _check_analyse_opt(opt: DotDict):
    """Perform manual checks on opt-sturcture."""
    LOG.debug("Checking Options.")

    # for label
    if opt.label is None:
        opt.label = f"Amplitude Detuning for Beam {opt.beam:d}"

    filter_opt = None
    if (opt.bbq_in is not None) and (opt.bbq_in != INPUT_PREVIOUS):
        # check if cleaning is properly specified
        all_filter_opt = dict(
            cut=opt.get_subdict(
                ["bbq_filtering_method", "window_length", "tunes", "tune_cut", "fine_window", "fine_cut"]
            ),
            minmax=opt.get_subdict(
                ["bbq_filtering_method", "window_length", "tunes_minmax", "fine_window", "fine_cut"]
            ),
            outliers=opt.get_subdict(["bbq_filtering_method", "window_length", "outlier_limit"]),
        )

        for method, params in all_filter_opt.items():
            if opt.bbq_filtering_method == method:
                missing_params = [k for k, v in params.items() if v is None]
                if any(missing_params):
                    raise KeyError(
                        f"Missing parameters for cleaning method {method}: '{list2str(missing_params)}'"
                    )
                filter_opt = params

        if filter_opt.bbq_filtering_method == "cut":
            filter_opt[f"tunes_minmax"] = [
                minmax for t in opt.tunes for minmax in (t - opt.tune_cut, t + opt.tune_cut)
            ]
            filter_opt.pop("tune_cut")
            filter_opt.pop("tunes")

        if filter_opt.bbq_filtering_method != "outliers":
            # check fine cleaning
            if bool(filter_opt.fine_cut) != bool(filter_opt.fine_window):
                raise KeyError(
                    "To activate fine cleaning, both fine cut and fine window need to be specified"
                )

    if opt.output is not None:
        opt.output = Path(opt.output)

    return opt, filter_opt


def _get_bbq_data(beam: int, input_: Union[Path, str, int], kick_df: TfsDataFrame) -> TfsDataFrame:
    """
    Return BBQ data from input, either file or timber fill, as a ``TfsDataFrame``.

    Note: the ``input_`` parameter is always parsed from the commandline as a string, but could be 'kick'
    or a kickfile name or an integer. All these options will be tried until one works.
    """
    try:
        fill_number = int(input_)
    except (TypeError, ValueError) as e:  # input_ is a file name or the string 'kick'
        if input_ == INPUT_KICK:
            LOG.debug("Getting timber data from kick times")
            timber_keys, bbq_cols = _get_timber_keys_and_bbq_columns(beam)
            t_start = min(kick_df.index.to_numpy())
            t_end = max(kick_df.index.to_numpy())
            t_delta = timedelta(seconds=DTIME)
            data = timber_extract.extract_between_times(
                t_start - t_delta, t_end + t_delta, keys=timber_keys, names=dict(zip(timber_keys, bbq_cols))
            )
        else:  # input_ is a file name or path
            LOG.debug(f"Getting bbq data from file '{str(input_):s}'")
            data = read_timed_dataframe(input_)
            if not len(data.index):
                raise ValueError(f"No entries in {str(input_):s}.")
            #  Drop old moving average columns as these will be computed again later
            data.drop([get_mav_col(p) for p in PLANES if get_mav_col(p) in data.columns], axis="columns")

    else:  # input_ is a number, assumed to be a fill number
        LOG.debug(f"Getting timber data from fill '{input_:d}'")
        timber_keys, bbq_cols = _get_timber_keys_and_bbq_columns(beam)
        data = timber_extract.lhc_fill_to_tfs(
            fill_number, keys=timber_keys, names=dict(zip(timber_keys, bbq_cols))
        )
    return data


def _get_timber_keys_and_bbq_columns(beam: int) -> Tuple[List[str], List[str]]:
    keys = [get_timber_bbq_key(plane, beam) for plane in PLANES]
    cols = [get_bbq_col(plane) for plane in PLANES]
    return keys, cols


def _should_do_corrected(kick_df, bbq_in) -> List:
    if bbq_in is None:
        return []
    if bbq_in == INPUT_PREVIOUS and not any(CORRECTED in col for col in kick_df.columns):
        return []
    return [True]


def _save_options(opt: DotDict) -> None:
    if opt.output:
        save_config(Path(opt.output), opt, __file__)


def _get_ampdet_data_as_array(data: Dict[Any, AmpDetData], column: str) -> ArrayLike:
    """ Returns a matrix with number of rows as entries in data,
    each containing the values from the given column of the AmpDetData.
    e.g. [[Jx0, Jx1, Jx2, ....]
          [Jy0, Jy1, Jy2, ....]]
    """
    return np.vstack([getattr(d, column) for d in data.values()])


# Script Mode ##################################################################


if __name__ == "__main__":
    analyse_with_bbq_corrections()
