""" 
Average K-Modulation Results
----------------------------

Average muliple K-Modulation results into a single file/dataframe.


**Arguments:**

*--Required--*

- **meas_paths** *(PathOrStr)*:

    Directories of K-modulation results to average.


*--Optional--*

- **output_dir** *(PathOrStr)*:

    Path to the directory where to write the output files.


- **betastar** *(float)*:

    Model beta-star values (x, y) of measurements. Only used for filename and plot.


- **ip** *(int)*:

    IP this result is from. Only used for filename and plot.


- **plot**:

    Plot the averaged results.

    action: ``store_true``


- **show_plots**:

    Show the plots.

    action: ``store_true``

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint

from omc3.optics_measurements.constants import (
    AVERAGED_BETASTAR_FILENAME,
    AVERAGED_BPM_FILENAME,
    BEAM,
    BEAM_DIR,
    ERR,
    EXT,
    LABEL,
    LSA_FILE_NAME,
    NAME,
    RESULTS_FILE_NAME,
    TIME,
    S,
)
from omc3.plotting.plot_kmod_results import plot_kmod_results
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config
from omc3.utils.stats import weighted_error, weighted_mean

if TYPE_CHECKING:
    from collections.abc import Sequence

    from generic_parser import DotDict

LOG = logging_tools.get_logger(__name__)

COLUMNS_TO_DROP: tuple[str, ...] = (TIME, )
COLUMNS_NO_AVERAGE: tuple[str, ...] = (S, LABEL, NAME)

def _get_params() -> EntryPointParameters:
    """
    A function to create and return EntryPointParameters for K-modulation average.
    """
    params = EntryPointParameters()
    params.add_parameter(
        name="meas_paths",
        required=True,
        nargs="+",
        type=PathOrStr,
        help="Directories of K-modulation results to average.",
    )
    params.add_parameter(
        name="ip", 
        type=int, 
        help="IP this result is from. Only used for filename and plot."
    )
    params.add_parameter(
        name="betastar",
        type=float,
        nargs="+",
        help="Model beta-star values (x, y) of measurements. Only used for filename and plot.",
    )
    params.add_parameter(
        name="output_dir",
        type=PathOrStr,
        help="Path to the directory where to write the output files.",
    )
    params.add_parameter(
        name="plot", 
        action="store_true", 
        help="Plot the averaged results."
    )
    params.add_parameter(
        name="show_plots", 
        action="store_true", 
        help="Show the plots."
    )
    return params


@entrypoint(_get_params(), strict=True)
def average_kmod_results(opt: DotDict) -> dict[int, tfs.TfsDataFrame]:
    """
    Reads kmod results and averages over the different measurements.

    Args:
        meas_paths (Sequence[Path|str]):
            Directories of K-modulation results to aver.

        ip (int):
            The specific IP to average over.
            Only used for filename.

        betastar (float):
            The model beta-star values (x, y) of the measurements.
            If a single value is given, beta-star_x == beta-star_y is assumed.
            Only used for filename.

        output_dir (Path|str):
            Path to the output directory to write out the averaged results. Optional.

        plot (bool):
            If True, plots the averaged results. Default: False.
        
        show_plots (bool):
            If True, show the plots. Default: False.

    Returns:
        Dictionary of averaged kmod-result DataFrames by beams for the 
        bpm-data and with key `0` for the beta-star data.
    """
    LOG.info("Starting K-mod averaging.")
    if opt.output_dir is not None:
        if opt.betastar is None:
            raise ValueError("Betastar not given. Cannot write out results.")
        
        if opt.ip is None:
            raise ValueError("IP not given. Cannot write out results.")

        opt.output_dir = Path(opt.output_dir)
        opt.output_dir.mkdir(exist_ok=True)
        save_config(opt.output_dir, opt, __file__)


    meas_paths = [Path(m) for m in opt.meas_paths]

    averaged_results = get_average_betastar_results(meas_paths)
    averaged_bpm_results = get_average_bpm_results(meas_paths)

    if opt.output_dir is not None:
        if len(opt.betastar) == 1:
            opt.betastar = [opt.betastar[0], opt.betastar[0]]

        filename = AVERAGED_BETASTAR_FILENAME.format(ip=opt.ip, betastar_x=opt.betastar[0], betastar_y=opt.betastar[1])
        tfs.write(opt.output_dir / f'{filename}{EXT}', averaged_results, save_index=BEAM)
        
        for beam, df in averaged_bpm_results.items():
            filename = AVERAGED_BPM_FILENAME.format(beam=beam, ip=opt.ip, betastar_x=opt.betastar[0], betastar_y=opt.betastar[1])
            tfs.write(opt.output_dir / f'{filename}{EXT}', df, save_index=NAME)

    if opt.plot:
        plot_kmod_results(
            data=averaged_results, 
            ip=opt.ip, 
            betastar=opt.betastar,
            output_dir=opt.output_dir, 
            show=opt.show_plots
        )

    averaged_bpm_results[0] = averaged_results
    return averaged_bpm_results


def get_average_betastar_results(meas_paths: Sequence[Path]) -> tfs.TfsDataFrame:
    """
    Calculate the average betastar results for the given measurements.

    Args:
        meas_paths: The paths to the measurements.

    Returns:
        The final results as a DataFrame; both beams merged.
    """
    LOG.debug("Averaging beta-star results.")
    final_results = {}
    for beam in [1, 2]:
        try:
            all_dfs = [
                tfs.read(dir_path / f"{BEAM_DIR}{beam}" / f"{RESULTS_FILE_NAME}{EXT}")
                for dir_path in meas_paths
            ]
        except FileNotFoundError as e:
            LOG.warning(f"Could not find all results for beam {beam}. Skipping.", exc_info=e)
            continue

        mean_df =  _get_averaged_df(all_dfs)
        if LABEL in mean_df.columns:
            mean_df[NAME] = mean_df[LABEL].apply(lambda s: f"IP{s[-1]}")
            mean_df = mean_df.drop(columns=[LABEL])
        
        mean_df[BEAM] = beam
        final_results[beam] = mean_df
    final_df = tfs.concat(final_results.values()).set_index(BEAM)
    return final_df


def get_average_bpm_results(meas_paths: Sequence[Path]) -> dict[int, tfs.TfsDataFrame]:
    """
    Calculate the average results for BPMs/IPs for the given measurements.

    Args:
        meas_paths: The paths to the measurements.

    Returns:
        final_results: A dictionary containing the average bpm betas results for each beam.
    """
    LOG.debug("Averaging bpm results.")
    final_results = {}

    for beam in [1, 2]:
        try:
            all_dfs = [
                tfs.read(dir_path / f"{BEAM_DIR}{beam}" / f"{LSA_FILE_NAME}{EXT}", index=NAME)
                for dir_path in meas_paths
            ]
        except FileNotFoundError as e:
            LOG.warning(f"Could not find all results for beam {beam}. Skipping.", exc_info=e)
            continue
        
        final_results[beam] = _get_averaged_df(all_dfs)
    return final_results


def _get_averaged_df(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """ Calculate the average over the data in the given dfs.
    
    This function calculates the means and errors over the given dataframes, 
    using the weighted_mean and weighted_error functions from the `stats` module,
    which takes the standard deviation of the data and their errors into account.
    If no error column is present, 

    It is assumed the same columns are present in all dataframes, 
    and the average is only done on rows, that have common indices.
    The order of rows and columns is irrelevant.

    In case only a single dataframe is given, this frame is returned, instead of doing calculations.
    """
    # Select columns for averaging, determines column order of output
    columns = dfs[0].columns
    no_average_cols = list(columns.intersection(COLUMNS_NO_AVERAGE))
    drop_cols = list(columns.intersection(COLUMNS_TO_DROP))

    # Check if we actually need to average
    if len(dfs) == 1:
        LOG.warning("Only single DataFrame given for averaging -> Returning it.")
        return dfs[0].drop(columns=drop_cols)

    # Check indices, which also determines row order of output
    index = dfs[0].index
    for df in dfs[1:]:
        index = index.intersection(df.index)

    if not len(index):
        msg = "Cannot perform averaging as the files all have no common indices."
        raise KeyError(msg)


    data_cols = [
        col for col in columns 
        if (not col.startswith(ERR)) and (col not in no_average_cols + drop_cols)
    ]

    # Compute the weighted mean and weighted error for each column 
    avg_df = dfs[0].loc[index, no_average_cols]

    for data_col in data_cols:
        err_col = f"{ERR}{data_col}"
        if err_col not in columns:
            err_col = None

        # Select data to average
        data = np.array([df.loc[index, data_col].values for df in dfs])
        errors = None
        if err_col is not None:
            errors = np.array([df.loc[index, err_col].values for df in dfs])
        
        # Compute weighted mean and error
        avg_df.loc[index, data_col] = weighted_mean(data, errors, axis=0)
        avg_df.loc[index, err_col] = weighted_error(data, errors, axis=0, t_value_corr=False)

    return avg_df


if __name__ == "__main__":
    average_kmod_results()
