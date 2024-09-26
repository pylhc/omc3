""" 
Average K-Modulation Results
----------------------------

Average muliple K-Modulation results into a single file/dataframe.


**Arguments:**

*--Required--*

- **betastar** *(float)*:

    Model beta-star value of measurements.


- **ip** *(int)*:

    Specific ip to average over.


- **meas_paths** *(PathOrStr)*:

    Directories of Kmod results to import.


*--Optional--*

- **output_dir** *(PathOrStr)*:

    Path to the directory where to write the output files.


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

import pandas as pd
import tfs
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint
from omc3.plotting.plot_kmod_results import plot_kmod_results 

from omc3.kmod.constants import (
    BEAM,
    BEAM_DIR,
    BETA,
    BETASTAR,
    ERR,
    EXT,
    LABEL,
    LSA_FILE_NAME,
    MDL,
    RESULTS_FILE_NAME,
    TIME,
    AVERAGED_BETASTAR_FILENAME,
    AVERAGED_BPM_FILENAME
)
from omc3.optics_measurements.constants import NAME
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from generic_parser import DotDict

LOG = logging_tools.get_logger(__name__)


def _get_params():
    """
    A function to create and return EntryPointParameters for Kmod average.
    """
    params = EntryPointParameters()
    params.add_parameter(
        name="meas_paths",
        required=True,
        nargs="+",
        type=PathOrStr,
        help="Directories of Kmod results to import.",
    )
    params.add_parameter(
        name="ip", 
        required=True, 
        type=int, 
        help="Specific ip to average over."
    )
    params.add_parameter(
        name="betastar",
        required=True,
        type=float,
        help="Model beta-star value of measurements.",
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
def average_kmod_results_entrypoint(opt: DotDict) -> dict[int, tfs.TfsDataFrame]:
    """
    Reads kmod results and averages over the different measurements.

    Args:
        meas_paths (Sequence[Path|str]):
            A sequence of kmod BPM results files to import. This can include either single 
            measurements (e.g., 'lsa_results.tfs') or averaged results 
            (e.g., 'averaged_bpm_beam1_ip1_beta0.22m.tfs').

        ip (int):
            The specific IP to average over.

        betastar (float):
            The model beta-star value of the measurements.

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
    LOG.info("Starting Kmod averaging.")
    if opt.output_dir is not None:
        opt.output_dir = Path(opt.output_dir)
        opt.output_dir.mkdir(exist_ok=True)
        save_config(opt.output_dir, opt, __file__)

    meas_paths = [Path(m) for m in opt.meas_paths]

    averaged_results = get_average_betastar_results(meas_paths, opt.betastar)
    averaged_bpm_results = get_average_bpm_betas_results(meas_paths)

    if opt.output_dir is not None:
        filename = AVERAGED_BETASTAR_FILENAME.format(ip=opt.ip, betastar=opt.betastar)
        tfs.write(opt.output_dir / f'{filename}{EXT}', averaged_results, save_index=BEAM)
        
        for beam, df in averaged_bpm_results.items():
            filename = AVERAGED_BPM_FILENAME.format(beam=beam, ip=opt.ip, betastar=opt.betastar)
            tfs.write( opt.output_dir / f'{filename}{EXT}', df, save_index=NAME)

    if opt.plot:
        plot_kmod_results(
            data=averaged_results, 
            ip=opt.ip, 
            output_dir=opt.output_dir, 
            show=opt.show_plots
        )

    averaged_bpm_results[0] = averaged_results
    return averaged_bpm_results


def get_average_betastar_results(meas_paths: Sequence[Path], betastar: float) -> tfs.TfsDataFrame:
    """
    Calculate the average betastar results for the given measurements.

    Args:
        meas_paths: The paths to the measurements.
        betastar: The model betastar value.

    Returns:
        The final results as a DataFrame; both beams merged.
    """
    final_results = []
    for beam in [1, 2]:
        try:
            all_dfs = [
                tfs.read(dir_path / f"{BEAM_DIR}{beam}" / f"{RESULTS_FILE_NAME}{EXT}").drop(
                    columns=[LABEL, TIME]
                )
                for dir_path in meas_paths
            ]
        except FileNotFoundError as e:
            LOG.warning(f"Could not find all results for beam {beam}. Skipping.", exc_info=e)
            continue

        mean_df, std_df =  _get_mean_and_std(all_dfs)

        for column in mean_df.columns:
            if not column.startswith(ERR):
                mean_df[f'{ERR}{column}'] = std_df[column]  # TODO: maybe add KMOD errors?

        mean_df[f'{BETASTAR}{MDL}'] = betastar
        mean_df[BEAM] = beam
        final_results.append(mean_df)
    final_results = tfs.concat(final_results).set_index(BEAM)
    return final_results


def get_average_bpm_betas_results(meas_paths: Sequence[Path]) -> dict[int, tfs.TfsDataFrame]:
    """
    Calculate the average bpm betas results for the given measurements.

    Args:
        meas_paths: The paths to the measurements.

    Returns:
        final_results: A dictionary containing the average bpm betas results for each beam.
    """
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
        
        mean_df, std_df =  _get_mean_and_std(all_dfs)

        for plane in "XY":
            mean_df[f'{ERR}{BETA}{plane}'] = std_df[f'{BETA}{plane}']
        final_results[beam] = mean_df
    return final_results


def _get_mean_and_std(dfs: Sequence[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:  
    grouped = tfs.concat(dfs, axis=0).groupby(level=0)  # append rows, group by index
    mean_df = grouped.mean()
    std_df = grouped.std(ddof=0)  # ddof=0 is default in numpy, ddof=1 is default in pandas
    return mean_df, std_df


if __name__ == "__main__":
    average_kmod_results_entrypoint()
