"""
Generate summary tables from K-modulation results.

This function collects and summarizes K-modulation results from multiple measurement directories.
For the given beam, it generates:
1. A DataFrame containing all imported K-modulation measurement results.
2. Optionally, a text file with the averaged results which can be used to create a logbook entry.

**Arguments:**

*--Required--*

- **meas_paths** *(Sequence[Path | str])*:

    Directories of K-modulation results to import.
    These need to be the paths to the root-folders containing B1 and B2 sub-dirs.

- **beam** *(int)*:

    Beam for which to import.

*--Optional--*

- **averaged_meas** *(dict[str, dict[int, tfs.TfsDataFrame]], default=None)*:
        Precomputed averaged K-modulation results. If provided, these are introduced in the
        summary table for the logbook.

- **output_dir** *(Path | str, default=None)*:
        Path to the directory where to write the output files.
        If None, no files are written.

- **logbook** *(str, default=None)*:
        Name of the logbook in which to publish the formatted summary tables.
        If None, no logbook entry is created.

**Returns:**

Tuple containing:

1. **kmod_summaries** *(dict[str, list[tfs.TfsDataFrame]])*:
        Mapping of beam names to lists of K-modulation summary DataFrames.

2. **tables_logbook** *(dict[str, list[list[str]]])*:
        Mapping of beam names to lists of formatted tables suitable for logbooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from generic_parser import DotDict

from omc3.optics_measurements.constants import (
    BEAM_DIR,
    BETASTAR,
    BETAWAIST,
    ERR,
    EXT,
    LABEL,
    RESULTS_FILE_NAME,
    WAIST,
)
from omc3.scripts.create_logbook_entry import main as create_logbook_entry
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Sequence

LOG = logging_tools.get_logger(__name__)

# Constants definitions for K-modulation
COLS_X = [f"{BETASTAR}X", f"{ERR}{BETASTAR}X", f"{BETAWAIST}X", f"{ERR}{BETAWAIST}X", f"{WAIST}X", f"{ERR}{WAIST}X"]
COLS_Y = [f"{BETASTAR}Y", f"{ERR}{BETASTAR}Y", f"{BETAWAIST}Y", f"{ERR}{BETAWAIST}Y", f"{WAIST}Y", f"{ERR}{WAIST}Y"]
IP_COLUMN = "IP"
NAME_COLUMN = "NAME"

def output_kmod_summary_tables(
        meas_paths: Sequence[Path | str],
        beam: int,
        averaged_meas: dict[str, dict[int, tfs.TfsDataFrame]] | None = None,
        output_dir: Path | str | None = None,
        logbook: str | None = None,
        ) -> tuple[dict[str, list[tfs.TfsDataFrame]],dict[str, list[list[str]]]]:

    """
    Args:
        meas_paths (Sequence[Path | str]): Paths to the K-modulation results.
        beam (int): Beam for which to average.
        averaged_meas (dict[int, tfs.TfsDataFrame]): If not None, averaged K-modulation results over all measurements are included in the summary tables. Default: None.
        output_dir (Path | str | None): Path to the output directory. Defaults to None.
        logbook (str = None): If provided, create a logbook entry containing the .txt summary tables to the given logbook. Default: None.

    Returns:
        Tuple[dict[str, tfs.TfsDataFrame], dict[str, list[str]]]:
            - Dictionary mapping beam names to K-modulation summary DataFrames.
            - Dictionary mapping beam names to lists of formatted text tables
    """

    LOG.info(f"Starting kmod summary importing for {BEAM_DIR}{beam}.")
    grouped = _collect_kmod_results(meas_paths=meas_paths, beam=beam)

    if averaged_meas is not None:
        grouped_averaged = _collect_averaged_kmod_results(averaged_meas=averaged_meas, beam=beam)

    LOG.debug(f"Processing result for: {beam}")
    if grouped:
        kmod_summary = tfs.concat(grouped, ignore_index=True)
        if averaged_meas:
            kmod_summary_averaged = tfs.concat(grouped_averaged, ignore_index=True)
        table_logbook = _prepare_logbook_table(beam=beam, kmod_summary = kmod_summary, kmod_summary_averaged = kmod_summary_averaged)

    if output_dir is not None:
        _save_outputs(beam=beam, save_output_dir=output_dir, table_logbook=table_logbook, summary_df=kmod_summary)
    else:
        LOG.info("Output_dir not provided: skipping saving files for all beams.")

    if logbook:
        _logbook_entry(beam=beam, logbook=logbook, logbook_file=table_logbook)
    if not logbook:
        LOG.info("Logbook name not provided: logbook entry not created.")

    return kmod_summary, table_logbook

def _collect_kmod_results(
    meas_paths: Sequence[Path | str],
    beam: int,
    ) -> list[tfs.TfsDataFrame]:

    """
    Gathers the various Kmod results.tfs dataframes, taking only cols_x and cols_y values.

    Args:
        meas_paths (Sequence[Path | str]): List of measurement directories containing beam subfolders.
        beam (int): Beam number to process.

    Returns:
        list[tfs.TfsDataFrame]: List containing grouped kmod results.
    """

    grouped = []
    for path in meas_paths:
        path = Path(path)
        LOG.info(f"Reading results: {path.absolute()}.")
        file_path = path / f"{BEAM_DIR}{beam}" / f"{RESULTS_FILE_NAME}{EXT}"
        if not file_path.exists():
            LOG.warning(f"Missing {RESULTS_FILE_NAME}{EXT}: {file_path}")
        else:
            meas_name = file_path.parent.parent.name
            data_file = tfs.read(file_path)
            try:
                label = data_file[f"{LABEL}"].iloc[0] # takes magnet names from label. ex. [MQXA1.L5-MQXA1.R5] from [0  MQXA1.L5-MQXA1.R5 Name: LABEL, dtype: object]
            except KeyError as e:
                LOG.debug("Could not parse IP from LABEL. Skipping entry.", exc_info=e)
                ip_name = "None"
            else:
                second_magnet = label.split("-")[1] # takes right magnet from magnet names. ex. [MQXA1.R5] from [MQXA1.L5-MQXA1.R5]
                relevant = second_magnet.split(".")[-1] # takes last part of the second magnet name. ex. [R5] from [MQXA1.R5]
                ip_number = relevant[1:] # isolate IP number. ex. [5] from [R5]
                ip_name = f"{IP_COLUMN}{ip_number}"
            df = data_file[COLS_X + COLS_Y].iloc[[0]] # returns a DataFrame with one row
            df.insert(0, NAME_COLUMN, meas_name)
            df.insert(0, IP_COLUMN, ip_name)
            grouped.append(df)
    return grouped

def _collect_averaged_kmod_results(
    averaged_meas: dict[str, dict[int, tfs.TfsDataFrame]],
    beam: int
    ) -> list[tfs.TfsDataFrame]:

    """
    Gathers the various averaged Kmod results dataframes, taking only cols_x and cols_y values.

    Args:
        meas_paths (Sequence[Path | str]): List of measurement directories containing beam subfolders.
        beam (int): Beam number to process.

    Returns:
        list[tfs.TfsDataFrame]: List containing grouped averaged kmod results.
    """

    grouped_averaged = []
    for av_ip, av_res in averaged_meas.items():
        LOG.info(f"Reading averaged results: {BEAM_DIR}{beam}, {av_ip}")
        av_tab = av_res[0] # 0: averaged results table, 1, 2: are betx, bety for IP closest elements
        av_tab_beam = av_tab.loc[[beam]]
        df_averaged = av_tab_beam[COLS_X + COLS_Y]
        df_averaged.insert(0, IP_COLUMN, av_ip)
        grouped_averaged.append(df_averaged)
    return grouped_averaged

def _prepare_logbook_table(
    beam: int,
    kmod_summary: tfs.TfsDataFrame,
    kmod_summary_averaged: tfs.TfsDataFrame | None = None,
    ) -> list[str]:

    """
    Prepare formatted logbook tables from K-modulation summary data.

    Args:
        beam (int): Beam number to process.
        kmod_summary (tfs.TfsDataFrame): DataFrame containing collected Kmod results.
        kmod_summary_averaged (tfs.TfsDataFrame | None): Optional DataFrame containing averaged Kmod results.

    Returns:
        list[str]: List of formatted string tables. Includes separate sections for X-plane, Y-plane, and optionally averaged results.
    """

    kmod_summary_x = kmod_summary[[IP_COLUMN, NAME_COLUMN] + COLS_X]
    kmod_summary_y = kmod_summary[[IP_COLUMN, NAME_COLUMN] + COLS_Y]
    kmod_summary_x_averaged = kmod_summary_averaged[[IP_COLUMN] + COLS_X]
    kmod_summary_y_averaged = kmod_summary_averaged[[IP_COLUMN] + COLS_Y]

    table_logbook = []
    for plane, df in [("X", kmod_summary_x), ("Y", kmod_summary_y)]:
        table_str = df.to_string()
        table_logbook.append(f"\n============================================ {BEAM_DIR}{beam} Results ({plane}-plane) ====================================================\n\n{table_str}\n")
    if kmod_summary_x_averaged is not None and kmod_summary_y_averaged is not None:
        for plane, df_averaged in [("X", kmod_summary_x_averaged), ("Y", kmod_summary_y_averaged)]:
            table_str_averaged = df_averaged.to_string()
            table_logbook.append(f"\n======================================== {BEAM_DIR}{beam} Averaged Results ({plane}-plane) ===============================================\n\n{table_str_averaged}\n")
    return table_logbook

def _save_outputs(
    beam: int,
    table_logbook: list[str],
    summary_df: tfs.TfsDataFrame,
    save_output_dir: Path | str,
    ) -> None:

    """
    Save logbook text output and .tfs summary for a given beam.

    Args:
        beam (int): Beam number to process.
        table_logbook (list[str]): Single .txt file.
        summary_df (tfs.TfsDataFrame): Single .tfs dataframe.
        save_output_dir (Path | str ): Path to the saving output directory.
    """

    logbook_table_path = save_output_dir / f"{BEAM_DIR}{beam}_kmod_summary.txt"
    summary_path = save_output_dir / f"{BEAM_DIR}{beam}_kmod_summary{EXT}"
    with logbook_table_path.open("w") as f:
        f.write("\n".join(table_logbook))
    tfs.write(summary_path, summary_df)
    LOG.info(f"Writing .txt summary output file {logbook_table_path}.")
    LOG.info(f"Writing {EXT} summary output file {summary_path}.")
    return

def _logbook_entry(
    beam: int,
    logbook: str,
    logbook_file: str | list[str]
    ) -> None:

    """
    Create logbook entry with .txt generated table for a given beam.

    Args:
        beam (int): Beam number to process.
        logbook (str): logbook name, ex. 'LHC_OMC'
        logbook_file (str | list[str]): .txt file to save..
    """

    if isinstance(logbook_file, list):
        logbook_text = "\n".join(logbook_file)
    elif isinstance(logbook_file, str):
        logbook_text = logbook_file
    else:
        raise TypeError(f"logbook_file must be str or list[str], got {type(logbook_file)}")

    logbook_filename = f"{BEAM_DIR}{beam}_kmod_summary"
    logbook_event = DotDict({
            "text": logbook_text,
            # "files": [],
            # "filenames": [],
            "logbook": f"{logbook}"
        })
    LOG.info(f"Creating logbook entry for {logbook_filename} to {logbook}.")
    _ = create_logbook_entry(logbook_event)

# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    output_kmod_summary_tables()
