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
    1. **kmod_summary** *(tfs.TfsDataFrame)*: Combined K-modulation results for the given beam.
    2. **table_logbook** *(list[str])*: Formatted text tables suitable for logbook entries.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from generic_parser import DotDict
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint

from omc3.optics_measurements.constants import (
    BEAM_DIR,
    BETASTAR,
    BETAWAIST,
    EFFECTIVE_BETAS_FILENAME,
    ERR,
    EXT,
    LABEL,
    NAME,
    RESULTS_FILE_NAME,
    WAIST,
)
from omc3.scripts.create_logbook_entry import main as create_logbook_entry
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr

if TYPE_CHECKING:
    from collections.abc import Sequence

LOG = logging_tools.get_logger(__name__)


def _get_params() -> EntryPointParameters:
    params = EntryPointParameters()

    params.add_parameter(
        name="meas_paths",
        type=PathOrStr,
        nargs="+",
        required=True,
        help="Directories of K-modulation results to import.",
    )
    params.add_parameter(
        name="beam",
        type=int,
        required=True,
        help="Beam for which to import.",
    )
    params.add_parameter(
        name="averaged_meas",
        default=None,
        help="Optional precomputed averaged K-modulation results.",
    )
    params.add_parameter(
        name="lumi_imbalance",
        action="store_true",
        help="Include luminosity imbalance in the summary tables.",
    )
    params.add_parameter(
        name="output_dir",
        type=PathOrStr,
        default=None,
        help="Path to the directory where to write the output files.",
    )
    params.add_parameter(
        name="logbook",
        type=str,
        default=None,
        help="Name of the logbook to publish the summary tables.",
    )

    return params


# Constants definitions for K-modulation
COLS_X = [
    f"{BETASTAR}X",
    f"{ERR}{BETASTAR}X",
    f"{BETAWAIST}X",
    f"{ERR}{BETAWAIST}X",
    f"{WAIST}X",
    f"{ERR}{WAIST}X",
]
COLS_Y = [
    f"{BETASTAR}Y",
    f"{ERR}{BETASTAR}Y",
    f"{BETAWAIST}Y",
    f"{ERR}{BETAWAIST}Y",
    f"{WAIST}Y",
    f"{ERR}{WAIST}Y",
]
IP_COLUMN = "IP"
NAME_COLUMN = "NAME"


@entrypoint(_get_params(), strict=True)
def output_kmod_summary_tables(opt: DotDict) -> tuple[tfs.TfsDataFrame, list[str]]:
    """
    Args:
        meas_paths (Sequence[Path | str]): Paths to the K-modulation results.
        beam (int): Beam for which to average.
        averaged_meas (dict[str, dict[int, tfs.TfsDataFrame]]): If not None, averaged K-modulation results over all measurements are included in the summary tables. Default: None.
        lumi_imbalance (bool): If True, luminosity imbalance results are included in the summary tables. Default: False.
        output_dir (Path | str): Path to the output directory. Defaults to None.
        logbook (str): If provided, create a logbook entry containing the .txt summary tables to the given logbook. Default: None.

    Returns:
        Tuple[tfs.TfsDataFrame, list[str]]:
            - Dataframe containing K-modulation summary.
            - List of formatted text table containing K-modulation summary.
    """
    kmod_summary_averaged = None
    kmod_summary_lumiimb = None

    LOG.info(f"Starting kmod summary importing for {BEAM_DIR}{opt.beam}.")
    grouped = _collect_kmod_results(beam=opt.beam, meas_paths=opt.meas_paths)

    if opt.averaged_meas is not None:
        grouped_averaged = _collect_averaged_kmod_results(
            beam=opt.beam, averaged_meas=opt.averaged_meas
        )

    if opt.lumi_imbalance:
        kmod_summary_lumiimb = _get_lumi_imbalance(output_dir=opt.output_dir)
    else:
        LOG.info("Luminosity imbalance calculation skipped.")

    LOG.debug(f"Processing result for: {opt.beam}")
    if grouped:
        kmod_summary = tfs.concat(grouped, ignore_index=True)
        if opt.averaged_meas is not None:
            kmod_summary_averaged = tfs.concat(grouped_averaged, ignore_index=True)
        table_logbook = _prepare_logbook_table(
            beam=opt.beam,
            kmod_summary=kmod_summary,
            kmod_summary_averaged=kmod_summary_averaged,
            kmod_summary_lumiimb=kmod_summary_lumiimb,
        )

        if opt.output_dir is not None:
            _save_outputs(
                beam=opt.beam,
                save_output_dir=opt.output_dir,
                txt_to_save="\n".join(table_logbook),
                df_to_save=kmod_summary,
            )
        else:
            LOG.info("Output_dir not provided: skipping saving files for all beams.")

        if opt.logbook:
            _summary_logbook_entry(
                beam=opt.beam, logbook=opt.logbook, logbook_entry_text="\n".join(table_logbook)
            )
        else:
            LOG.info("Logbook name not provided: logbook entry not created.")

    return kmod_summary, table_logbook


def _collect_kmod_results(beam: int, meas_paths: Sequence[Path | str]) -> list[tfs.TfsDataFrame]:
    """
    Gathers the various Kmod results.tfs dataframes, taking only cols_x and cols_y values.

    Args:
        beam (int): Beam number to process.
        meas_paths (Sequence[Path | str]): List of kmod measurement directories containing beam subfolders.

    Returns:
        list[tfs.TfsDataFrame]: List containing grouped kmod results.
    """

    LOG.info("Grouping kmod results.")

    grouped = []
    for path in meas_paths:
        path = Path(path)
        LOG.info(f"Reading measurement results at '{path.absolute()}'.")
        file_path = path / f"{BEAM_DIR}{beam}" / f"{RESULTS_FILE_NAME}{EXT}"
        if not file_path.exists():
            LOG.warning(f"Missing {RESULTS_FILE_NAME}{EXT}: {file_path}")
        else:
            meas_name = path.name
            result_df = tfs.read(file_path)
            try:
                label = result_df[LABEL].iloc[
                    0
                ]  # takes magnet names from label, e.g. MQXA1.L5-MQXA1.R5
            except KeyError as e:
                LOG.debug("Could not parse IP from LABEL. Skipping entry.", exc_info=e)
                ip_name = "None"
            else:
                second_magnet = label.split("-")[
                    1
                ]  # takes right magnet from magnet names. ex. [MQXA1.R5] from [MQXA1.L5-MQXA1.R5]
                relevant = second_magnet.split(".")[
                    -1
                ]  # takes last part of the second magnet name. ex. [R5] from [MQXA1.R5]
                ip_number = relevant[1:]  # isolate IP number. ex. [5] from [R5]
                ip_name = f"{IP_COLUMN}{ip_number}"
            df = result_df[COLS_X + COLS_Y].iloc[[0]]  # returns a DataFrame with one row
            df.insert(0, NAME_COLUMN, meas_name)
            df.insert(0, IP_COLUMN, ip_name)
            grouped.append(df)
    return grouped


def _collect_averaged_kmod_results(
    beam: int, averaged_meas: dict[str, dict[int, tfs.TfsDataFrame]]
) -> list[tfs.TfsDataFrame]:
    """
    Gathers the various averaged Kmod results dataframes, taking only cols_x and cols_y values.

    Args:
        beam (int): Beam number to process.
        averaged_meas (dict[str, dict[int, tfs.TfsDataFrame]]): Precomputed averaged K-modulation results.

    Returns:
        list[tfs.TfsDataFrame]: The gathered averaged kmod results, grouped and filtered for the relevant columns only.
    """
    LOG.info("Grouping averaged kmod results.")

    grouped_averaged = []
    for av_ip, av_res in averaged_meas.items():
        LOG.info(f"Reading averaged results: {BEAM_DIR}{beam}, {av_ip}")
        av_tab = av_res[
            0
        ]  # 0: averaged results table, 1, 2: are betx, bety for IP closest elements
        if beam not in av_tab.index:
            LOG.warning("Beam %s not found in averaged results for IP %s. Skipping.", beam, av_ip)
            continue
        av_tab_beam = av_tab.loc[[beam]]
        df_averaged = av_tab_beam[COLS_X + COLS_Y]
        df_averaged.insert(0, IP_COLUMN, av_ip)
        grouped_averaged.append(df_averaged)
    return grouped_averaged


def _get_lumi_imbalance(output_dir: Path | str) -> str:
    """
    Gathers the various luminosity imbalance dataframes in a single str, one line for each file.

    Args:
        output_dir (Path | str ): Path to the folder with lumi imbalance files. If None, luminosity imbalance is skipped.
    Returns:
        str: Formatted table showing grouped luminosity imbalance results, one line per file.
    """
    LOG.info("Gathering luminosity imbalance results.")

    if output_dir is None:
        LOG.info("No output_dir provided: skipping luminosity imbalance calculation.")
        return ""

    output_dir = Path(output_dir)
    prefix = EFFECTIVE_BETAS_FILENAME.split("{")[
        0
    ]  # if the file name starts with EFFECTIVE_BETAS_FILENAME till {ip}
    report_lines = []
    for df_path in output_dir.iterdir():
        if not df_path.is_file() or not df_path.name.startswith(prefix):
            continue

        df = tfs.read(df_path)
        lumi_imbalance = float(df.headers.get("LUMIIMBALANCE", 0))
        err_lumi_imbalance = float(df.headers.get("ERRLUMIIMBALANCE", 0))
        ips = df[NAME].tolist()

        # final check
        if lumi_imbalance is not None and err_lumi_imbalance is not None and len(ips) >= 2:
            report_lines.append(
                f"Luminosity imbalance in between {ips[0]} and {ips[1]} is {lumi_imbalance} ± {err_lumi_imbalance}"
            )

    return "\n".join(report_lines)


def _prepare_logbook_table(
    beam: int,
    kmod_summary: tfs.TfsDataFrame,
    kmod_summary_averaged: tfs.TfsDataFrame | None = None,
    kmod_summary_lumiimb: str | None = None,
) -> list[str]:
    """
    Prepare formatted logbook tables from K-modulation summary data.

    Args:
        beam (int): Beam number to process.
        kmod_summary (tfs.TfsDataFrame): DataFrame containing collected Kmod results.
        kmod_summary_averaged (tfs.TfsDataFrame | None): Optional DataFrame containing averaged Kmod results.
        kmod_summary_lumiimb (str | None): Optional string containing formatted luminosity imbalance results.

    Returns:
        list[str]: List of formatted tables. Includes separate entries for X and Y planes, and optionally averaged results.
    """
    LOG.info("Formatting the tables to text.")

    kmod_summary_x = kmod_summary[[IP_COLUMN, NAME_COLUMN] + COLS_X]
    kmod_summary_y = kmod_summary[[IP_COLUMN, NAME_COLUMN] + COLS_Y]

    logbook_tables = []
    if kmod_summary_lumiimb:
        logbook_tables.append(
            f"=============================== Luminosity Imbalance =======================================\n{kmod_summary_lumiimb}\n"
        )
    for plane, df in [("X", kmod_summary_x), ("Y", kmod_summary_y)]:
        table_str = df.to_string()
        logbook_tables.append(
            f"\n=============================== {BEAM_DIR}{beam} Results ({plane}-plane) =======================================\n\n{table_str}\n"
        )
    if kmod_summary_averaged is not None:
        kmod_summary_x_averaged = kmod_summary_averaged[[IP_COLUMN] + COLS_X]
        kmod_summary_y_averaged = kmod_summary_averaged[[IP_COLUMN] + COLS_Y]
        for plane, df_averaged in [("X", kmod_summary_x_averaged), ("Y", kmod_summary_y_averaged)]:
            table_str_averaged = df_averaged.to_string()
            logbook_tables.append(
                f"\n=========================== {BEAM_DIR}{beam} Averaged Results ({plane}-plane) ==================================\n\n{table_str_averaged}\n"
            )
    return logbook_tables


def _save_outputs(
    beam: int,
    txt_to_save: str,
    df_to_save: tfs.TfsDataFrame,
    save_output_dir: Path | str,
) -> None:
    """
    Save logbook text output and .tfs summary for a given beam.

    Args:
        beam (int): Beam number to process.
        txt_to_save (str): .txt to be saved.
        df_to_save (tfs.TfsDataFrame): .tfs dataframe to be saved.
        save_output_dir (Path | str ): Path to the saving output directory.
    """

    save_output_dir = Path(save_output_dir)
    logbook_table_path = save_output_dir / f"{BEAM_DIR}{beam}_kmod_summary.txt"
    summary_path = save_output_dir / f"{BEAM_DIR}{beam}_kmod_summary{EXT}"
    LOG.info(f"Writing .txt summary output file {logbook_table_path}.")
    LOG.info(f"Writing {EXT} summary output file {summary_path}.")
    with logbook_table_path.open("w") as f:
        f.write(txt_to_save)
    tfs.write(summary_path, df_to_save)


def _summary_logbook_entry(beam: int, logbook: str, logbook_entry_text: str) -> None:
    """
    Create logbook entry with .txt generated table for a given beam.

    Args:
        beam (int): Beam number to process.
        logbook (str): logbook name, ex. 'LHC_OMC'
        logbook_entry_text (str): Text for logbook entry.
    """

    if isinstance(logbook_entry_text, str):
        logbook_text = logbook_entry_text
    else:
        raise TypeError(f"entry_text must be str, got {type(logbook_entry_text)}.")

    logbook_filename = f"{BEAM_DIR}{beam}_kmod_summary"
    logbook_event = DotDict(
        {
            "text": logbook_text,
            # "files": [],
            # "filenames": [],
            "logbook": logbook,
        }
    )
    LOG.info(f"Creating logbook entry for {logbook_filename} to {logbook}.")
    _ = create_logbook_entry(logbook_event)


# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    output_kmod_summary_tables()
