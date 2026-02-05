from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from generic_parser import DotDict

from omc3.optics_measurements.constants import (
    AVERAGED_BETASTAR_FILENAME,
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

if TYPE_CHECKING:
    from collections.abc import Sequence

LOG = logging_tools.get_logger(__name__)

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


def _collect_kmod_results(beam: int, meas_paths: Sequence[Path | str]) -> list[tfs.TfsDataFrame]:
    """
    Gathers the kmod results.tfs dataframes, taking only cols_x and cols_y values for the given beam.

    Args:
        beam (int): Beam number to process.
        meas_paths (Sequence[Path | str]): List of kmod measurement directories containing beam subfolders.

    Returns:
        list[tfs.TfsDataFrame]: List containing grouped kmod results.
    """

    LOG.info("Gathering kmod results.")

    grouped: list[tfs.TfsDataFrame] = []
    meas_path in map(Path, meas_paths):
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


def _collect_averaged_kmod_results(beam: int, output_dir: Path | str) -> list[tfs.TfsDataFrame]:
    """
    Gathers the averaged kmod results dataframes, taking only cols_x and cols_y values for the given beam.

    Args:
        beam (int): Beam number to process.
        output_dir (Path | str ): Path to the folder with averaged kmod dataframes. If None, averaged kmod results are not collected.

    Returns:
        list[tfs.TfsDataFrame]: List containing grouped averaged kmod results.
    """
    LOG.info("Gathering averaged kmod results.")

    if output_dir is None:
        LOG.info("No output_dir provided: skipping averaged kmod results gathering.")
        return []

    output_dir = Path(output_dir)
    prefix = AVERAGED_BETASTAR_FILENAME.split("{")[
        0
    ]  # if the file name starts with AVERAGED_BETASTAR_FILENAME till {ip}
    grouped_averaged = []
    for df_path in output_dir.iterdir():
        if not df_path.is_file() or not df_path.name.startswith(prefix):
            continue
        df = tfs.read(df_path)
        if beam not in df.index:
            LOG.warning("Beam %s not found in averaged results. Skipping.", beam)
            continue
        df_beam_row = df.loc[[beam]]
        df_beam_row_col = df_beam_row[COLS_X + COLS_Y]
        ip_name = df_beam_row[NAME]
        df_beam_row_col.insert(0, IP_COLUMN, ip_name)
        grouped_averaged.append(df_beam_row_col)
    return grouped_averaged


def _collect_lumi_imbalance_results(output_dir: Path | str) -> str:
    """
    Gathers the luminosity imbalance results.

    Args:
        output_dir (Path | str ): Path to the folder with luminosity imbalance dataframes. If None, luminosity imbalance results are not collected.
    Returns:
        str: Formatted table showing grouped luminosity imbalance results, one line per file.
    """
    LOG.info("Gathering luminosity imbalance results.")

    if output_dir is None:
        LOG.info("No output_dir provided: skipping luminosity imbalance gathering.")
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
    meas_paths: Sequence[Path | str],
    kmod_averaged_output_dir: Path | str | None = None,
    lumi_imb_output_dir: Path | str | None = None,
) -> tuple[tfs.TfsDataFrame, list[str]]:
    """
    Prepare formatted logbook tables from K-modulation summary data.

    Args:
        beam (int): Beam number to process.
        meas_paths (Sequence[Path | str]): List of kmod measurement directories containing beam subfolders.
        kmod_averaged_output_dir (Path | str): Path to the directory containing averaged kmod tfs results files. Defaults to None.
        lumi_imb_output_dir (Path | str): Path to the output directory containing luminosity imbalance tfs results files. Defaults to None.

    Returns:
        Tuple[tfs.TfsDataFrame, list[str]]:
            - Dataframe containing K-modulation summary.
            - List of formatted text table containing K-modulation summary.
    """
    LOG.info("Formatting the tables to text.")

    grouped_kmod = _collect_kmod_results(beam=beam, meas_paths=meas_paths)
    if grouped_kmod:
        kmod_summary = tfs.concat(grouped_kmod, ignore_index=True)
        kmod_summary_x = kmod_summary[[IP_COLUMN, NAME_COLUMN] + COLS_X]
        kmod_summary_y = kmod_summary[[IP_COLUMN, NAME_COLUMN] + COLS_Y]

    logbook_tables = []
    if lumi_imb_output_dir is not None:
        kmod_summary_lumiimb = _collect_lumi_imbalance_results(output_dir=lumi_imb_output_dir)
        logbook_tables.append(
            f"=============================== Luminosity Imbalance =======================================\n{kmod_summary_lumiimb}\n"
        )
    else:
        LOG.info("Luminosity imbalance results not included in the text table.")

    for plane, df in [("X", kmod_summary_x), ("Y", kmod_summary_y)]:
        table_str = df.to_string()
        logbook_tables.append(
            f"\n=============================== {BEAM_DIR}{beam} Results ({plane}-plane) =======================================\n\n{table_str}\n"
        )

    if kmod_averaged_output_dir is not None:
        grouped_kmod_averaged = _collect_averaged_kmod_results(
            beam=beam, output_dir=kmod_averaged_output_dir
        )
        if grouped_kmod_averaged:
            kmod_summary_averaged = tfs.concat(grouped_kmod_averaged, ignore_index=True)
            kmod_summary_x_averaged = kmod_summary_averaged[[IP_COLUMN] + COLS_X]
            kmod_summary_y_averaged = kmod_summary_averaged[[IP_COLUMN] + COLS_Y]
            for plane, df_averaged in [("X", kmod_summary_x_averaged), ("Y", kmod_summary_y_averaged)]:
                table_str_averaged = df_averaged.to_string()
                logbook_tables.append(
                    f"\n=========================== {BEAM_DIR}{beam} Averaged Results ({plane}-plane) ==================================\n\n{table_str_averaged}\n"
                )
    else:
        LOG.info("Averaged kmod results not included in the text table.")

    return kmod_summary, logbook_tables


def _save_outputs(
    beam: int,
    df: tfs.TfsDataFrame,
    logbook_text: str,
    output_dir: Path | str,
) -> None:
    """
    Save logbook text output and .tfs summary for a given beam.

    Args:
        beam (int): Beam number to process.
        save_output_dir (Path | str ): Path to the saving output directory.
        txt_to_save (str): .txt to be saved.
        df_to_save (tfs.TfsDataFrame): .tfs dataframe to be saved.

    """

    save_output_dir = Path(save_output_dir)
    logbook_table_path = save_output_dir / f"{BEAM_DIR}{beam}_kmod_summary.txt"
    summary_path = save_output_dir / f"{BEAM_DIR}{beam}_kmod_summary{EXT}"
    LOG.info(f"Writing .txt summary output file {logbook_table_path}.")
    logbook_table_path.write_text(txt_to_save)
    LOG.info(f"Writing {EXT} summary output file {summary_path}.")
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
