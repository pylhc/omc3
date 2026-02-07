from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from generic_parser import DotDict
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint

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
from omc3.utils.iotools import PathOrStr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tfs import TfsDataFrame

LOG = logging_tools.get_logger(__name__)

# Constants definitions for K-modulation
COLS_X: list[str] = [
    f"{BETASTAR}X",
    f"{ERR}{BETASTAR}X",
    f"{BETAWAIST}X",
    f"{ERR}{BETAWAIST}X",
    f"{WAIST}X",
    f"{ERR}{WAIST}X",
]
COLS_Y: list[str] = [
    f"{BETASTAR}Y",
    f"{ERR}{BETASTAR}Y",
    f"{BETAWAIST}Y",
    f"{ERR}{BETAWAIST}Y",
    f"{WAIST}Y",
    f"{ERR}{WAIST}Y",
]
IP_COLUMN: str = "IP"
KMOD_FILENAME: str = "kmod_summary"

# ----- Script Mode ----- #


def _get_params() -> EntryPointParameters:
    """
    Creates and returns the parameters for the Kmodulation summary functionality.
    """
    params = EntryPointParameters()
    params.add_parameter(
        name="beam",
        type=int,
        help="Beam number to process.",
    )
    params.add_parameter(
        name="meas_paths",
        type=PathOrStr,
        nargs="+",
        help="Paths to K-modulation measurement directories.",
    )
    params.add_parameter(
        name="kmod_averaged_output_dir",
        type=PathOrStr,
        default=None,
        help="Directory containing averaged K-modulation results.",
    )
    params.add_parameter(
        name="lumi_imb_output_dir",
        type=PathOrStr,
        default=None,
        help="Directory containing luminosity imbalance results.",
    )
    params.add_parameter(
        name="output_dir",
        type=PathOrStr,
        help="Directory to write summary files in.",
    )
    params.add_parameter(
        name="logbook",
        type=str,
        default=None,
        help="Logbook name (e.g. LHC_OMC) to publish the summary to.",
    )
    return params


@entrypoint(_get_params(), strict=True)
def generate_kmod_summary(opt: DotDict) -> TfsDataFrame:
    """
    Reads kmod results, potentially including averaged results and
    luminosity imbalance, and generates a summary dataframe then
    saves the dataframe and a text summary to disk. If provided with
    a logbook name, post that summary to the logbook.

    Args:
        beam (int):
            Beam number to process results for.
        meas_paths (Sequence[Path|str]):
            Directories of imported K-modulation results.
        kmod_averaged_output_dir (PathOrStr):
            Directory containing averaged K-modulation results. Optional.
        lumi_imb_output_dir (PathOrStr):
            Directory containing luminosity imbalance results. Optional.
        output_dir (Path|str):
            Directory to write summary files in.
        logbook (str):
            Logbook name to publish the summary to. Optional.

    Returns:
        A TfsDataFrame with all gathered results summarized.
    """
    # Create the output directory - do not overwrite if called
    # by kmod_imported and provided exisiting kmod results dir
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    # Generate a summary dataframe and the various text summaries
    # (there is one per beam per IP + lumi imbalance)
    df, tables = gather_results_and_summaries(
        beam=opt.beam,
        meas_paths=opt.meas_paths,
        kmod_averaged_output_dir=opt.kmod_averaged_output_dir,
        lumi_imb_output_dir=opt.lumi_imb_output_dir,
    )

    # Join all these summaries and export to disk
    logbook_entry = "\n".join(filter(None, tables))
    save_summary(beam=opt.beam, df=df, summary=logbook_entry, output_dir=output_dir)

    # Potentially send this to logbook as well
    if opt.logbook is not None:
        logbook_file = output_dir / f"{BEAM_DIR}{opt.beam}_{KMOD_FILENAME}.txt"
        post_summary_to_logbook(
            beam=opt.beam,
            logbook=opt.logbook,
            logbook_entry_text=logbook_entry,
            logbook_entry_file=logbook_file,
        )

    return df


# ----- Separate Functionality ----- #


def gather_results_and_summaries(
    beam: int,
    meas_paths: Sequence[Path | str],
    kmod_averaged_output_dir: Path | str | None = None,
    lumi_imb_output_dir: Path | str | None = None,
) -> tuple[TfsDataFrame, str]:
    """
    Prepare K-modulation summary dataframe as well as formatted logbook tables.
    There is a single complete summary dataframe, and one summary text per scenario
    (per beam, including averaged etc).


    Args:
        beam (int): Beam number to process.
        meas_paths (Sequence[Path|str]): Directories of imported K-modulation results containing beam subfolders.
        kmod_averaged_output_dir (Path | str): Path to the directory containing averaged kmod TFS results files. Optional.
        lumi_imb_output_dir (Path | str): Path to the output directory containing luminosity imbalance TFS results files. Optional.

    Returns:
        Tuple[tfs.TfsDataFrame, list[str]]:
            - Dataframe containing K-modulation summary.
            - List of formatted text tables containing K-modulation (intermediate) summaries.
    """
    LOG.info("Formatting the tables to text.")

    grouped_kmod = collect_kmod_results(beam=beam, meas_paths=meas_paths)
    if grouped_kmod:
        kmod_summary = tfs.concat(grouped_kmod, ignore_index=True)
    else:
        LOG.warning(f"No K-mod results found for beam {beam}, skipping.")
        kmod_summary = tfs.TfsDataFrame(columns=[IP_COLUMN, NAME] + COLS_X + COLS_Y)
    kmod_summary_x = kmod_summary[[IP_COLUMN, NAME] + COLS_X]
    kmod_summary_y = kmod_summary[[IP_COLUMN, NAME] + COLS_Y]

    logbook_table: list[str] = []
    if lumi_imb_output_dir is not None:
        kmod_summary_lumiimb = collect_lumi_imbalance_results(output_dir=lumi_imb_output_dir)
        logbook_table.append(_format_summary("Luminosity Imbalance", kmod_summary_lumiimb))
    else:
        LOG.info("Luminosity imbalance results not included in the text table.")

    for plane, df in [("X", kmod_summary_x), ("Y", kmod_summary_y)]:
        logbook_table.append(_format_summary(f"{BEAM_DIR}{beam} Results ({plane}-plane)", df))

    if kmod_averaged_output_dir is not None:
        grouped_kmod_averaged = collect_averaged_kmod_results(
            beam=beam, kmod_averaged_output_dir=kmod_averaged_output_dir
        )

        if grouped_kmod_averaged:
            kmod_summary_averaged = tfs.concat(grouped_kmod_averaged, ignore_index=True)
            kmod_summary_x_averaged = kmod_summary_averaged[[IP_COLUMN] + COLS_X]
            kmod_summary_y_averaged = kmod_summary_averaged[[IP_COLUMN] + COLS_Y]
            for plane, df_averaged in (
                ("X", kmod_summary_x_averaged),
                ("Y", kmod_summary_y_averaged),
            ):
                logbook_table.append(
                    _format_summary(
                        f"{BEAM_DIR}{beam} Averaged Results ({plane}-plane)", df_averaged
                    )
                )
    else:
        LOG.info("Averaged kmod results not included in the text table.")

    return kmod_summary, logbook_table


def collect_kmod_results(beam: int, meas_paths: Sequence[Path | str]) -> list[TfsDataFrame]:
    """
    Gathers the kmod results.tfs dataframes, taking only relevant column values for the given beam.

    Args:
        beam (int): Beam number to process.
        meas_paths (Sequence[Path | str]): Directories of imported K-modulation results containing beam subfolders.

    Returns:
        list[tfs.TfsDataFrame]: A list with all the gathered dataframes.
    """
    LOG.info("Gathering kmod results.")
    result: list[TfsDataFrame] = []

    for dirpath in map(Path, meas_paths):
        LOG.info(f"Reading measurement results from '{dirpath.absolute()}' directory.")
        file_path = dirpath / f"{BEAM_DIR}{beam}" / f"{RESULTS_FILE_NAME}{EXT}"

        if not file_path.exists():
            LOG.warning(f"Missing results file: {file_path}")
            continue

        meas_name = dirpath.name
        kmod_df = tfs.read(file_path)
        ip_name = _extract_ip_name(kmod_df)

        df = kmod_df[COLS_X + COLS_Y].iloc[[0]]  # returns a DataFrame with one row
        df.insert(0, NAME, meas_name)
        df.insert(0, IP_COLUMN, ip_name)
        result.append(df)

    return result


def collect_averaged_kmod_results(
    beam: int, kmod_averaged_output_dir: Path | str | None
) -> list[TfsDataFrame]:
    """
    Gathers the averaged kmod results dataframes, taking only relevant column values for the given beam.

    Args:
        beam (int): Beam number to process.
        kmod_averaged_output_dir (Path | str | None): Path to the directory containing averaged kmod TFS results files.

    Returns:
        list[tfs.TfsDataFrame]: A list with all the gathered dataframes. Empty if no path was provided.
    """
    LOG.info("Gathering averaged kmod results.")
    result: list[tfs.TfsDataFrame] = []

    if kmod_averaged_output_dir is None:
        LOG.info("No output_dir provided, skipping.")
        return result

    kmod_averaged_output_dir = Path(kmod_averaged_output_dir)

    # The expected file name is based on the AVERAGED_BETASTAR_FILENAME
    # constant. We check the starting part, up to the IP number
    prefix = AVERAGED_BETASTAR_FILENAME.split("{")[0]

    for df_path in kmod_averaged_output_dir.glob(f"{prefix}*"):
        if not df_path.is_file():
            continue

        avg_df = tfs.read(df_path)
        beam_row = avg_df[avg_df["BEAM"] == beam]

        if beam_row.empty:
            LOG.warning(f"Beam {beam} not found in averaged results, skipping.")
            continue

        df = beam_row[COLS_X + COLS_Y]
        ip_name = beam_row[NAME].iloc[0]
        df.insert(0, IP_COLUMN, ip_name)
        result.append(df)

    return result


def collect_lumi_imbalance_results(output_dir: Path | str | None) -> str:
    """
    Gathers the luminosity imbalance results from effective betas files.
    Returns a formatted multi-line string, one line per valid file.

    Args:
        output_dir (Path | str ): Path to the folder with luminosity imbalance dataframes. If None, luminosity imbalance results are not collected.
    Returns:
        str: Formatted table showing grouped luminosity imbalance results, one line per file.
    """
    LOG.info("Gathering luminosity imbalance results.")

    if output_dir is None:
        LOG.info("No output_dir provided, skipping.")
        return ""

    output_dir = Path(output_dir)
    # if the file name starts with EFFECTIVE_BETAS_FILENAME till {ip}
    prefix = EFFECTIVE_BETAS_FILENAME.split("{")[0]
    report_lines: list[str] = []
    for df_path in output_dir.glob(f"{prefix}*"):
        if not df_path.is_file():
            continue
        df = tfs.read(df_path)
        lumi_imbalance = df.headers.get("LUMIIMBALANCE")
        lumi_imbalance = float(lumi_imbalance) if lumi_imbalance is not None else None
        err_lumi_imbalance = df.headers.get("ERRLUMIIMBALANCE")
        err_lumi_imbalance = float(err_lumi_imbalance) if err_lumi_imbalance is not None else None
        ips = df[NAME].tolist()
        # final check
        if lumi_imbalance is not None and err_lumi_imbalance is not None and len(ips) >= 2:
            report_lines.append(
                f"Luminosity imbalance in between {ips[0]} and {ips[1]} is {lumi_imbalance} ± {err_lumi_imbalance}"
            )
    return "\n".join(report_lines)


def save_summary(
    beam: int,
    df: TfsDataFrame,
    summary: str,
    output_dir: Path | str,
) -> None:
    """
    Save logbook text output and .tfs summary for a given beam.

    Args:
        beam (int): Beam number to process.
        df (tfs.TfsDataFrame): A kmod summary TfsDataFrame to save to disk.
        summary (str): The summary as a string, as potentially sent to the logbook,
            to be saved to a .txt file.
        output_dir (Path | str): Path to the directory in which to save both dataframe and logbook text.
    """
    save_output_dir = Path(output_dir)
    logbook_table_path = save_output_dir / f"{BEAM_DIR}{beam}_{KMOD_FILENAME}.txt"
    summary_path = save_output_dir / f"{BEAM_DIR}{beam}_{KMOD_FILENAME}{EXT}"
    LOG.info(f"Writing .txt summary output file {logbook_table_path}.")
    logbook_table_path.write_text(summary)
    LOG.info(f"Writing {EXT} summary output file {summary_path}.")
    tfs.write(summary_path, df)


def post_summary_to_logbook(
    beam: int,
    logbook: str,
    logbook_entry_text: str,
    logbook_entry_file: str | Path | None = None,
) -> None:
    """
    Create logbook entry with .txt generated table for a given beam.

    Args:
        beam (int): Beam number to process.
        logbook (str): logbook name, ex. 'LHC_OMC'
        logbook_entry_text (str): Text for logbook entry.
        logbook_entry_file (str | Path): File to attach at the logbook entry. Defaults to None.
    """
    logbook_filename = f"{BEAM_DIR}{beam}_kmod_summary"
    logbook_event = DotDict(
        {
            "text": logbook_entry_text,
            "logbook": logbook,
        }
    )

    if logbook_entry_file is not None:
        logbook_event["files"] = [logbook_entry_file]

    LOG.info(f"Creating logbook entry for {logbook_filename} to {logbook}.")
    _ = create_logbook_entry(logbook_event)


# ----- Helpers ----- #


def _extract_ip_name(result_df: TfsDataFrame) -> str | None:
    """
    Extract IP name from magnet in LABEL column.

    Args:
        result_df (tfs.TfsDataFrame): TfsDataFrame containing the kmod measurement.

    Returns:
        str: IP name in the form 'IP{number}'.
    """
    LOG.debug("Extracting IP name from dataframe")

    try:
        # takes magnet names from label, e.g. MQXA1.L5-MQXA1.R5
        magnets_label = result_df[LABEL].iloc[0]
    except KeyError as exc:
        LOG.debug(f"Missing '{LABEL}' column, cannot extract IP.", exc_info=exc)
        return None

    try:
        # Take right magnet name from label e.g. "MQXA1.R5" from "MQXA1.L5-MQXA1.R5"
        second_magnet = magnets_label.split("-")[1]
        # Take right part of the magnet's name e.g. "R5" from "MQXA.R5"
        ip_and_side = second_magnet.split(".")[-1]
        # isolate IP number e.g. "5" from "R5"
        ip_number = ip_and_side[1:]
        return f"{IP_COLUMN}{ip_number}"
    except (IndexError, ValueError) as exc:
        LOG.debug(f"Malformed magnets label value: {magnets_label}", exc_info=exc)
        return None


def _format_summary(title: str, df: TfsDataFrame) -> str:
    """
    Format a summary text from the dataframe's data prefixed with a
    header including the title, based on the dataframe's text width.

    Args:
        title (str): The header title.
        df (tfs.TfsDataFrame): The dataframe to include.

    Returns:
        str: Formatted section text.
    """
    df_text = df.to_string(index=False)

    if not df_text.strip():
        LOG.warning("No summary data results found, skipping")
        return ""

    width = len(df_text.splitlines()[0])
    header = f" {title} ".center(width, "=")
    return f"{header}\n{df_text}\n"


# ----- Commandline Entry Point ----- #

if __name__ == "__main__":
    generate_kmod_summary()
