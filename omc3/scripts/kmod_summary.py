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
from omc3.utils.iotools import PathOrStr, save_config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tfs import TfsDataFrame

LOGGER = logging_tools.get_logger(__name__)

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
    opt.output_dir = Path(opt.output_dir)
    if not opt.output_dir.exists():
        opt.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(opt.output_dir, opt, __file__)

    # Generate a summary dataframe and the various text summaries
    # (there is one per beam per IP + lumi imbalance)
    df, summaries = gather_results_and_summaries(
        beam=opt.beam,
        meas_paths=opt.meas_paths,
        kmod_averaged_dir=opt.kmod_averaged_output_dir,
        lumi_imbalance_dir=opt.lumi_imb_output_dir,
    )

    # Join all these summaries and export to disk
    logbook_entry = "\n".join(filter(None, summaries))
    save_summary(beam=opt.beam, df=df, summary=logbook_entry, output_dir=opt.output_dir)

    # Potentially send this to logbook as well
    if opt.logbook is not None:
        logbook_file = opt.output_dir / f"{BEAM_DIR}{opt.beam}_{KMOD_FILENAME}.txt"
        post_summary_to_logbook(
            beam=opt.beam,
            logbook_name=opt.logbook,
            entry=logbook_entry,
            attachment=logbook_file,
        )

    return df


# ----- Separate Functionality ----- #


def gather_results_and_summaries(
    beam: int,
    meas_paths: Sequence[Path | str],
    kmod_averaged_dir: Path | str | None = None,
    lumi_imbalance_dir: Path | str | None = None,
) -> tuple[TfsDataFrame, str]:
    """
    Prepare K-modulation summary dataframe as well as formatted logbook tables.
    There is a single complete summary dataframe, and one summary text per scenario
    (per beam, including averaged etc).


    Args:
        beam (int): Beam number to process.
        meas_paths (Sequence[Path|str]): Directories of imported K-modulation results
            containing beam subfolders.
        kmod_averaged_dir (Path | str): Path to the directory containing averaged K-modulation
            TFS results files. Optional.
        lumi_imbalance_dir (Path | str): Path to the output directory containing luminosity
            imbalance TFS results files (effective betas files). Optional.

    Returns:
        Tuple[tfs.TfsDataFrame, list[str]]:
            - Dataframe containing K-modulation summary.
            - List of formatted text tables containing K-modulation (intermediate) summaries.
    """
    LOGGER.debug("Gathering Kmod results and generating summaries.")
    summaries: list[str] = []

    # ----- Gathering and summaries for kmod results ----- #
    kmod_results: list[TfsDataFrame] = collect_kmod_results(beam=beam, meas_paths=meas_paths)
    if kmod_results:
        kmod_summary: TfsDataFrame = tfs.concat(kmod_results, ignore_index=True)
    else:
        LOGGER.warning(f"No K-mod results found for beam {beam}.")
        kmod_summary = tfs.TfsDataFrame(columns=[IP_COLUMN, NAME] + COLS_X + COLS_Y)

    kmod_summary_x: TfsDataFrame = kmod_summary[[IP_COLUMN, NAME] + COLS_X]
    kmod_summary_y: TfsDataFrame = kmod_summary[[IP_COLUMN, NAME] + COLS_Y]

    # ----- Gathering summaries for lumi imbalance results ----- #
    if lumi_imbalance_dir is not None:
        kmod_summary_lumiimb = collect_lumi_imbalance_results(lumi_imbalance_dir=lumi_imbalance_dir)
        summaries.append(_format_summary("Luminosity Imbalance", kmod_summary_lumiimb))

    # ----- Adding K-mod summaries (after lumi imbalance if present) ----- #
    for plane, df in [("X", kmod_summary_x), ("Y", kmod_summary_y)]:
        summaries.append(_format_summary(f"{BEAM_DIR}{beam} Results ({plane}-plane)", df))

    # ----- Gethering and summaries for averaged kmod results ----- #
    if kmod_averaged_dir is not None:
        kmod_avg_results = collect_averaged_kmod_results(
            beam=beam, kmod_averaged_output_dir=kmod_averaged_dir
        )

        if len(kmod_avg_results):  # could return []
            kmod_summary_averaged: TfsDataFrame = tfs.concat(kmod_avg_results, ignore_index=True)
            kmod_averaged_x: TfsDataFrame = kmod_summary_averaged[[IP_COLUMN] + COLS_X]
            kmod_averaged_y: TfsDataFrame = kmod_summary_averaged[[IP_COLUMN] + COLS_Y]
            for plane, df_averaged in (("X", kmod_averaged_x), ("Y", kmod_averaged_y)):
                summaries.append(
                    _format_summary(
                        f"{BEAM_DIR}{beam} Averaged Results ({plane}-plane)", df_averaged
                    )
                )

    return kmod_summary, summaries


def collect_kmod_results(beam: int, meas_paths: Sequence[Path | str]) -> list[TfsDataFrame]:
    """
    Gathers the kmod results.tfs dataframes, taking only relevant column values for the given beam.

    Args:
        beam (int): Beam number to process.
        meas_paths (Sequence[Path | str]): Directories of imported K-modulation results containing beam subfolders.

    Returns:
        list[tfs.TfsDataFrame]: A list with all the gathered dataframes.
    """
    LOGGER.debug("Gathering kmod results.")
    result: list[TfsDataFrame] = []

    for dirpath in map(Path, meas_paths):
        LOGGER.info(f"Reading measurement results from '{dirpath.absolute()}' directory.")
        file_path = dirpath / f"{BEAM_DIR}{beam}" / f"{RESULTS_FILE_NAME}{EXT}"

        if not file_path.exists():
            LOGGER.warning(f"Missing results file: {file_path}")
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
    LOGGER.debug("Gathering averaged kmod results.")
    result: list[tfs.TfsDataFrame] = []

    if kmod_averaged_output_dir is None:
        LOGGER.info("No directory provided for averaged kmod, skipping.")
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
            LOGGER.warning(f"Beam {beam} not found in averaged results, skipping.")
            continue

        df = beam_row[COLS_X + COLS_Y]
        ip_name = beam_row[NAME].iloc[0]
        df.insert(0, IP_COLUMN, ip_name)
        result.append(df)

    return result


def collect_lumi_imbalance_results(lumi_imbalance_dir: Path | str | None) -> str:
    """
    Gathers the luminosity imbalance results from effective betas files.
    Returns a formatted multi-line summary string, one line per valid file.

    Args:
        lumi_imbalance_dir (Path | str | None): Path to the output directory containing
            luminosity imbalance TFS results files.

    Returns:
        str: Formatted table showing grouped luminosity imbalance results, one line per file.
    """
    LOGGER.debug("Gathering luminosity imbalance results.")
    report_lines: list[str] = []

    if lumi_imbalance_dir is None:
        LOGGER.info("No directory provided for lumi imbalance, skipping.")
        return ""

    lumi_imbalance_dir = Path(lumi_imbalance_dir)

    # The expected file name is based on the EFFECTIVE_BETAS_FILENAME
    # constant. We check the starting part, up to the IP number
    prefix = EFFECTIVE_BETAS_FILENAME.split("{")[0]

    for df_path in lumi_imbalance_dir.glob(f"{prefix}*"):
        if not df_path.is_file():
            continue

        # Load dataframe and get lumi imbalance from headers
        df = tfs.read(df_path)
        lumi_imbalance: float | None = df.headers.get("LUMIIMBALANCE", None)
        err_lumi_imbalance: float | None = df.headers.get("ERRLUMIIMBALANCE", None)

        # Get the IPs. There should be two in this file and the lumi
        # (error) from the headers imbalance is from first to second IP
        ips: list[str] = df[NAME].tolist()
        if lumi_imbalance is not None and err_lumi_imbalance is not None and len(ips) >= 2:
            report_lines.append(
                f"Luminosity imbalance between {ips[0]} and {ips[1]}: {lumi_imbalance} ± {err_lumi_imbalance}"
            )

    return "\n".join(report_lines)


def save_summary(beam: int, df: TfsDataFrame, summary: str, output_dir: Path | str) -> None:
    """
    Save logbook text output and .tfs summary for a given beam.

    Args:
        beam (int): Beam number to process.
        df (tfs.TfsDataFrame): A kmod summary TfsDataFrame to save to disk.
        summary (str): The summary as a string, as potentially sent to the logbook,
            to be saved to a .txt file.
        output_dir (Path | str): Path to the directory in which to save both dataframe
            and logbook text.
    """
    save_output_dir = Path(output_dir)
    logbook_table_path = save_output_dir / f"{BEAM_DIR}{beam}_{KMOD_FILENAME}.txt"
    summary_path = save_output_dir / f"{BEAM_DIR}{beam}_{KMOD_FILENAME}{EXT}"

    LOGGER.debug(f"Writing .txt summary output file {logbook_table_path}.")
    logbook_table_path.write_text(summary)

    LOGGER.debug(f"Writing {EXT} summary output file {summary_path}.")
    tfs.write(summary_path, df)


def post_summary_to_logbook(
    beam: int, logbook_name: str, entry: str, attachment: str | Path | None = None
) -> None:
    """
    Create logbook entry with summary for a given beam, in the provided
    logbook. Potentially attach a file if provided.

    Args:
        beam (int): Beam the summary was processed for.
        logbook_name (str): Logbook to post to, e.g. 'LHC_OMC'.
        entry (str): Text for logbook entry.
        attachment (str | Path): File to attach at the logbook entry. Optional.
    """
    logbook_filename = f"{BEAM_DIR}{beam}_kmod_summary"
    LOGGER.info(f"Creating logbook entry for {logbook_filename} to {logbook_name}.")

    logbook_event = DotDict(
        {
            "text": entry,
            "logbook": logbook_name,
        }
    )

    if attachment is not None:
        logbook_event["files"] = [attachment]

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
    LOGGER.debug("Extracting IP name from dataframe")

    try:
        # takes magnet names from label, e.g. MQXA1.L5-MQXA1.R5
        magnets_label = result_df[LABEL].iloc[0]
    except KeyError as exc:
        LOGGER.warning(f"Missing '{LABEL}' column, cannot extract IP.", exc_info=exc)
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
        LOGGER.warning(f"Malformed magnets label value: {magnets_label}", exc_info=exc)
        return None


def _format_summary(title: str, content: TfsDataFrame | str) -> str:
    """
    Format a summary text from the dataframe's data prefixed with a
    header including the title, based on the dataframe's text width.

    Args:
        title (str): The header title.
        content (tfs.TfsDataFrame | str): The summary data to format, which
            can be a TfsDataFrame with compiled results or a directly a
            text with summary info (in the case of lumi imbalance).

    Returns:
        str: Formatted section text with included header title.
    """
    text = content if isinstance(content, str) else content.to_string(index=False)

    if not text.strip():
        LOGGER.warning("No summary data results found, skipping")
        return ""

    width = len(text.splitlines()[0])
    header = f" {title} ".center(width, "=")
    return f"{header}\n{text}\n"


# ----- Commandline Entry Point ----- #

if __name__ == "__main__":
    generate_kmod_summary()
