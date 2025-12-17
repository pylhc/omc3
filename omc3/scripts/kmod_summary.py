"""
Generate summary tables from K-modulation results.

This function collects and summarizes K-modulation results from multiple measurement directories.
For each beam, it generates:
1. A DataFrame containing all imported K-modulation measurement results.
2. Optionally, a text file with the averaged results which can be used to create a logbook entry.

**Arguments:**

*--Required--*

- **meas_paths** *(Sequence[Path | str])*:

    Directories of K-modulation results to import.
    These need to be the paths to the root-folders containing B1 and B2 sub-dirs.

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

from omc3.optics_measurements.constants import BEAM_DIR, BETASTAR, BETAWAIST, ERR, LABEL, WAIST
from omc3.scripts.create_logbook_entry import main
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
from generic_parser import DotDict

LOG = logging_tools.get_logger(__name__)

def output_kmod_summary_tables(
        meas_paths: Sequence[Path | str],
        averaged_meas: dict[str, dict[int, tfs.TfsDataFrame]] | None = None,
        output_dir: Path | str | None = None,
        logbook: str | None = None,
        ) -> tuple[dict[str, list[tfs.TfsDataFrame]],dict[str, list[list[str]]]]:

    """
    Args:
        meas_paths (Sequence[Path | str]): Paths to the K-modulation results.
        averaged_meas (dict[int, tfs.TfsDataFrame]): If not None, averaged K-modulation results over all measurements are included in the summary tables. Default: None.
        output_dir (Path | str | None): Path to the output directory. Defaults to None.
        logbook (str = None): If provided, create a logbook entry containing the .txt summary tables to the given logbook. Default: None.

    Returns:
        Tuple[dict[str, tfs.TfsDataFrame], dict[str, list[str]]]:
            - Dictionary mapping beam names to K-modulation summary DataFrames.
            - Dictionary mapping beam names to lists of formatted text tables
    """
    LOG.info("Starting kmod summary importing.")
    cols_x = [f"{BETASTAR}X", f"{ERR}{BETASTAR}X", f"{BETAWAIST}X", f"{ERR}{BETAWAIST}X", f"{WAIST}X", f"{ERR}{WAIST}X"]
    cols_y = [f"{BETASTAR}Y", f"{ERR}{BETASTAR}Y", f"{BETAWAIST}Y", f"{ERR}{BETAWAIST}Y", f"{WAIST}Y", f"{ERR}{WAIST}Y"]
    beams = [f"{BEAM_DIR}{1}", f"{BEAM_DIR}{2}"]

    grouped = {beam: [] for beam in beams}
    """
    Gathers the various Kmod results.tfs dataframes, taking only cols_x and cols_y values.
    Then it stores them in grouped, divided by beam.
    """
    for path in meas_paths:
        path = Path(path)
        LOG.info(f"Reading results: {path}.")
        for beam in beams:
            file_path = path / beam / "results.tfs"
            if not file_path.exists():
                LOG.warning(f"Missing results.tfs: {file_path}")
            else:
                file_name = file_path.parent.parent.name
                data_file = tfs.read(file_path)
                try:
                    label = data_file[f"{LABEL}"].iloc[0]
                except KeyError as e:
                    LOG.debug("Could not parse IP from LABEL. Skipping entry.", exc_info=e)
                    ip_name = "None"
                else:
                    second_magnet = label.split("-")[1]
                    relevant = second_magnet.split(".")[-1]
                    ip_number = relevant[1:]
                    ip_name = f"IP{ip_number}"
                df = data_file[cols_x + cols_y].iloc[[0]]
                df.insert(0, "NAME", file_name)
                df.insert(0, "IP", ip_name)
                grouped[beam].append(df)

    grouped_averaged = {beam: [] for beam in beams}
    if averaged_meas is not None:
        """
        Gathers the various averaged Kmod results dataframes, taking only cols_x and cols_y values.
        Then it stores them in grouped_averaged, divided by beam.
        """
        for beam_num in [1,2]:
            for av_ip, av_res in averaged_meas.items():
                LOG.info(f"Reading averaged results: {BEAM_DIR}{beam_num}, {av_ip}")
                av_tab = av_res[0] # 0: averaged results table, 1, 2: are betx, bety for IP closest elements
                av_tab_beam = av_tab.loc[[beam_num]]
                df_averaged = av_tab_beam[cols_x + cols_y]
                df_averaged.insert(0, "IP", av_ip)
                grouped_averaged[f"{BEAM_DIR}{beam_num}"].append(df_averaged)

    kmod_summaries = {beam: [] for beam in beams}
    tables_logbook = {beam: [] for beam in beams}
    """Create the Kmod summary .tfs table and the .txt file for the logbook."""
    for beam in beams:
        LOG.debug(f"Processing result for: {beam}")
        if grouped[beam]:
            kmod_summary = tfs.concat(grouped[beam], ignore_index=True)
            kmod_summary_x = kmod_summary[["IP", "NAME"] + cols_x]
            kmod_summary_y = kmod_summary[["IP", "NAME"] + cols_y]
            if averaged_meas:
                kmod_summary_averaged = tfs.concat(grouped_averaged[beam], ignore_index=True)
                kmod_summary_x_averaged = kmod_summary_averaged[["IP"] + cols_x]
                kmod_summary_y_averaged = kmod_summary_averaged[["IP"] + cols_y]
            table_logbook = []
            for plane, df in [("X", kmod_summary_x), ("Y", kmod_summary_y)]:
                table_str = df.to_string()
                table_logbook.append(f"\n============================================ {beam} Results ({plane}-plane) ====================================================\n\n{table_str}\n")
            if averaged_meas:
                for plane, df_averaged in [("X", kmod_summary_x_averaged), ("Y", kmod_summary_y_averaged)]:
                    table_str_averaged = df_averaged.to_string()
                    table_logbook.append(f"\n======================================== {beam} Averaged Results ({plane}-plane) ===============================================\n\n{table_str_averaged}\n")
            kmod_summaries[beam] = kmod_summary
            tables_logbook[beam] = table_logbook

    if output_dir is not None:
        for beam in beams:
            _save_outputs(beam=beam, save_output_dir=output_dir, table_logbook=tables_logbook[beam], summary_df=kmod_summaries[beam])
            if logbook:
                _logbook_entry(logbook=logbook, beam=beam, save_output_dir=output_dir)
        if not logbook:
            LOG.info("Logbook name not provided: logbook entry not created.")
    else:
        LOG.info("Output_dir not provided: skipping file output for all beams.")

    return kmod_summaries, tables_logbook

def _save_outputs(
    beam: str,
    table_logbook: list[str],
    summary_df: tfs.TfsDataFrame,
    save_output_dir: Path | str,
    ) -> None:

    """
    Save logbook text output and .tfs summary for a given beam.

    Args:
        beam (str): beam, ex. B1, B2
        table_logbook (list[str]): Single .txt file.
        summary_df (tfs.TfsDataFrame): Single .tfs dataframe.
        save_output_dir (Path | str ): Path to the saving output directory.
    """

    logbook_table_path = save_output_dir / f"{beam}_tables_logbook.txt"
    summary_path = save_output_dir / f"{beam}_kmod_summary.tfs"
    with logbook_table_path.open("w") as f:
        f.write("\n".join(table_logbook))
    tfs.write(summary_path, summary_df)
    LOG.info(f"Writing logbook summary output file {logbook_table_path}.")
    LOG.info(f"Writing .tfs summary output file {summary_path}.")
    return

def _logbook_entry(
    logbook: str,
    beam: str,
    save_output_dir: Path | str,
    ) -> None:

    """
    Create logbook entry with .txt generated table for a given beam.
    Args:
        logbook (str): logbook name, ex. 'LHC_OMC'
        beam (str): beam. ex. B1, B2
        save_output_dir (Path | str ): Path to the saving output directory.
    """

    logbook_file = save_output_dir / f"{beam}_tables_logbook.txt"
    logbook_filename = f"{beam}_kmod_summary.txt"
    with Path.open(logbook_file, "r") as f:
        logbook_text = f.read()
    logbook_event = DotDict({
            "text": logbook_text,
            "files": [logbook_file],
            "filenames": [logbook_filename],
            "logbook": f"{logbook}",
        })
    _ = main(logbook_event)
    LOG.info(f"Creating logbook entry for {logbook_filename} to {logbook}.")
    return

# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    output_kmod_summary_tables()
