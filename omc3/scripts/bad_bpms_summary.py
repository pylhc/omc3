"""
Bad BPMs Summary
----------------

Scans all measurements in a list of given GUI output folders and compiles a list of bad BPMs with
their given number of appearances after 'harpy' and 'isolation forest'.



.. admonition:: Usage

  Get bad BPMs for LHC-Beam 1 from September 2024 and 2024-10-03

  .. code-block:: none 

    python -m omc3.scripts.bad_bpms_summary --dates 2024-09-* 2024-10-03 --accel_glob LHCB1 --outfile bad_bpms_sep_2024.txt  --print_percentage 50 



*--Required--*

- **dates** *(str)*:

    Dates to include in analysis. This should be either subfolders in
    `root` or glob-patterns for those.


*--Optional--*

- **accel_glob** *(str)*:

    Accelerator name (glob for the sub-directories).

    default: ``LHCB*``


- **outfile** *(PathOrStr)*:

    Path to the file to write out.


- **print_percentage** *(float)*:

    Print out BPMs that appear in more than this percentage of
    measurements.


- **root** *(PathOrStr)*:

    Path to the root directory, containing the dates.

    default: ``/user/slops/data/LHC_DATA/OP_DATA/Betabeat``


"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs
from generic_parser import EntryPointParameters, entrypoint

from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, OptionalFloat

if TYPE_CHECKING:
    from collections.abc import Sequence
    from generic_parser import DotDict

LOG = logging_tools.get_logger(__name__)

# Constants ---
ROOT = Path("/user/slops/data/LHC_DATA/OP_DATA/Betabeat")
IFOREST = "IFOREST"
HARPY = "HARPY"

# Columns ---
NAME = "NAME"
ACCEL = "ACCELERATOR"
PLANE = "PLANE"
SOURCE = "SOURCE"
REASON = "REASON"
COUNT = "COUNT"
FILE = "FILE"
FILE_COUNT = "FILE_COUNT"
PERCENTAGE = "PERCENTAGE"

# Files ---
MEASUREMENTS_DIR = "Measurements"
RESULTS_DIR = "Results"
BAD_BPMS_HARPY = "*.bad_bpms_*"
BAD_BPMS_IFOREST = "bad_bpms_iforest_*.tfs"


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="dates",
        type=str,
        nargs="+",
        required=True,
        help=("Dates to include in analysis. "
        "This should be either subfolders in `root` or glob-patterns for those."
        )
    )
    params.add_parameter(
        name="root",
        type=PathOrStr,
        default=ROOT,
        help="Path to the root directory, containing the dates."
    )
    params.add_parameter(
        name="outfile",
        type=PathOrStr,
        help="Path to the file to write out." 
    )
    params.add_parameter(
        name="print_percentage",
        type=OptionalFloat,
        help="Print out BPMs that appear in more than this percentage of measurements." 
    )
    params.add_parameter(
        name="accel_glob",
        type=str,
        default="LHCB*",
        help="Accelerator name (glob for the sub-directories)."
    )
    return params


@entrypoint(get_params(), strict=True)
def bad_bpms_summary(opt: DotDict) -> tfs.TfsDataFrame:
    outfile = None
    if opt.outfile is not None:
        outfile = Path(opt.outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
    
    df_collection = collect_bad_bpms(Path(opt.root), opt.dates, opt.accel_glob)
    if outfile is not None:
        tfs.write(outfile.with_stem(f"{outfile.stem}_collected"), df_collection)
    
    df_evaluated = evaluate(df_collection)
    if outfile is not None:
        tfs.write(outfile, df_evaluated)

    if opt.print_percentage is not None:
        print_results(df_evaluated, opt.print_percentage)

    return df_evaluated


# Collection of Data ---

def get_empty_df() -> tfs.TfsDataFrame:
    """ Create an empty TfsDataFrame with the correct column names. """
    return tfs.TfsDataFrame(columns=[NAME, ACCEL, PLANE, SOURCE, FILE])


def collect_bad_bpms(root: Path, dates: Sequence[Path | str], accel_glob: str) -> tfs.TfsDataFrame:
    """ Create a TfsDataFrame with all bad-bpms within selected dates.

    Args:
        root (Path): Root path to the GUI output folder.
        dates (Sequence[Path | str]): List of dates or glob patterns to collect bad-bpms from.
        accel_glob (str): Accelerator name (glob for the sub-directories).

    Returns:
        tfs.TfsDataFrame: TfsDataFrame with all bad-bpms within selected dates.
    
    """
    dfs = []

    def collect_and_append(date_dir: Path):
        """ Helper to collect for date_dir and append to dfs if not None. """
        df_new = collect_date(date_dir, accel_glob)
        if df_new is not None:
            dfs.append(df_new)

    # Loop over dates ---
    for date in dates:
        date_dir = root / date
        if date_dir.is_dir():
            collect_and_append(date_dir)

        else:  
            for date_dir in root.glob(date):
                collect_and_append(date_dir)
    
    # Check and return ---
    if not len(dfs):
        LOG.warning("No bad-bpms found! Resulting TfsDataFrame will be empty.")
        return get_empty_df()

    return tfs.concat(dfs, axis="index", ignore_index=True) 


def collect_date(date_dir: Path, accel_glob: str) -> tfs.TfsDataFrame | None:
    """ Collect bad-bpms for a single date, by checking the sub-directories 
    which conform to the `accel_glob` pattern.

    In each accel directory, check for sub-directories named `Measurements` and `Results`,
    which in turn contain which in turn have entries containing the bad-bpms files.
    
    Args:
        date_dir (Path): Path to the date directory.
        accel_glob (str): Accelerator name (glob for the sub-directories).

    Returns:
        tfs.TfsDataFrame: TfsDataFrame with all bad-bpms for the date.
    """
    dfs: list[tfs.TfsDataFrame] = []

    for accel_dir in date_dir.glob(accel_glob):
        for subdir_name in (MEASUREMENTS_DIR, RESULTS_DIR):
            analysis_stage_dir = accel_dir / subdir_name 
            if not analysis_stage_dir.is_dir():
                continue

            for data_dir in analysis_stage_dir.iterdir():
                if not data_dir.is_dir():
                    continue

                df_collected = collect_bad_bpm_files_in_dir(data_dir)
                if df_collected is not None:
                    df_collected.loc[:, ACCEL] = accel_dir.name
                    dfs.append(df_collected)

    if not len(dfs):
        return None

    return tfs.concat(dfs, axis="index", ignore_index=True) 
    

def collect_bad_bpm_files_in_dir(directory: Path) -> tfs.TfsDataFrame | None:
    """ Collect bad-bpms for a single measurement directory.
    
    Args:
        directory (Path): Path to the directory possibly containing bad-bpm files of type `file_types`.

    Returns:
        tfs.TfsDataFrame: TfsDataFrame with all bad-bpms from the given directory.
    
    """
    readers_map = {
        BAD_BPMS_HARPY: read_harpy_bad_bpms_file,
        BAD_BPMS_IFOREST: read_iforest_bad_bpms_file
    }

    dfs: list[tfs.TfsDataFrame] = []

    for glob_pattern, reader in readers_map.items():
        for bad_bpms_file in directory.glob(glob_pattern):
            new_df = reader(bad_bpms_file)
            if new_df is not None:
                dfs.append(new_df)

    if not len(dfs):
        return None
    
    return tfs.concat(dfs, axis="index", ignore_index=True)


# File Readers --
            
def read_harpy_bad_bpms_file(svd_file: Path) -> tfs.TfsDataFrame:
    """ Reads a harpy bad-bpm file and returns a TfsDataFrame with all unique bad-bpms.
    
    Args:
        svd_file (Path): Path to the bad-bpm file.

    Returns:
        tfs.TfsDataFrame: TfsDataFrame with all unique bad-bpms.

    """
    TO_IGNORE = ("not found in model",)
    TO_MARK = ("known bad bpm",)
    COMMENT = "#"

    plane = svd_file.name[-1]

    # Read and parse file 
    lines = svd_file.read_text().splitlines()
    lines = [line.strip().split(maxsplit=1) for line in lines]
    lines = [(line[0].strip(), line[1].lower().strip()) for line in lines]

    lines = [line for line in lines if not line[0].startswith(COMMENT) and line[1] not in TO_IGNORE]
    bpms = set(f"[{line[0]}]" if line[1] in TO_MARK else line[0] for line in lines)

    # Create DataFrame    
    df = get_empty_df()
    df.loc[:, NAME] = list(bpms)
    df.loc[:, PLANE] = plane.upper()
    df.loc[:, SOURCE] = HARPY
    df.loc[:, FILE] = str(svd_file)
    return df


def read_iforest_bad_bpms_file(iforest_file: Path) -> tfs.TfsDataFrame:
    """ Reads an iforest bad-bpm file and returns a TfsDataFrame with all unique bad-bpms.
    
    Args:
        iforest_file (Path): Path to the bad-bpm file.

    Returns:
        tfs.TfsDataFrame: TfsDataFrame with all unique bad-bpms.

    """
    df_iforest = tfs.read(iforest_file)
    plane = iforest_file.stem[-1]

    df = get_empty_df()
    df.loc[:, NAME] = list(set(df_iforest[NAME]))  # hint: be sure to ignore index
    df.loc[:, PLANE] = plane.upper()
    df.loc[:, SOURCE] = IFOREST 
    df.loc[:, FILE] = str(iforest_file)
    return df


# Evaluaion ----


def evaluate(df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """ Evaluates the gathered bad-bpms and returns a TfsDataFrame with the results.

    The evaluation is based on the following criteria:
    - Count how often a BPM is bad
    - Count the total number of (unique) files for each combination of accelerator, source and plane

    From this information the percentage of how often a BPM is deemed bad is calculated.
    
    Args:
        df (tfs.TfsDataFrame): TfsDataFrame with all bad-bpms.

    Returns:
        tfs.TfsDataFrame: TfsDataFrame with the evaluated results.
    """
    # Count how often a BPM is bad
    df_counted = df.groupby([NAME, ACCEL, SOURCE, PLANE]).size().reset_index(name=COUNT)

    # Count the total number of (unique) files for each combination of accelerator, source and plane
    file_count = df.groupby([ACCEL, SOURCE, PLANE])[FILE].nunique().reset_index(name=FILE_COUNT)
    df_counted = df_counted.merge(file_count, how="left", on=[ACCEL, SOURCE, PLANE])

    df_counted.loc[:, PERCENTAGE] = round(
        (df_counted[COUNT] / df_counted[FILE_COUNT]) * 100, 2
    )
    
    df_counted = tfs.TfsDataFrame(df_counted.sort_values(PERCENTAGE, ascending=False), headers=df.headers)
    return df_counted


def print_results(df_counted: tfs.TfsDataFrame, print_percentage: float):
    """ Log the results to console (INFO level if logger is setup, print otherwise). 
    
    Args:
        df_counted (tfs.TfsDataFrame): TfsDataFrame with the evaluated results.
        print_percentage (float): Print out BPMs that appear in more than this percentage of measurements.
    """
    percentage_mask = df_counted[PERCENTAGE] >= print_percentage
    printer = print
    if LOG.hasHandlers():
        printer = LOG.info
    
    planes = df_counted[PLANE].unique()

    printer("Bad BPMs Summary. Hint: '[BPM]' were filtered as known bad BPMs.")
    for accel in sorted(df_counted[ACCEL].unique()):
        accel_mask = df_counted[ACCEL] == accel
        for source in sorted(df_counted[SOURCE].unique()):
            source_mask = df_counted[SOURCE] == source

            df_filtered = df_counted.loc[source_mask & accel_mask, :]
            if len(planes) == 2:
                # Merge X and Y for nicer output ---
                df_x = df_filtered.loc[df_filtered[PLANE] == "X", :].set_index(NAME)
                df_y = df_filtered.loc[df_filtered[PLANE] == "Y", :].set_index(NAME)

                df_merged = pd.merge(df_x, df_y, how="outer", left_index=True, right_index=True, suffixes=("X", "Y"))
                df_merged['max_pct'] = df_merged[[f"{PERCENTAGE}X", f"{PERCENTAGE}Y"]].max(axis=1)
                df_merged = df_merged.sort_values(by='max_pct', ascending=False)
                df_merged = df_merged.loc[df_merged['max_pct'] >= print_percentage, :]

                # Print Table ---
                header = f"{'BPM':>20s}  {'X':^18s}  {'Y':^18s}\n"
                msg = header + "\n".join(
                    f"{name:>20s}  " + 
                    "  ".join(
                        (
                        "{:^18s}".format("-") if np.isnan(row[f'{FILE_COUNT}{plane}']) else 
                        f"{row[f'{PERCENTAGE}{plane}']:5.1f}% "
                        "{:<11s}".format(f"({int(row[f'{COUNT}{plane}']):d}/{int(row[f'{FILE_COUNT}{plane}']):d})")
                        for plane in ('X', 'Y')
                        )
                    )
                    for name, row in df_merged.iterrows() 
                )

            else:
                # Print a list ---
                df_filtered = df_counted.loc[percentage_mask & source_mask & accel_mask, :]
                msg = "\n".join(
                    f"{row[NAME]:>20s} {row[PLANE]}: {row[PERCENTAGE]:5.1f}% ({row[COUNT]}/{row[FILE_COUNT]})" 
                    for _,row in df_filtered.iterrows()
                )
            printer(f"Highest bad BPMs of {accel} from {source}:\n{msg}")


# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    bad_bpms_summary()
