"""
Bad BPMs Summary
----------------

Scans all measurements in a list of given GUI output folders and compiles a list of bad BPMs with
their given number of appearances after 'harpy' and 'isolation forest'.
"""
from collections.abc import Sequence
from pathlib import Path

import tfs
from generic_parser import DotDict, EntryPointParameters, entrypoint

from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr

LOG = logging_tools.get_logger(__name__)

# Constants ---
ROOT = Path("/user/slops/data/LHC_DATA/OP_DATA/Betabeat")
IFOREST = "IFOREST"
HARPY = "HARPY"
KEY = "HEADER_KEY"

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
MEASUREMENT_DIR = "Measurements"
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
        type=float,
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
def bad_bpms_summary(opt: DotDict):
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


# Collection of Data ---

def get_empty_df() -> tfs.TfsDataFrame:
    """ Create an empty TfsDataFrame with the correct column names. """
    return tfs.TfsDataFrame(columns=[NAME, ACCEL, PLANE, SOURCE, FILE])


def collect_bad_bpms(root: Path, dates: Sequence[Path | str], accel_glob: str) -> tfs.TfsDataFrame:
    """ Create a TfsDataFrame with all bad-bpms within selcted dates.

    Args:
        root (Path): Root path to the GUI output folder.
        dates (Sequence[Path | str]): List of dates or glob patterns to collect bad-bpms from.
        accel_glob (str): Accelerator name (glob for the sub-directories).

    Returns:
        tfs.TfsDataFrame: TfsDataFrame with all bad-bpms within selcted dates.
    
    """
    dfs = []
    for date in dates:
        date_dir = root / date
        if date_dir.is_dir():
            dfs.append(collect_date(date_dir, accel_glob))
            continue

        for date_dir in root.glob(date):
            dfs.append(collect_date(date_dir, accel_glob))
    return tfs.concat(dfs, axis="index", ignore_index=True) 


def collect_date(date_dir: Path, accel_glob: str) -> tfs.TfsDataFrame:
    """ Collect bad-bpms for a single date.
    
    Args:
        date_dir (Path): Path to the date directory.
        accel_glob (str): Accelerator name (glob for the sub-directories).

    Returns:
        tfs.TfsDataFrame: TfsDataFrame with all bad-bpms for the date.
    """
    dfs: list[tfs.TfsDataFrame] = []

    for accel_dir in date_dir.glob(accel_glob):
        measurements_dir = accel_dir / MEASUREMENT_DIR
        if not measurements_dir.is_dir():
            continue

        for measurement in measurements_dir.iterdir():
            if not measurement.is_dir():
                continue

            df_collected = collect_measurement_dir(measurement)
            df_collected.loc[:, ACCEL] = accel_dir.name
            dfs.append(df_collected)

    if not len(dfs):
        return get_empty_df()

    return tfs.concat(dfs, axis="index", ignore_index=True) 
    

def collect_measurement_dir(measurement_dir: Path) -> tfs.TfsDataFrame:
    """ Collect bad-bpms for a single measurement directory.
    
    Args:
        measurement_dir (Path): Path to the measurement directory.

    Returns:
        tfs.TfsDataFrame: TfsDataFrame with all bad-bpms from the measurement directory.
    
    """
    dfs: list[tfs.TfsDataFrame] = []

    readers_map = {
        BAD_BPMS_HARPY: read_harpy_bad_bpms_file,
        BAD_BPMS_IFOREST: read_iforest_bad_bpms_file
    }

    for glob_pattern, reader in readers_map.items():
        for bad_bpms_file in measurement_dir.glob(glob_pattern):
            dfs.append(reader(bad_bpms_file))

    if not len(dfs):
        return get_empty_df()
    
    return tfs.concat(dfs, axis="index", ignore_index=True)


# File Readers --
            
def read_harpy_bad_bpms_file(svd_file: Path) -> tfs.TfsDataFrame:
    """ Reads a harpy bad-bpm file and returns a TfsDataFrame with all unique bad-bpms.
    
    Args:
        svd_file (Path): Path to the bad-bpm file.

    Returns:
        tfs.TfsDataFrame: TfsDataFrame with all unique bad-bpms.

    """
    TO_INGNORE = ["not found in model", "known bad bpm"]
    COMMENT = "#"

    plane = svd_file.name[-1]

    with svd_file.open() as f:
        lines = f.readlines()
    lines = [line.strip().split(maxsplit=1) for line in lines]
    lines = [line for line in lines if not line[0].startswith(COMMENT) and line[1].lower() not in TO_INGNORE]
    
    df = get_empty_df()
    df.loc[:, NAME] = list(set(line[0] for line in lines))
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
    """ Prints the results to the console (INFO level). 
    
    Args:
        df_counted (tfs.TfsDataFrame): TfsDataFrame with the evaluated results.
        print_percentage (float): Print out BPMs that appear in more than this percentage of measurements.
    """
    percentage_mask = df_counted[PERCENTAGE] >= print_percentage

    for accel in sorted(df_counted[ACCEL].unique()):
        accel_mask = df_counted[ACCEL] == accel
        for source in sorted(df_counted[SOURCE].unique()):
            source_mask = df_counted[SOURCE] == source
            df_filtered = df_counted.loc[percentage_mask & source_mask & accel_mask, :]
            msg = "\n".join(
                f"{row[NAME]:>20s} {row[PLANE]}: {row[PERCENTAGE]:5.1f}% ({row[COUNT]}/{row[FILE_COUNT]})" 
                for _,row in df_filtered.iterrows()
            )
            LOG.info(f"Highest bad BPMs of {accel} from {source}:\n" + msg)


# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    bad_bpms_summary()
