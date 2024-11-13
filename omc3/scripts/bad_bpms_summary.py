"""
Bad BPMs Summary
----------------

Scans all measurements in a list of given GUI output folders and compiles a list of bad BPMs with
their given number of appearances after 'harpy' and 'isolation forest'.
"""
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from generic_parser import DotDict, EntryPointParameters, entrypoint
import pandas as pd
import tfs

from omc3.utils.iotools import PathOrStr
from omc3.utils import logging_tools

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

def collect_bad_bpms(root: Path, dates: Sequence[Path | str], accel_glob: str) -> tfs.TfsDataFrame:
    """ Create a TfsDataFrame with all bad-bpms within selcted dates. 
    
    
    
    """
    dfs = []
    for date in dates:
        date_dir = root / date
        if date_dir.is_dir():
            dfs.append(collect_date(date_dir, accel_glob))
            continue

        for date_dir in root.glob(date):
            dfs.append(collect_date(date_dir, accel_glob))
    return tfs.concat(dfs, axis="index", ignore_index=True, new_headers=get_headers_sum(dfs)) 


def collect_date(date_dir: Path, accel_glob: str) -> tfs.TfsDataFrame:
    dfs: list[tfs.TfsDataFrame] = []

    for accel_dir in date_dir.glob(accel_glob):
        measurements_dir = accel_dir / MEASUREMENT_DIR
        for measurement in measurements_dir.iterdir():
            if not measurement.is_dir():
                continue

            df_collected = collect_measurement_dir(measurement, accel=accel_dir.name)
            dfs.append(df_collected)

    if not len(dfs):
        return get_empty_df()

    return tfs.concat(dfs, axis="index", ignore_index=True, new_headers=get_headers_sum(dfs)) 
    

def collect_measurement_dir(measurement_dir: Path, accel: str) -> tfs.TfsDataFrame:
    dfs: list[tfs.TfsDataFrame] = []
    headers = defaultdict(int)

    readers_map = {
        BAD_BPMS_HARPY: read_harpy_bad_bpms_file,
        BAD_BPMS_IFOREST: read_iforest_bad_bpms_file
    }

    for glob_pattern, reader in readers_map.items():
        for bad_bpms_file in measurement_dir.glob(glob_pattern):
            new_df, meta = reader(bad_bpms_file, accel)
            headers[_get_header_key(meta)] += 1
            dfs.append(new_df)

    if not len(dfs):
        return get_empty_df()
    
    return tfs.concat(dfs, axis="index", ignore_index=True, new_headers=headers)


# File Readers --
            
def read_harpy_bad_bpms_file(svd_file: Path, accel: str) -> tuple[tfs.TfsDataFrame, pd.Series]:
    TO_INGNORE = ["not found in model"]
    COMMENT = "#"

    plane = svd_file.name[-1]

    with svd_file.open() as f:
        lines = f.readlines()
    lines = [line.strip().split(maxsplit=1) for line in lines]
    lines = [line for line in lines if not line[0].startswith(COMMENT) and line[1].lower() not in TO_INGNORE]
    
    df = get_empty_df()
    df.loc[:, NAME] = list(set(line[0] for line in lines))
    meta = pd.Series({PLANE: plane.upper(), SOURCE: HARPY, ACCEL: accel})
    df.loc[:, meta.keys()] = meta.to_list()
    return df, meta


def read_iforest_bad_bpms_file(iforest_file: Path, accel: str) -> tuple[tfs.TfsDataFrame, pd.Series]:
    df_iforest = tfs.read(iforest_file)
    plane = iforest_file.stem[-1]

    df = get_empty_df()
    df.loc[:, NAME] = list(set(df_iforest[NAME]))  # hint: be sure to ignore index
    meta = pd.Series({PLANE: plane.upper(), SOURCE: IFOREST, ACCEL: accel})
    df.loc[:, meta.keys()] = meta.to_list()
    return df, meta

# Helper --

def get_empty_df() -> tfs.TfsDataFrame:
    return tfs.TfsDataFrame(columns=[NAME, ACCEL, PLANE, SOURCE])


def get_headers_sum(dfs: Sequence[tfs.TfsDataFrame]) -> dict[str, int]:
    if not len(dfs):
        return {}

    all_keys = set(key for df in dfs for key in df.headers.keys())
    return {
        key: sum(df.headers[key] for df in dfs if key in df.headers) for key in all_keys
    }


def _get_header_key(series: pd.Series) -> str:
    return f"{series[SOURCE]}_{series[ACCEL]}_{series[PLANE]}"


# Evaluaion ----


def evaluate(df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    df_counted = df.groupby(list(df.columns)).size().reset_index(name=COUNT)
    df_counted = tfs.TfsDataFrame(df_counted.sort_values(COUNT, ascending=False), headers=df.headers)
    df_counted.loc[:, PERCENTAGE] = round(
        (df_counted[COUNT] / df_counted.apply(_get_header_key, axis="columns").map(df_counted.headers)) * 100, 2
    )
    return df_counted


def print_results(df_counted: tfs.TfsDataFrame, print_percentage: float):
    percentage_mask = df_counted[PERCENTAGE] >= print_percentage

    for accel in sorted(df_counted[ACCEL].unique()):
        accel_mask = df_counted[ACCEL] == accel
        for source in sorted(df_counted[SOURCE].unique()):
            source_mask = df_counted[SOURCE] == source
            df_filtered = df_counted.loc[percentage_mask & source_mask & accel_mask, :]
            msg = "\n".join(
                f"{row[NAME]:>20s} {row[PLANE]}: {row[PERCENTAGE]:5.1f}% ({row[COUNT]}/{df_counted.headers[_get_header_key(row)]})" 
                for _,row in df_filtered.iterrows()
            )
            LOG.info(f"Highest bad BPMs of {accel} from {source}:\n" + msg)


# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    bad_bpms_summary(
        root="/home/jdilly/mnt/user/slops/data/LHC_DATA/OP_DATA/Betabeat",
        dates=["2023-09-06"],
        accel_glob="LHCB*",
        outfile="bad_bpms_summary.tfs",
        print_percentage=25.,
    )
