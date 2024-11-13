"""
Bad BPMs Summary
----------------

Scans all measurements in a list of given GUI output folders (`DATES`) and compiles a list of bad BPMs with
their given number of appearances after SVD and isolation forest.

Output will be written to `"bad_bpms.txt"`

Usage:
    1. Make sure that the measurements have the desired cleaning method applied.
    If needed rerun the measurements with the GUI.

    2. Adapt the `DATES` list at the beginning of this script accordingly

    3. a) Run this script in `OP_DATA` or
       b) `ln -s OP_DATA/Betabeat` into your working dir or
       c) change `ROOT` to a parent dir of your GUI output

"""
from collections.abc import Sequence
from pathlib import Path
from generic_parser import DotDict, EntryPointParameters, entrypoint
import tfs

from omc3.utils.iotools import PathOrStr
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


ROOT = Path("/user/slops/data/LHC_DATA/OP_DATA/Betabeat")

# Columns ---
NAME = "NAME"
ACCEL = "ACCELERATOR"
PLANE = "PLANE"
COUNT = "COUNT"
SOURCE = "SOURCE"

# Files ---
MEASUREMENT_DIR = "Measurements"
BAD_BPMS_SVD = "*.bad_bpms_*"
BAD_BPMS_IFOREST = "bad_bpms_iforest_*.tfs"


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="root",
        type=PathOrStr,
        default=ROOT,
        help="Path to the root directory, containing the dates."
    )
    params.add_parameter(
        name="dates",
        type=str,
        nargs="+",
        required=True,
        help=("Dates to include in analysis. "
        "This should be either subfolders in `root` or glob-patterns."
        )
    )
    params.add_parameter(
        name="outfile",
        type=PathOrStr,
        help="Path to the file to write out." 
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
    df_collection = collect_bad_bpms(Path(opt.root), opt.dates, opt.accel_glob)



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
    return tfs.concat(dfs, axis="index") 


def get_empty_df() -> tfs.TfsDataFrame:
    return tfs.TfsDataFrame(columns=[NAME, ACCEL, PLANE, COUNT, SOURCE])


def collect_date(date_dir: Path, accel_glob: str) -> tfs.TfsDataFrame:
    dfs = []
    for accel_dir in date_dir.glob(accel_glob):
        measurements_dir = accel_dir / MEASUREMENT_DIR
        for measurement in measurements_dir.iterdir():
            if not measurement.is_dir():
                continue

            df_collected = collect_measurement_dir(measurement)
            df_collected.loc[:, ACCEL] = accel_dir.name
            dfs.append(df_collected)
    return tfs.concat(dfs, axis="index") 
    

def collect_measurement_dir(measurement_dir: Path):
    dfs = []
    for svd_file in measurement_dir.glob(BAD_BPMS_SVD):
        dfs.append(read_svd_bad_bpms_file(svd_file))

    for iforest_file in measurement_dir.glob(BAD_BPMS_IFOREST):
        dfs.append(read_iforest_bad_bpms_file(iforest_file))
    return tfs.concat(dfs, axis="index")
        
            





def get_bad_bpms_for_beam_and_plane_svd(date: str, plane: str, beam: str, bad_bpms_list):
    (measurements, measurements_iforest) = bad_bpms_per_date(ROOT / date , plane, beam)
    for m in measurements:
        appeared = []
        #print(open(m).read())
        for bpm in open(m):
            words = bpm.split()
            if words[0] in appeared:
                continue
            if "model" in words and "not" in words:
                continue
            appeared.append(words[0])
            if words[0] in bad_bpms_list:
                bad_bpms_list[words[0]] = bad_bpms_list[words[0]] + 1
            else:
                bad_bpms_list[words[0]] = 1
    return len(measurements)

def get_bad_bpms_for_beam_and_plane_iforest(date: str, plane: str, beam: str, bad_bpms_list):
    (measurements, measurements_iforest) = bad_bpms_per_date(ROOT / date , plane, beam)
    for m in measurements_iforest:
        appeared = []
        #print(open(m).read())
        bpm_tfs = tfs.read_tfs(m, index="NAME")
        for bpm in bpm_tfs.index:
            if bpm in appeared:
                continue
            appeared.append(bpm)
            if bpm in bad_bpms_list:
                bad_bpms_list[bpm] = bad_bpms_list[bpm] + 1
            else:
                bad_bpms_list[bpm] = 1
    return len(measurements_iforest)


def do_print(msg: str):
    global OUTFILE
    print(msg)
    OUTFILE.write(f"{msg}\n")

def get_the_bad_bpms():
    for plane in ["x", "y"]:
        for beam in ["1", "2"]:
            
            pl = "H" if plane == "x" else "V"
            do_print("")
            do_print(f"BEAM {beam} {pl}")
            do_print("{:20} | {} | {} | {:8}  | {:7}".format())
            do_print("---------------------|-------|------|------------|-----------")
            bad_bpms_list_iforest = {}
            bad_bpms_list_svd = {}

            n_svd = 0
            n_iforest = 0
            for date in DATES:
                print(f"date {date}")
                n_svd += get_bad_bpms_for_beam_and_plane_svd(date, plane, beam, bad_bpms_list_svd)
                n_iforest += get_bad_bpms_for_beam_and_plane_iforest(date, plane, beam, bad_bpms_list_iforest)

            print(f"SVD [{n_svd} files]")
            for (k,v) in bad_bpms_list_svd.items():
                v_perc = v / n_svd * 100
                if v_perc > 75:
                    do_print(f"{k:20} | {plane:5} | {beam:4} | {v_perc:8.2f} % | SVD")
                elif v_perc > 50:
                    do_print(f"{k:20} | {plane:5} | {beam:4} | {v_perc:8.2f} % | SVD")
#                elif v_perc > 25:
#                    do_print(f"{k:20} | {plane:5} | {beam:4} | {v_perc:8.2f} % | SVD")


            print(f"IFOREST [{n_iforest} files]")
            for (k,v) in bad_bpms_list_iforest.items():
                v_perc = v / n_iforest * 100
                if v_perc > 75:
                    do_print(f"{k:20} | {plane:5} | {beam:4} | {v_perc:8.2f} % | IFOREST")
                elif v_perc > 50:
                    do_print(f"{k:20} | {plane:5} | {beam:4} | {v_perc:8.2f} % | IFOREST")
#                elif v_perc > 25:
#                    do_print(f"{k:20} | {plane:5} | {beam:4} | {v_perc:8.2f} % | IFOREST")

if __name__ == "__main__":
    bad_bpms_summary()
