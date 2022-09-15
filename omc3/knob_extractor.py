"""
Knob Extractor
--------------

Will extract knobs and give information about the current beam process (no it doesn't?).
Fetches data from nxcals through pytimber using the StateTracker fields.
"""
import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Union

import pandas as pd
from dateutil.relativedelta import relativedelta

from generic_parser import EntryPointParameters, entrypoint
from omc3.utils.logging_tools import get_logger
from omc3.utils.mock import cern_network_import

LOGGER = get_logger(__name__)

pytimber = cern_network_import("pytimber")


AFS_ACC_MODELS_LHC = Path("/") / "afs" / "cern.ch" / "eng" / "acc-models" / "lhc" / "current"
ACC_MODELS_LHC = Path("acc-models-lhc")
KNOBS_TXT_PATH = Path("operation") / "knobs.txt"

KNOBS_TXT_MDLDIR = ACC_MODELS_LHC / KNOBS_TXT_PATH
KNOBS_TXT_AFS = AFS_ACC_MODELS_LHC / KNOBS_TXT_PATH

MINUS_CHARS = ("_", "-")

KNOB_NAMES = {
    "sep": [
        "LHCBEAM:IP1-SEP-H-MM",
        "LHCBEAM:IP1-SEP-V-MM",
        "LHCBEAM:IP5-SEP-H-MM",
        "LHCBEAM:IP5-SEP-V-MM",
    ],
    "xing": [
        "LHCBEAM:IP1-XING-V-MURAD",
        "LHCBEAM:IP1-XING-H-MURAD",
        "LHCBEAM:IP5-XING-V-MURAD",
        "LHCBEAM:IP5-XING-H-MURAD",
    ],
    "chroma": [
        "LHCBEAM1:QPH",
        "LHCBEAM1:QPV",
        "LHCBEAM2:QPH",
        "LHCBEAM2:QPV",
    ],
    "ip_offset": [
        "LHCBEAM:IP1-OFFSET-V-MM",
        "LHCBEAM:IP2-OFFSET-V-MM",
        "LHCBEAM:IP5-OFFSET-H-MM",
        "LHCBEAM:IP8-OFFSET-H-MM",
    ],
    "disp": [
        "LHCBEAM:IP1-SDISP-CORR-SEP",
        "LHCBEAM:IP1-SDISP-CORR-XING",
        "LHCBEAM:IP5-SDISP-CORR-SEP",
        "LHCBEAM:IP5-SDISP-CORR-XING",
    ],
    "mo": [
        "LHCBEAM1:LANDAU_DAMPING",
        "LHCBEAM2:LANDAU_DAMPING"
    ]
}

USAGE_EXAMPLES = """Usage Examples:

python knob_extractor.py disp chroma --time 2022-05-04T14:00     
    extracts the chromaticity and dispersion knobs at 14h on May 4th 2022

python knob_extractor.py disp chroma --time now _2h 
    extracts the chromaticity and dispersion knobs as of 2 hours ago

python knob_extractor.py --state
    prints the current StateTracker/State metadata

python knob_extractor.py disp sep xing chroma ip_offset mo --time now
    extracts the current settings for all the knobs
"""


def get_params():
    return EntryPointParameters(
        knobs=dict(
            type=str,
            nargs='*',
            help=(
                "A list of knob names or categories to extract. "
                f"Available categories are: {', '.join(KNOB_NAMES.keys())}."
            ),
            default=list(KNOB_NAMES.keys()),
        ),
        time=dict(
            type=str,
            nargs='+',
            help=("Triggers the extraction and"
                  "defines the time at which to extract the knob settings. "
                  "'now': extracts the current knob setting. "
                  "<time>: extracts the knob setting for a given time. "
                  "<time> <timedelta>: extracts the knob setting for a given time, "
                  "with an offset of <timedelta> "
                  "the format of timedelta is '((\\d+)(\\w))+' "
                  "with the second token being one of "
                  "s(seconds), m(minutes), h(hours), d(days), w(weeks), M(months) "
                  "e.g 7m = 7 minutes, 1d = 1day, 7m30s = 7 min 30 secs. "
                  "a prefix '_' specifies a negative timedelta"
                  )
        ),
        state=dict(
            action='store_true',
            help="Prints the state of the statetracker. Does not extract anything."),
        output=dict(
            type=str,
            default='knobs.madx',
            help="Specify user-defined output path. This should probably be `model_dir/knobs.madx`"),
        knobs_txt=dict(
            type=str,
            help="user defined path to knob.txt."
        ),
    )


@dataclass
class KnobEntry:
    madx: str
    lsa: str
    scaling: float
    value: float = None

    def get_madx(self):
        if not self.value:
            return f"! {self.madx} : No Value extracted"
        return f"{self.madx} := {self.value * self.scaling};"


KnobsDict = Dict[str, KnobEntry]


@entrypoint(
    get_params(), strict=True,
    argument_parser_args=dict(
        epilog=USAGE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog="Knob Extraction Tool."
    )
)
def main(opt):
    if not opt.time and not opt.state:
        raise ValueError("No functionality selected. Set either `time` or `state`.")

    ldb = pytimber.LoggingDB(source="nxcals")
    knobs_extract = None

    if opt.time is not None:
        time = _parse_time(opt.time)
        knobs_def_file = _get_knobs_def_file(opt.knobs_txt)
        knobs_dict = _load_knobs_dict(knobs_def_file)
        knobs_extract = _extract(ldb, knobs_dict, opt.knobs, time)
        _write_knobsfile(opt.output, knobs_extract, time)

    if opt.state:
        raise NotImplementedError("StateTracker State not implemented.")
        # LOGGER.info("---- STATE ------------------------------------")
        # # TODO: ceck for available fields (and checking the exact name)
        # # and prepare for better presentation
        # LOGGER.info(ldb.get("LhcStateTracker:State", t1))
        # LOGGER.info(ldb.get("LhcStateTracker/State", t1))

    return knobs_extract


def _extract(ldb, knobs_dict: KnobsDict, knob_categories: Sequence[str], time: datetime) -> Dict[str, KnobsDict]:
    """
    Main function to gather data from  the state-tracker.

    Args:
        ldb: The pytimber database.
        knobs_dict (KnobsDict): A mapping of all knob-names to KnobEntries.
        knob_categories (Sequence[str]): Knob Categories or Knob-Names to extract.
        time (datetime): The time, when to extract.

    Returns:
        Dict[str, KnobsDict]: Contains all the extracted knobs, grouped by categories.
        When extraction was not possible, the value attribute of the respective KnobEntry is still None

    """
    LOGGER.info("---- EXTRACTING KNOBS -------------------------")
    LOGGER.info(f"extracting knobs for {time}")
    knobs = {}

    LOGGER.info("---- KNOBS ------------------------------------")
    for category in knob_categories:
        knobs[category] = {}  # only here to group them on output

        for knob in KNOB_NAMES.get(category, category):
            knobs[category][knob] = knobs_dict[knob]

            LOGGER.info(f"Looking for {knob:<34s} ")
            knobkey = f"LhcStateTracker:{knob}:target"
            knobvalue = ldb.get(knobkey, time)
            if knobkey not in knobvalue:
                LOGGER.warning(f"No value for {knob} found!")
                continue

            LOGGER.debug(f"Some value for {knob} extracted.")
            timestamps, values = knobvalue[knobkey]
            if len(values) == 0:
                LOGGER.warning(f"No value for {knob} found for given time!")
                continue

            value = values[-1]
            if not math.isfinite(value):
                LOGGER.warning(f"Value for {knob} is not a number or infinite!")
                continue

            LOGGER.info(f"Knob value for {knob} extracted: {value} (unscaled)")
            knobs[category][knob].value = value
    return knobs


def _write_knobsfile(output: Union[Path, str], collected_knobs: Dict[str, KnobsDict], time):
    """ Takes the collected knobs and writes them out into a text-file. """
    with open(output, "w") as outfile:
        outfile.write(f"!! --- knobs extracted by knob_extractor\n")
        outfile.write(f"!! --- extracted knobs for time {time}\n\n")
        for category, knobs in collected_knobs.items():
            if len(knobs) > 1:
                outfile.write(f"!! --- {category:10} --------------------\n")

            for knob, knob_entry in knobs.items():
                outfile.write(f"{knob_entry.get_madx()}\n")
            outfile.write("\n")
        outfile.write("\n")


# Knobs Dict -------------------------------------------------------------------

def _get_knobs_def_file(user_defined=None) -> Path:
    """ Check which knobs-definition file is appropriate to take. """
    if user_defined is not None:
        LOGGER.info(f"Using user defined knobs.txt: '{user_defined}")
        return Path(user_defined)

    if KNOBS_TXT_MDLDIR.is_file():
        LOGGER.info(f"Using model folder's knobs.txt: '{KNOBS_TXT_MDLDIR}")
        return KNOBS_TXT_MDLDIR

    if KNOBS_TXT_AFS.is_file():
        # if all fails, fall back to lhc acc-models
        LOGGER.info(f"Using fallback knobs.txt: '{KNOBS_TXT_AFS}'")
        return KNOBS_TXT_AFS

    raise FileNotFoundError("None of the knobs-definition files are available.")


def _load_knobs_dict(file_path: Union[Path, str]) -> KnobsDict:
    """ Load the knobs-definition file and convert into KnobsDict.
    Each line in this file should consist of four comma separated entries:
    madx-name, lsa-name, scaling factor, knob-test value.

    Args:
        file_path (Path): Path to the knobs definition file.

    Returns:
        Dictionary with LSA names (but with colon instead of /) as
        keys and KnobEntries (without values) as value.
    """
    dtypes = {"madx": str, "lsa": str, "scaling": float, "test": float}
    converters = {'madx': str.strip, 'lsa': str.strip}  # strip whitespaces
    df = pd.read_csv(file_path, comment="#", names=dtypes.keys(), dtype=dtypes, converters=converters)
    df = df.drop(columns="test").set_index("lsa", drop=False)
    return {
        r[0].replace("/", ":"): KnobEntry(**r[1].to_dict()) for r in df.iterrows()
    }


# Time Tools -------------------------------------------------------------------

def _parse_time(time: Sequence[str]) -> datetime:
    """ Parse time from given time-input. """
    t = _parse_time_from_str(time[0])
    if len(time) > 1:
        t = _add_time_delta(t, time[1])
    return t


def _parse_time_from_str(time_str: str) -> datetime:
    """ Parse time from given string. """
    # Now? ---
    if time_str == "now":
        return datetime.now()

    # ISOFormat? ---
    try:
        return datetime.fromisoformat(time_str)
    except (TypeError, ValueError):
        pass

    # Timestamp? ---
    try:
        return datetime.fromtimestamp(int(time_str))
    except (TypeError, ValueError):
        pass

    raise ValueError(f"Couldn't read datetime '{time_str}'")


def _add_time_delta(time: datetime, delta_str: str) -> datetime:
    """ Parse delta-string and add time-delta to time. """
    sign = -1 if delta_str[0] in MINUS_CHARS else 1
    all_deltas = re.findall(r"(\d+)(\w)", delta_str)

    # following ISO-8601 for time durations
    char_map = dict(
        s='seconds', m='minutes', h='hours',
        d='days', w='weeks', M='months', Y="years",
    )

    # add all deltas
    time_parts = {char_map[delta[1]]: sign * int(delta[0]) for delta in all_deltas}
    time = time + relativedelta(**time_parts)

    return time


if __name__ == "__main__":
    main()
