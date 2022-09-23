r"""
Knob Extractor
--------------

Entrypoint to extract knobs from ``NXCALS`` at a given time, and eventually write them out to a file.
This script can also be used to display information about the StateTracker State.
The data is fetched from ``NXCALS`` through ``pytimber`` using the **StateTracker** fields.

.. note::
    Please note that access to the GPN is required to use this functionality.

**Arguments:**

*--Optional--*

- **knob_definitions** *(PathOrStrOrDataFrame)*:

    User defined path to the knob-definitions, or (via python) a dataframe
    containing the knob definitions with the columns 'madx', 'lsa' and
    'scaling'.


- **knobs** *(str)*:

    A list of knob names or categories to extract. Available categories
    are: sep, xing, chroma, ip_offset, disp, mo.

    default: ``['sep', 'xing', 'chroma', 'ip_offset', 'disp', 'mo']``


- **output** *(PathOrStr)*:

    Specify user-defined output path. This should probably be
    `model_dir/knobs.madx`

    default: ``knobs.madx``


- **state**:

    Prints the state of the StateTracker. Does not extract anything else.

    action: ``store_true``


- **time** *(str)*:

    At what time to extract the knobs. Accepts ISO-format (YYYY-MM-
    DDThh:mm:ss), timestamp or 'now'. The default timezone for the ISO-
    format is local time, but you can force e.g. UTC by adding +00:00.

    default: ``now``


- **timedelta** *(str)*:

    Add this timedelta to the given time. The format of timedelta is
    '((\d+)(\w))+' with the second token being one of s(seconds),
    m(minutes), h(hours), d(days), w(weeks), M(months) e.g 7m = 7 minutes,
    1d = 1day, 7m30s = 7 min 30 secs. A prefix '_' specifies a negative
    timedelta. This allows for easily getting the setting e.g. 2h ago:
    '_2h' while setting the `time` argument to 'now' (default).


"""
import argparse
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import tfs
from dateutil.relativedelta import relativedelta

from generic_parser import EntryPointParameters, entrypoint
from omc3.utils.iotools import PathOrStr, PathOrStrOrDataFrame
from omc3.utils.logging_tools import get_logger
from omc3.utils.mock import cern_network_import

pytimber = cern_network_import("pytimber")

LOGGER = get_logger(__name__)

AFS_ACC_MODELS_LHC = Path("/afs/cern.ch/eng/acc-models/lhc/current")
ACC_MODELS_LHC = Path("acc-models-lhc")
KNOBS_FILE_ACC_MODELS = ACC_MODELS_LHC / "operation" / "knobs.txt"
KNOBS_FILE_AFS = AFS_ACC_MODELS_LHC / "operation" / "knobs.txt"

MINUS_CHARS: Tuple[str, ...] = ("_", "-")

class Col:
    """ Columns used in this script. """
    madx: str = "madx"
    lsa: str = "lsa"
    scaling: str = "scaling"
    value: str = "value"


KNOB_CATEGORIES: Dict[str, List[str]] = {
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

python -m omc3.knob_extractor --knobs disp chroma --time 2022-05-04T14:00     
    extracts the chromaticity and dispersion knobs at 14h on May 4th 2022

python -m omc3.knob_extractor --knobs disp chroma --time now _2h 
    extracts the chromaticity and dispersion knobs as of 2 hours ago

python -m omc3.knob_extractor --state
    prints the current StateTracker/State metadata

python -m omc3.knob_extractor --knobs disp sep xing chroma ip_offset mo --time now
    extracts the current settings for all the knobs
"""


def get_params():
    return EntryPointParameters(
        knobs=dict(
            type=str,
            nargs='*',
            help=(
                "A list of knob names or categories to extract. "
                f"Available categories are: {', '.join(KNOB_CATEGORIES.keys())}."
            ),
            default=list(KNOB_CATEGORIES.keys()),
        ),
        time=dict(
            type=str,
            help=(
                "At what time to extract the knobs. "
                "Accepts ISO-format (YYYY-MM-DDThh:mm:ss), timestamp or 'now'. "
                "The default timezone for the ISO-format is local time, "
                "but you can force e.g. UTC by adding +00:00."
            ),
            default="now",
        ),
        timedelta=dict(
            type=str,
            help=(
                "Add this timedelta to the given time. "
                "The format of timedelta is '((\\d+)(\\w))+' "
                "with the second token being one of "
                "s(seconds), m(minutes), h(hours), d(days), w(weeks), M(months) "
                "e.g 7m = 7 minutes, 1d = 1day, 7m30s = 7 min 30 secs. "
                "A prefix '_' specifies a negative timedelta. "
                "This allows for easily getting the setting "
                "e.g. 2h ago: '_2h' while setting the `time` argument to 'now' (default)."
            ),
        ),
        state=dict(
            action='store_true',
            help=(
                "Prints the state of the StateTracker. "
                "Does not extract anything else."
            ),
        ),
        output=dict(
            type=PathOrStr,
            help=(
                "Specify user-defined output path. "
                "This should probably be `model_dir/knobs.madx`"
            ),
        ),
        knob_definitions=dict(
            type=PathOrStrOrDataFrame,
            help=(
                "User defined path to the knob-definitions, "
                "or (via python) a dataframe containing the knob definitions "
                "with the columns 'madx', 'lsa' and 'scaling'."
            ),
        ),
    )


@entrypoint(
    get_params(), strict=True,
    argument_parser_args=dict(
        epilog=USAGE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog="Knob Extraction Tool."
    )
)
def main(opt) -> Optional[pd.DataFrame]:
    """ Main knob extracting function. """
    ldb = pytimber.LoggingDB(source="nxcals", loglevel=logging.ERROR)
    time = _parse_time(opt.time, opt.timedelta)

    if opt.state:
        # only print the state of the StateTracker - the MetaState!
        # I still don't know what this does,
        # because I only get back that the variable does not exist. (jdilly)
        state = ldb.get("LhcStateTracker:State", time)  # do first to have output together
        LOGGER.info("---- STATE ------------------------------------")
        LOGGER.info(state)
        return None
    
    knobs_dict = _parse_knobs_defintions(opt.knob_definitions)
    knobs_extract = _extract_and_gather(ldb, knobs_dict, opt.knobs, time)
    if opt.output:
        _write_knobsfile(opt.output, knobs_extract, time)
    return knobs_extract


def extract(ldb, knob_categories: Sequence[str], time: datetime) -> Dict[str, float]:
    """
    Main function to gather data from  the state-tracker.

    Args:
        ldb (pytimber.LoggingDB): The pytimber database.
        knob_categories (Sequence[str]): Knob Categories or Knob-Names to extract.
        time (datetime): The time, when to extract.

    Returns:
        Dict[str, float]: Contains all the extracted knobs.
        When extraction was not possible, the value attribute of the respective KnobEntry is still None

    """
    LOGGER.info(f"---- EXTRACTING KNOBS @ {time} ----")
    knobs = {}

    for category in knob_categories:
        for knob in KNOB_CATEGORIES.get(category, [category]):
            knobkey = f"LhcStateTracker:{knob}:target"
            knobs[knob] = None  # to log that this was tried to be extracted.

            knobvalue = ldb.get(knobkey, time.timestamp())  # use timestamp to preserve timezone info
            if knobkey not in knobvalue:
                LOGGER.warning(f"No value for {knob} found")
                continue

            timestamps, values = knobvalue[knobkey]
            if len(values) == 0:
                LOGGER.debug(f"No value for {knob} found")
                continue

            value = values[-1]
            if not math.isfinite(value):
                LOGGER.debug(f"Value for {knob} is not a number or infinite")
                continue

            LOGGER.info(f"Knob value for {knob} extracted: {value} (unscaled)")
            knobs[knob] = value

    return knobs


def _extract_and_gather(ldb, knobs_definitions: pd.DataFrame,
                        knob_categories: Sequence[str],
                        time: datetime) -> pd.DataFrame:
    """
    Main function to gather data from  the state-tracker.

    Args:
        ldb (pytimber.LoggingDB): The pytimber database.
        knobs_definitions (KnobsDict): A mapping of all knob-names to KnobEntries.
        knob_categories (Sequence[str]): Knob Categories or Knob-Names to extract.
        time (datetime): The time, when to extract.

    Returns:
        pd.Dataframe: Contains all the extracted knobs.
        When extraction was not possible, the value attribute of the respective entry is NAN

    """
    extracted_knobs = extract(ldb, knob_categories=knob_categories, time=time)
    undefined_knobs = [knob for knob in extracted_knobs if knob not in knobs_definitions.index]
    if undefined_knobs:
        raise KeyError(
            "The following knob(s) could not be found "
            f"in the knob-definitions: '{', '.join(undefined_knobs)}'"
        )

    knob_names = list(extracted_knobs.keys())
    knobs = pd.DataFrame(index=knob_names, columns=[Col.lsa, Col.madx, Col.scaling, Col.value])
    knobs[[Col.lsa, Col.madx, Col.scaling]] = knobs_definitions.loc[knob_names, :]
    knobs[Col.value] = pd.Series(extracted_knobs)
    return knobs


def _write_knobsfile(output: Union[Path, str], collected_knobs: pd.DataFrame, time):
    """ Takes the collected knobs and writes them out into a text-file. """
    collected_knobs = collected_knobs.copy()  # to not modify the return df

    # Sort the knobs by category
    category_knobs = {}
    for category, category_names in KNOB_CATEGORIES.items():
        names = [name for name in collected_knobs.index if name in category_names]
        if not names:
            continue

        category_knobs[category] = collected_knobs.loc[names, :]
        collected_knobs = collected_knobs.drop(index=names)
    category_knobs["Other Knobs"] = collected_knobs

    # Write them out
    with open(output, "w") as outfile:
        outfile.write(f"!! --- knobs extracted by knob_extractor\n")
        outfile.write(f"!! --- extracted knobs for time {time}\n\n")
        for category, knobs_df in category_knobs.items():
            outfile.write(f"!! --- {category:10} --------------------\n")
            for knob, knob_entry in knobs_df.iterrows():
                outfile.write(f"{get_madx_command(knob_entry)}\n")
            outfile.write("\n")
        outfile.write("\n")


# Knobs Dict -------------------------------------------------------------------

def _get_knobs_def_file(user_defined: Optional[Union[Path, str]] = None) -> Path:
    """ Check which knobs-definition file is appropriate to take. """
    if user_defined is not None:
        LOGGER.info(f"Using user defined knobs.txt: '{user_defined}")
        return Path(user_defined)

    if KNOBS_FILE_ACC_MODELS.is_file():
        LOGGER.info(f"Using model folder's knobs.txt: '{KNOBS_FILE_ACC_MODELS}")
        return KNOBS_FILE_ACC_MODELS

    if KNOBS_FILE_AFS.is_file():
        # if all fails, fall back to lhc acc-models
        LOGGER.info(f"Using fallback knobs.txt: '{KNOBS_FILE_AFS}'")
        return KNOBS_FILE_AFS

    raise FileNotFoundError("None of the knobs-definition files are available.")


def _load_knobs_dict(file_path: Union[Path, str]) -> pd.DataFrame:
    """ Load the knobs-definition file and convert into KnobsDict.
    Each line in this file should consist of four comma separated entries:
    madx-name, lsa-name, scaling factor, knob-test value.
    Alternatively, a TFS-file is also allowed, but needs to have the suffix ``.tfs``.

    Args:
        file_path (Path): Path to the knobs definition file.

    Returns:
        Dataframe with LSA names (but with colon instead of /) as
        keys and KnobEntries (without values) as value.
    """
    if Path(file_path).suffix == ".tfs":
        # just in case someone wants to give tfs files (hidden feature)
        df = tfs.read_tfs(file_path)
    else:
        # parse csv file (the official way)
        dtypes = {Col.madx: str, Col.lsa: str, Col.scaling: float, "test": float}
        converters = {Col.madx: str.strip, Col.lsa: str.strip}  # strip whitespaces
        df = pd.read_csv(file_path, comment="#", names=dtypes.keys(), dtype=dtypes, converters=converters)
    return _to_knobs_dataframe(df)


def _to_knobs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """ Converts a DataFrame into the required Dictionary structure.

    Args:
        df (pd.DataFrame): DataFrame containing at least the columns
                           'lsa', 'madx', 'scaling' (upper or lowercase)

    Returns:
        Dataframe with LSA names (but with colon instead of /) as
        keys and 'lsa', 'madx', 'scaling' and (empty) 'value' columns.

    """
    df.columns = df.columns.astype(str).str.lower()
    df = df[[Col.lsa, Col.madx, Col.scaling]].set_index(Col.lsa, drop=False)
    df.index = df.index.map(lsa2name)
    return df


def _parse_knobs_defintions(knobs_def_input: Optional[Union[Path, str, pd.DataFrame]]) -> pd.DataFrame:
    """ Parse the given knob-definitions either from a csv-file or from a DataFrame. """
    if isinstance(knobs_def_input, pd.DataFrame):
        return _to_knobs_dataframe(knobs_def_input)

    # input points to a file or is None
    knobs_def_file = _get_knobs_def_file(knobs_def_input)
    return _load_knobs_dict(knobs_def_file)


def get_madx_command(knob_data: pd.Series) -> str:
    if Col.value not in knob_data.index:
        raise KeyError("Value entry not found in extracted knob_data. "
                       "Something went wrong as it should at least be NaN.")
    if knob_data[Col.value] is None or pd.isna(knob_data[Col.value]):
        return f"! {knob_data[Col.madx]} : No Value extracted"
    return f"{knob_data[Col.madx]} := {knob_data[Col.value] * knob_data[Col.scaling]};"


# Time Tools -------------------------------------------------------------------

def _parse_time(time: str, timedelta: str = None) -> datetime:
    """ Parse time from given time-input. """
    t = _parse_time_from_str(time)
    if timedelta:
        t = _add_time_delta(t, timedelta)
    return t


def _parse_time_from_str(time_str: str) -> datetime:
    """ Parse time from given string. """
    # Now? ---
    if time_str.lower() == "now":
        return datetime.now()

    # ISOFormat? ---
    try:
        return datetime.fromisoformat(time_str)
    except (TypeError, ValueError):
        LOGGER.debug("Could not parse time string as ISO format")
        pass

    # Timestamp? ---
    try:
        return datetime.fromtimestamp(int(time_str))
    except (TypeError, ValueError):
        LOGGER.debug("Could not parse time string as a timestamp")
        pass

    raise ValueError(f"Couldn't read datetime '{time_str}'")


def _add_time_delta(time: datetime, delta_str: str) -> datetime:
    """ Parse delta-string and add time-delta to time. """
    sign = -1 if delta_str[0] in MINUS_CHARS else 1
    all_deltas = re.findall(r"(\d+)(\w)", delta_str)  # tuples (value, timeunit-char)

    # mapping char to the time-unit as accepted by relativedelta,
    # following ISO-8601 for time durations
    char2unit = dict(
        s='seconds', m='minutes', h='hours',
        d='days', w='weeks', M='months', Y="years",
    )

    # add all deltas, which are tuples of (value, timeunit-char)
    time_parts = {char2unit[delta[1]]: sign * int(delta[0]) for delta in all_deltas}
    time = time + relativedelta(**time_parts)

    return time


# Other tools ------------------------------------------------------------------

def lsa2name(lsa_name: str) -> str:
    """LSA name -> Variable in Timber/StateTracker conversion."""
    return lsa_name.replace("/", ":")


def name2lsa(name: str) -> str:
    """Variable in Timber/StateTracker -> LSA name conversion."""
    return name.replace(":", "/")


if __name__ == "__main__":
    main()
