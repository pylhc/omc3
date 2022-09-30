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
from dateutil.relativedelta import relativedelta

import tfs
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
STATE_VARIABLES: Dict[str, str] = {
    'opticName': 'Optics',
    'beamProcess': 'Beam Process',
    'opticId': 'Optics ID',
    'hyperCycle': 'HyperCycle',
    # 'secondsInBeamProcess ': 'Beam Process running (s)',
}


class Col:
    """ DataFrame Columns used in this script. """
    madx: str = "madx"
    lsa: str = "lsa"
    scaling: str = "scaling"
    value: str = "value"


class Head:
    """ TFS Headers used in this script."""
    time: str = "EXTRACTION_TIME"


KNOB_CATEGORIES: Dict[str, List[str]] = {
    "sep": [
        "LHCBEAM:IP1-SEP-H-MM",
        "LHCBEAM:IP1-SEP-V-MM",
        "LHCBEAM:IP5-SEP-H-MM",
        "LHCBEAM:IP5-SEP-V-MM",
        "LHCBEAM:IP2-SEP-H-MM",
        "LHCBEAM:IP2-SEP-V-MM",
        "LHCBEAM:IP8-SEP-H-MM",
        "LHCBEAM:IP8-SEP-V-MM",
    ],
    "xing": [
        "LHCBEAM:IP1-XING-V-MURAD",
        "LHCBEAM:IP1-XING-H-MURAD",
        "LHCBEAM:IP5-XING-V-MURAD",
        "LHCBEAM:IP5-XING-H-MURAD",
        "LHCBEAM:IP2-XING-V-MURAD",
        "LHCBEAM:IP2-XING-H-MURAD",
        "LHCBEAM:IP8-XING-V-MURAD",
        "LHCBEAM:IP8-XING-H-MURAD",
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
        # hint: knobs for the other planes do not exist
    ],
    "disp": [
        "LHCBEAM:IP1-SDISP-CORR-SEP",
        "LHCBEAM:IP1-SDISP-CORR-XING",
        "LHCBEAM:IP5-SDISP-CORR-SEP",
        "LHCBEAM:IP5-SDISP-CORR-XING",
        # hint: these knobs do not exist for IP2 and IP8
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
def main(opt) -> tfs.TfsDataFrame:
    """ Main knob extracting function. """
    ldb = pytimber.LoggingDB(source="nxcals", loglevel=logging.ERROR)
    time = _parse_time(opt.time, opt.timedelta)

    if opt.state:
        # Only print the state of the machine.
        state_dict = get_state(ldb, time)
        return _get_state_as_df(state_dict, time)

    # Actually extract knobs.
    knobs_dict = _parse_knobs_defintions(opt.knob_definitions)
    knobs_extract = _extract_and_gather(ldb, knobs_dict, opt.knobs, time)
    if opt.output:
        _write_knobsfile(opt.output, knobs_extract)
    return knobs_extract


# State Extraction -------------------------------------------------------------

def get_state(ldb, time: datetime) -> Dict[str, str]:
    """
    Standalone function to gather and log state data from  the StateTracker.

    Args:
        ldb (pytimber.LoggingDB): The pytimber database.
        time (datetime): The time, when to get the state.

    Returns:
        Dict[str, str]: Dictionary of state-variable and the extracted state value.
    """
    state_dict = {}
    LOGGER.info(f"---- STATE @ {time} ----")
    for variable, name in STATE_VARIABLES.items():
        tracker_variable = f"LhcStateTracker:State:{variable}"
        state = ldb.get(tracker_variable, time.timestamp())[tracker_variable][1][-1]
        LOGGER.info(f"{f'{name}:':<13s} {state}")
        state_dict[variable] = state
    return state_dict


def _get_state_as_df(state_dict: Dict[str, str], time: datetime) -> tfs.TfsDataFrame:
    """
    Convert extracted StateTracker state-data into a TfsDataFrame
    To be consistent with the main functions output.

    Args:
        state_dict (Dict[str, str]): Extracted State Dictionary
        time (datetime): The time, when to get the state.

    Returns:
        tfs.DataFrame: States packed into dataframe with readable index.
    """
    state_df = tfs.TfsDataFrame(index=list(STATE_VARIABLES.values()),
                                columns=[Col.value, Col.lsa],
                                headers={Head.time: time})
    for name, value in state_dict.items():
        state_df.loc[STATE_VARIABLES[name], Col.lsa] = name
        state_df.loc[STATE_VARIABLES[name], Col.value] = value
    return state_df


# Knobs Extraction -------------------------------------------------------------

def extract(ldb, knobs: Sequence[str], time: datetime) -> Dict[str, float]:
    """
    Standalone function to gather data from  the StateTracker.
    Extracts data via pytimber's LoggingDB for the knobs given
    (either by name or by category) in knobs.

    Args:
        ldb (pytimber.LoggingDB): The pytimber database.
        knobs (Sequence[str]): Knob Categories or Knob-Names to extract.
        time (datetime): The time, when to extract.

    Returns:
        Dict[str, float]: Contains all the extracted knobs.
        When extraction was not possible, the value is None.

    """
    LOGGER.info(f"---- EXTRACTING KNOBS @ {time} ----")
    knobs_extracted = {}

    for category in knobs:
        for knob in KNOB_CATEGORIES.get(category, [category]):
            knobkey = f"LhcStateTracker:{knob}:target"
            knobs_extracted[knob] = None  # to log that this was tried to be extracted.

            knobvalue = ldb.get(knobkey, time.timestamp())  # use timestamp to preserve timezone info
            if knobkey not in knobvalue:
                LOGGER.debug(f"{knob} not found in StateTracker")
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
            knobs_extracted[knob] = value

    return knobs_extracted


def check_for_undefined_knobs(knobs_definitions: pd.DataFrame, knob_categories: Sequence[str]):
    """ Check that all knobs are actually defined in the knobs-definitions.


    Args:
        knobs_definitions (pd.DataFrame): A mapping of all knob-names to KnobEntries.
        knob_categories (Sequence[str]): Knob Categories or Knob-Names to extract.

    Raises:
        KeyError: If one or more of the knobs don't have a definition.

    """
    knob_names = [knob for category in knob_categories for knob in KNOB_CATEGORIES.get(category, [category])]
    undefined_knobs = [knob for knob in knob_names if knob not in knobs_definitions.index]
    if undefined_knobs:
        raise KeyError(
            "The following knob(s) could not be found "
            f"in the knob-definitions: '{', '.join(undefined_knobs)}'"
        )


def _extract_and_gather(ldb, knobs_definitions: pd.DataFrame,
                        knob_categories: Sequence[str],
                        time: datetime) -> tfs.TfsDataFrame:
    """
    Main function to gather data from the StateTracker and the knob-definitions.
    All given knobs (either in categories or as knob names) to be extracted
    are checked for being present in the ``knob_definitions``.
    A TfsDataFrame is returned, containing the knob-definitions of the
    requested knobs and the extracted value (or NAN if not successful).

    Args:
        ldb (pytimber.LoggingDB): The pytimber database.
        knobs_definitions (pd.DataFrame): A mapping of all knob-names to KnobEntries.
        knob_categories (Sequence[str]): Knob Categories or Knob-Names to extract.
        time (datetime): The time, when to extract.

    Returns:
        tfs.TfsDataframe: Contains all the extracted knobs, in columns containing
        their madx-name, lsa-name, scaling and extracted value.
        When extraction was not possible, the value of the respective entry is NAN.

    """
    check_for_undefined_knobs(knobs_definitions, knob_categories)
    extracted_knobs = extract(ldb, knobs=knob_categories, time=time)

    knob_names = list(extracted_knobs.keys())
    knobs = tfs.TfsDataFrame(index=knob_names,
                             columns=[Col.lsa, Col.madx, Col.scaling, Col.value],
                             headers={Head.time: time})
    knobs[[Col.lsa, Col.madx, Col.scaling]] = knobs_definitions.loc[knob_names, :]
    knobs[Col.value] = pd.Series(extracted_knobs)
    return knobs


def _write_knobsfile(output: Union[Path, str], collected_knobs: tfs.TfsDataFrame):
    """ Takes the collected knobs and writes them out into a text-file. """
    collected_knobs = collected_knobs.copy()  # to not modify the df

    # Sort the knobs by category
    category_knobs = {}
    for category, category_names in KNOB_CATEGORIES.items():
        names = [name for name in collected_knobs.index if name in category_names]
        if not names:
            continue

        category_knobs[category] = collected_knobs.loc[names, :]
        collected_knobs = collected_knobs.drop(index=names)

    if len(collected_knobs):  # leftover knobs without category
        category_knobs["Other Knobs"] = collected_knobs

    # Write them out
    with open(output, "w") as outfile:
        outfile.write(f"!! --- knobs extracted by knob_extractor\n")
        outfile.write(f"!! --- extracted knobs for time {collected_knobs.headers[Head.time]}\n\n")
        for category, knobs_df in category_knobs.items():
            outfile.write(f"!! --- {category:10} --------------------\n")
            for knob, knob_entry in knobs_df.iterrows():
                outfile.write(f"{get_madx_command(knob_entry)}\n")
            outfile.write("\n")
        outfile.write("\n")


# Knobs Definitions ------------------------------------------------------------

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


def load_knobs_definitions(file_path: Union[Path, str]) -> pd.DataFrame:
    """ Load the knobs-definition file and convert into a DataFrame.
    Each line in this file should consist of at least three comma separated
    entries in the following order: madx-name, lsa-name, scaling factor.
    Other columns are ignored.
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
        converters = {Col.madx: str.strip, Col.lsa: str.strip}  # strip whitespaces
        dtypes = {Col.scaling: float}
        names = (Col.madx, Col.lsa, Col.scaling)
        df = pd.read_csv(file_path,
                         comment="#",
                         usecols=list(range(len(names))),  # only read the first columns
                         names=names,
                         dtype=dtypes,
                         converters=converters)
    return _to_knobs_dataframe(df)


def _to_knobs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """ Adapts a DataFrame to the conventions used here:
    StateTracker variable name as index, all columns lower-case.

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
    return load_knobs_definitions(knobs_def_file)


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
