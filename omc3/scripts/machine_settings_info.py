"""
Machine Settings Overview
-------------------------

Prints an overview over the machine settings at a provided given time, or the current settings if
no time is given.
If an output path is given, all info will be written into tfs files,
otherwise a summary is logged into console.

Knob values can be extracted and the knob definition gathered.
For brevity reasons, this data is not logged into the summary in the console.
If a start time is given, the trim history for the given knobs can be written out as well.
This data is also not logged.

Can be run from command line, parameters as given in :meth:`pylhc.machine_settings_info.get_info`.
All gathered data is returned, if this function is called from python.

.. code-block:: none

   usage: machine_settings_info.py [-h] [--time TIME] [--start_time START_TIME]
                                   [--knobs KNOBS [KNOBS ...]] [--accel ACCEL]
                                   [--beamprocess BEAMPROCESS] [--output_dir OUTPUT_DIR]
                                   [--knob_definitions] [--source SOURCE] [--log]

  optional arguments:
  -h, --help            show this help message and exit
  --time TIME           UTC Time as 'Y-m-d H:M:S.f' or ISO format or AccDatetime object.
                        Acts as point in time or end time (if ``start_time`` is given).
  --start_time START_TIME
                        UTC Time as 'Y-m-d H:M:S.f' or ISO format or AccDatetime object.
                        Defines the beginning of the time-range.
  --knobs KNOBS [KNOBS ...]
                        List of knobnames. If `None` (or omitted) no knobs will be extracted.
                        If it is just the string ``'all'``, all knobs will be extracted
                        (can be slow). Use the string ``'default'`` for pre-defined knobs
                        of interest.
  --accel ACCEL         Accelerator name.
  --beamprocess BEAMPROCESS
                        Manual override for the Beamprocess
                        (otherwise taken at the given ``time``)
  --output_dir OUTPUT_DIR
                        Output directory.
  --knob_definitions    Set to extract knob definitions.
  --source SOURCE       Source to extract data from.
  --log                 Write summary into log (automatically done if no output path is given).

"""
from __future__ import annotations
from attr import dataclass

from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from generic_parser import EntryPointParameters, entrypoint

from omc3.knob_extractor import KNOB_CATEGORIES, name2lsa
from omc3.nxcals.lsa_utils import (
    BeamProcessInfo,
    KnobDefinition,
    OpticsInfo,
    get_beamprocess_with_fill_at_time,
    get_knob_definition,
    get_last_trim,
    get_optics_for_beamprocess_at_time,
    get_trim_history, FillInfo,
)
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr
from omc3.utils.mock import cern_network_import
from omc3.utils.time_tools import parse_time

spark_session_builder = cern_network_import("nxcals.spark_session_builder")
pjlsa: object = cern_network_import("pjlsa")


if TYPE_CHECKING:
    from pjlsa import LSAClient
    from pjlsa._pjlsa import TrimTuple
    from pyspark.sql import SparkSession


LOGGER = logging_tools.get_logger(__name__)


# Main #########################################################################

@dataclass
class MachineSettingsInfo:
    """Dataclass for Machine Settings Info."""
    time: datetime
    accelerator: str
    fill: FillInfo | None = None
    beamprocess: BeamProcessInfo | None = None
    optics: OpticsInfo | None = None
    trim_histories: dict[str, TrimTuple] | None = None
    trims: dict[str, float] | None = None
    knob_definitions: dict[str, KnobDefinition] | None = None


def _get_params() -> dict:
    """Parse Commandline Arguments and return them as options."""
    return EntryPointParameters(
        time={
            "type": str,
            "help": (
                "At what time to extract the data. "
                "Accepts ISO-format (YYYY-MM-DDThh:mm:ss) with timezone, timestamp or 'now'. "
                "Timezone must be specified for ISO-format (e.g. +00:00 for UTC)."
            ),
            "default": "now",
        },
        timedelta={
            "type": str,
            "help": (
                "Add this timedelta to the given time. "
                "The format of timedelta is '((\\d+)(\\w))+' "
                "with the second token being one of "
                "s(seconds), m(minutes), h(hours), d(days), w(weeks), M(months) "
                "e.g 7m = 7 minutes, 1d = 1day, 7m30s = 7 min 30 secs. "
                "A prefix '_' specifies a negative timedelta. "
                "This allows for easily getting the setting "
                "e.g. 2h ago: '_2h' while setting the `time` argument to 'now' (default)."
            ),
        },
        delta_days={
            "type": float,
            "help": "Number of days to look back for data in NXCALS.",
            "default": 0.25,
        },
        knobs={
            "default": None,
            "nargs": "+",
            "type": str,
            "help": "List of knobnames. "
            "If `None` (or omitted) no knobs will be extracted. "
            "If it is just the string ``'all'``, "
            "all knobs will be extracted (can be slow). "
            "Use the string ``'default'`` for pre-defined knobs of interest.",
        },
        accel={"default": "lhc", "type": str, "help": "Accelerator name."},
        output_dir={"default": None, "type": PathOrStr, "help": "Output directory."},
        knob_definitions={"action": "store_true", "help": "Set to extract knob definitions."},
        log={
            "action": "store_true",
            "help": "Write summary into log (automatically done if no output path is given).",
        },
    )


@entrypoint(_get_params(), strict=True)
def get_info(opt) -> MachineSettingsInfo:
    """
     Get info about **Beamprocess**, **Optics** and **Knobs** at given time.


     Returns:
         dict: Dictionary containing the given ``time`` and ``start_time``,
         the extracted ``beamprocess``-info and ``optics``-info, the
         ``trim_histories`` and current (i.e. at given ``time``) ``trims``
         and the ``knob_definitions``, if extracted.

    """
    spark, lsa_client = _get_clients()
    time = parse_time(opt.time, opt.timedelta)

    machine_info = MachineSettingsInfo(time=time, accelerator=opt.accel)

    # BeamProcess and Fill ---
    fill, beamprocess = get_beamprocess_with_fill_at_time(
        lsa_client=lsa_client,
        spark=spark,
        time=time,
        accelerator=opt.accel,
    )
    machine_info.beamprocess = beamprocess
    machine_info.fill = fill

    # Optics ---
    machine_info.optics = _get_optics(
        lsa_client=lsa_client,
        time=time,
        beamprocess=machine_info.beamprocess,
    )

    # Knobs ---
    if opt.knobs is not None:
        machine_info.trim_histories = _get_trim_history(
            lsa_client=lsa_client,
            knobs=opt.knobs,
            time=time,
            delta_days=opt.delta_days,
            beamprocess_info=machine_info.beamprocess,
        )
        if machine_info.trim_histories:
            machine_info.trims = get_last_trim(machine_info.trim_histories)

    if opt.knob_definitions:
        machine_info.knob_definitions = _get_knob_definitions(
            lsa_client=lsa_client,
            machine_info=machine_info,
        )

    # Output ---
    if opt.log:
        _log_info(machine_info)

    if opt.output_dir is not None:
        _write_output(opt.output_dir, machine_info)

    return machine_info







# Output #######################################################################


def _log_info(machine_info: MachineSettingsInfo) -> None:
    """Log a summary of the extracted info to console (Info-Level).

    Args:
        machine_info (MachineSettingsInfo): Extracted Machine Settings Info
    """
    summary = (
        "\n----------- Summary ---------------------\n"
        f"Accelerator:  {machine_info.accelerator}\n"
        f"Given Time:   {machine_info.time.isoformat()}\n"
        f"Fill:         {machine_info.fill.no:d}\n"
        f"Beamprocess:  {machine_info.beamprocess.name}\n"
        f"  Start:      {machine_info.beamprocess.start_time.isoformat()}\n"
        f"  Context:    {machine_info.beamprocess.context_category}\n"
        f"  Descr.:     {machine_info.beamprocess.description}\n"
    )

    if machine_info.optics is not None:
        summary += (
            f"Optics:       {machine_info.optics.name}\n"
            f"  Start:      {machine_info.optics.start_time.isoformat()}\n"
        )

    if machine_info.trims is not None:
        summary += "----------- Trims -----------------------\n"
        for trim, value in machine_info.trims.items():
            summary += f"{trim:30s}: {value:g}\n"

    summary += "-----------------------------------------\n\n"
    LOGGER.info(summary)


def _write_output(output_dir: Path | str, machine_info: MachineSettingsInfo) -> None:
    """Write all extracted info into files.

    Args:
        output_dir (Path | str): Output Directory
        machine_info (MachineSettingsInfo): Machine Settings Info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Re-Implement output writing functions



# def write_summary(
#     output_path: Path,
#     accel: str,
#     acc_time: AccDatetime,
#     bp_info: DotDict,
#     optics_info: DotDict = None,
#     trims: dict[str, float] = None,
# ):
#     """Write summary into a ``tfs`` file.

#     Args:
#         output_path (Path): Folder to write output file into
#         accel (str): Name of the accelerator
#         acc_time (AccDatetime): User given Time
#         bp_info (DotDict): BeamProcess Info Dictionary
#         optics_info (DotDict): Optics Info Dictionary
#         trims (dict): Trims key-value dictionary
#     """
#     if trims is not None:
#         trims = trims.items()

#     info_tfs = tfs.TfsDataFrame(trims, columns=[const.column_knob, const.column_value])
#     info_tfs.headers = OrderedDict(
#         [
#             ("Hint:", "All times given in UTC."),
#             (const.head_accel, accel),
#             (const.head_time, acc_time.cern_utc_string()),
#             (const.head_beamprocess, bp_info.Name),
#             (const.head_fill, bp_info.Fill),
#             (const.head_beamprocess_start, bp_info.StartTime.cern_utc_string()),
#             (const.head_context_category, bp_info.ContextCategory),
#             (const.head_beamprcess_description, bp_info.Description),
#         ]
#     )
#     if optics_info is not None:
#         info_tfs.headers.update(
#             OrderedDict(
#                 [
#                     (const.head_optics, optics_info.Name),
#                     (const.head_optics_start, optics_info.StartTime.cern_utc_string()),
#                 ]
#             )
#         )
#     tfs.write(output_path / const.info_name, info_tfs)


# def write_knob_defitions(output_path: Path, definitions: dict):
#     """Write Knob definitions into a **tfs** file."""
#     for knob, definition in definitions.items():
#         path = output_path / f"{knob.replace('/', '_')}{const.knobdef_suffix}"
#         tfs.write(path, definition, save_index=LSA_COLUMN_NAME)


# def write_trim_histories(
#     output_path: Path,
#     trim_histories: dict[str, TrimTuple],
#     accel: str,
#     acc_time: AccDatetime = None,
#     acc_start_time: AccDatetime = None,
#     bp_info: DotDict = None,
#     optics_info: DotDict = None,
# ):
#     """Write the trim histories into tfs files.
#     There are two time columns, one with timestamps as they are usually easier to handle
#     and one with the UTC-string, as they are more human-readable.

#     Args:
#         output_path (Path): Folder to write output file into
#         trim_histories (dict): trims histories as extracted via LSA.get_trim_history()
#         accel (str): Name of the accelerator
#         acc_time (AccDatetime): User given (End)Time
#         acc_start_time (AccDatetime): User given Start Time
#         bp_info (DotDict): BeamProcess Info Dictionary
#         optics_info (DotDict): Optics Info Dictionary
#     """
#     AccDT = AcceleratorDatetime[accel]  # noqa: N806

#     # Create headers with basic info ---
#     headers = OrderedDict([("Hint:", "All times are given in UTC."), (const.head_accel, accel)])

#     if acc_start_time:
#         headers.update({const.head_start_time: acc_start_time.cern_utc_string()})

#     if acc_time:
#         headers.update({const.head_end_time: acc_time.cern_utc_string()})

#     if bp_info:
#         headers.update(
#             {
#                 const.head_beamprocess: bp_info.Name,
#                 const.head_fill: bp_info.Fill,
#             }
#         )

#     if optics_info:
#         headers.update({const.head_optics: optics_info.Name})

#     # Write trim history per knob ----
#     for knob, trim_history in trim_histories.items():
#         trims_tfs = tfs.TfsDataFrame(
#             headers=headers, columns=[const.column_time, const.column_timestamp, const.column_value]
#         )
#         for timestamp, value in zip(trim_history.time, trim_history.data):
#             time = AccDT.from_timestamp(timestamp).cern_utc_string()
#             try:
#                 len(value)
#             except TypeError:
#                 # single value (as it should be)
#                 trims_tfs.loc[len(trims_tfs), :] = (time, timestamp, value)
#             else:
#                 # multiple values (probably weird)
#                 LOGGER.debug("Multiple values in trim for {knob} at {time}.")
#                 for item in value:
#                     trims_tfs.loc[len(trims_tfs), :] = (time, timestamp, item)

#         path = output_path / f"{knob.replace('/', '_')}{const.trimhistory_suffix}"
#         tfs.write(path, trims_tfs)


# Clients #####################################################################

def _get_clients() -> tuple[SparkSession, LSAClient]:
    """Initialize and return SparkSession and LSAClient."""

    # Set log level to WARNING to avoid too much logging from Spark/LSA ---
    log_level = logging_tools.WARNING
    log_level_str = "WARN"

    with logging_tools.change_log_level(log_level):
        spark: SparkSession = spark_session_builder.get_or_create(
            conf={"spark.ui.showConsoleProgress": "false"}
        )
        spark.sparkContext.setLogLevel(log_level_str)
        lsa_client: LSAClient = pjlsa.LSAClient()
        logging_tools.getLogger("py4j").setLevel(log_level)

    return spark, lsa_client

# Optics #######################################################################

def _get_optics(
    lsa_client: LSAClient,
    time: datetime,
    beamprocess: BeamProcessInfo,
    ) -> OpticsInfo | None:
    """Get Optics Info at given time.

    Args:
        lsa_client (LSAClient): LSA Client
        time (datetime): Given Time
        beamprocess (BeamProcessInfo): Beamprocess Info

    Returns:
        OpticsInfo | None: Optics Info Dictionary or None if no optics found
    """
    try:
        return get_optics_for_beamprocess_at_time(
            lsa_client=lsa_client,
            time=time,
            beamprocess=beamprocess,
        )
    except ValueError as e:
        LOGGER.error(str(e))
        return None

# Knobs ########################################################################


def _get_trim_history(
    lsa_client: LSAClient,
    knobs: list[str],
    time: datetime,
    delta_days: float,
    beamprocess_info: BeamProcessInfo,
    ) -> dict[str, TrimTuple]:
    """Get Trim Histories for given knobs."""
    if len(knobs) == 1:
        match knobs[0].lower():
            case "all":
                knobs = []  # will extract all knobs in get_trim_history
            case "default":
                knobs  = [name2lsa(knob) for category in KNOB_CATEGORIES.values() for knob in category]
            case _:
                pass  # use given knob as is

    return get_trim_history(
        lsa_client=lsa_client,
        beamprocess=beamprocess_info.name,
        knobs=knobs,
        start_time=time - timedelta(days=delta_days),
        end_time=time,
        accelerator=beamprocess_info.accelerator,
    )


def _get_knob_definitions(lsa_client: LSAClient, machine_info: MachineSettingsInfo) -> dict[str, KnobDefinition] | None:
    """Get knob definitions."""
    if not machine_info.optics:
        LOGGER.error("Knob defintions requested, but no optics found.")
        return None

    if not machine_info.trim_histories:
        LOGGER.error("Knob definitions requested, but no trims extracted.")
        return None

    LOGGER.debug("Extracting knob definitions.")
    optics = machine_info.optics.name
    defs = {}
    for knob in machine_info.trim_histories:
        try:
            defs[knob] = get_knob_definition(lsa_client, knob, optics)
        except ValueError as e:
            LOGGER.error(e.args[0])
    return defs


# Script Mode ##################################################################


if __name__ == "__main__":
    get_info()
