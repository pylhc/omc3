"""
Machine Settings Information
----------------------------

Prints an overview over the machine settings at a provided given time, or the current settings if
no time is given.
If an output path is given, all info will be written into tfs files,
a summary can also be logged into console.

Knob values can be extracted and the knob definition gathered.
For brevity reasons, this data is not logged into the summary in the console.
If a start time is given, the trim history for the given knobs can be written out as well.
This data is also not logged.

Can be run from command line, parameters as given in :meth:`omc3.machine_settings_info.get_info`.
All gathered data is returned, if this function is called from python.

.. code-block:: none

    usage: machine_settings_info.py [-h] [--time TIME] [--timedelta TIMEDELTA] [--delta_days DELTA_DAYS]
                                    [--knobs KNOBS [KNOBS ...]] [--accel ACCEL] [--output_dir OUTPUT_DIR]
                                    [--knob_definitions] [--log]

    options:
    -h, --help            show this help message and exit
    --time TIME           At what time to extract the data.
                          Accepts ISO-format (YYYY-MM-DDThh:mm:ss) with timezone, timestamp or 'now'.
                          Timezone must be specified for ISO-format (e.g. +00:00 for UTC).
    --timedelta TIMEDELTA
                            Add this timedelta to the given time. The format of timedelta is '((\\d+)(\\w))+' with the
                            second token being one of s(seconds), m(minutes), h(hours), d(days), w(weeks), M(months)
                            e.g 7m = 7 minutes, 1d = 1day, 7m30s = 7 min 30 secs.
                            A prefix '_' specifies a negative timedelta.
                            This allows for easily getting the setting
                            e.g. 2h ago: '_2h' while setting the `time` argument to 'now' (default).
    --delta_days DELTA_DAYS
                            Number of days to look back for data in NXCALS.
    --knobs KNOBS [KNOBS ...]
                            List of knobnames. If `None` (or omitted) no knobs will be extracted.
                            If it is just the string ``'all'``, all knobs will be extracted (can be slow).
                            Use the string ``'default'`` for pre-defined knobs of interest.
    --accel ACCEL         Accelerator name.
    --output_dir OUTPUT_DIR
                            Output directory.
    --knob_definitions    Set to extract knob definitions.
    --log                 Write summary into log.

"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from generic_parser import EntryPointParameters, entrypoint

from omc3.knob_extractor import KNOB_CATEGORIES, name2lsa
from omc3.machine_data_extraction.constants import (
    KNOB_DEFINITION_MADX,
    KNOB_DEFINITION_TFS,
    MSI_SUMMARY_FILENAME,
    TRIM_HISTORY_TFS,
)
from omc3.machine_data_extraction.constants import (
    MSISummaryColumn as Column,
)
from omc3.machine_data_extraction.constants import (
    MSISummaryHeader as Header,
)
from omc3.machine_data_extraction.data_classes import (
    MachineSettingsInfo,
    TrimHistories,
    TrimHistoryHeader,
)
from omc3.machine_data_extraction.lsa_beamprocesses import (
    BeamProcessInfo,
    get_beamprocess_with_fill_at_time,
)
from omc3.machine_data_extraction.lsa_knobs import (
    KnobDefinition,
    get_knob_definition,
    get_last_trim,
    get_trim_history,
)
from omc3.machine_data_extraction.lsa_optics import OpticsInfo, get_optics_for_beamprocess_at_time
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr
from omc3.utils.mock import cern_network_import
from omc3.utils.time_tools import parse_time

spark_session_builder = cern_network_import("nxcals.spark_session_builder")
pjlsa: object = cern_network_import("pjlsa")


if TYPE_CHECKING:
    from pjlsa import LSAClient
    from pyspark.sql import SparkSession


LOGGER = logging_tools.get_logger(__name__)


# Main #########################################################################

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
        accel={
            "default": "lhc",
            "type": str,
            "help": "Accelerator name."
        },
        output_dir={
            "default": None,
            "type": PathOrStr,
            "help": "Output directory."
        },
        knob_definitions={
            "action": "store_true",
            "help": "Set to extract knob definitions."
        },
        log={
            "action": "store_true",
            "help": "Write summary into log.",
        },
    )


@entrypoint(_get_params(), strict=True)
def get_info(opt) -> MachineSettingsInfo:
    """
     Get info about **Beamprocess**, **Optics** and **Knobs** at given time.

    **Arguments:**

    *--Optional--*

    - **accel** *(str)*:

        Accelerator name.

        default: ``lhc``


    - **delta_days** *(float)*:

        Number of days to look back for data in NXCALS.

        default: ``0.25``


    - **knob_definitions**:

        Set to extract knob definitions.

        action: ``store_true``


    - **knobs** *(str)*:

        List of knobnames. If `None` (or omitted) no knobs will be extracted.
        If it is just the string ``'all'``, all knobs will be extracted (can
        be slow). Use the string ``'default'`` for pre-defined knobs of
        interest.

        default: ``None``


    - **log**:

        Write summary into log (automatically done if no output path is
        given).

        action: ``store_true``


    - **output_dir** *(PathOrStr)*:

        Output directory.

        default: ``None``


    - **time** *(str)*:

        At what time to extract the data. Accepts ISO-format (YYYY-MM-
        DDThh:mm:ss) with timezone, timestamp or 'now'. Timezone must be
        specified for ISO-format (e.g. +00:00 for UTC).

        default: ``now``


    - **timedelta** *(str)*:

        Add this timedelta to the given time. The format of timedelta is
        '((\\d+)(\\w))+' with the second token being one of s(seconds),
        m(minutes), h(hours), d(days), w(weeks), M(months) e.g 7m = 7 minutes,
        1d = 1day, 7m30s = 7 min 30 secs. A prefix '_' specifies a negative
        timedelta. This allows for easily getting the setting e.g. 2h ago:
        '_2h' while setting the `time` argument to 'now' (default).

    Returns:
        MachineSettingsInfo: Extracted Machine Settings Info.
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
            machine_info.trims = get_last_trim(machine_info.trim_histories.trims)

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

    # Summary ---
    summary_df = _summary_df(machine_info)
    tfs.write(output_dir / MSI_SUMMARY_FILENAME, summary_df)

    # Knob Definitions ---
    if machine_info.knob_definitions is not None:
        for definition in machine_info.knob_definitions.values():
            value = 0.0 if machine_info.trims is None else machine_info.trims.get(definition.name, 0.0)

            madx = definition.to_madx(value=value)
            df = definition.to_tfs()

            (output_dir / f"{definition.output_name}{KNOB_DEFINITION_MADX}").write_text(madx)
            tfs.write(output_dir / f"{definition.output_name}{KNOB_DEFINITION_TFS}", df)

    # Trim Histories ---
    if machine_info.trim_histories is not None:
        if machine_info.optics is not None:
            machine_info.trim_histories.headers[TrimHistoryHeader.OPTICS] = machine_info.optics.name

        if machine_info.fill is not None:
            machine_info.trim_histories.headers[TrimHistoryHeader.FILL] = machine_info.fill.no

        for knob, trim_df in machine_info.trim_histories.to_tfs_dict().items():
            tfs.write(output_dir / f"{knob}{TRIM_HISTORY_TFS}", trim_df)


def _summary_df(machine_info: MachineSettingsInfo) -> tfs.TfsDataFrame:
    """Convert MachineSettingsInfo into a ``TfsDataFrame``.

    Args:
        machine_info (MachineSettingsInfo): Machine Settings Info.

    Returns:
        tfs.TfsDataFrame: Summary as TfsDataFrame.
    """
    trims = machine_info.trims.items() if machine_info.trims is not None else []

    info_tfs = tfs.TfsDataFrame(trims, columns=[Column.KNOB, Column.VALUE])
    info_tfs.headers =         {
            Header.ACCEL: machine_info.accelerator,
            Header.TIME: machine_info.time.isoformat(),
            Header.BEAMPROCESS: machine_info.beamprocess.name,
            Header.FILL: machine_info.fill.no,
            Header.BEAMPROCESS_START: machine_info.beamprocess.start_time.isoformat(),
            Header.CONTEXT_CATEGORY: machine_info.beamprocess.context_category,
            Header.BEAMPROCESS_DESCRIPTION: machine_info.beamprocess.description,
        }

    if machine_info.optics is not None:
        info_tfs.headers.update({
            Header.OPTICS: machine_info.optics.name,
            Header.OPTICS_START: machine_info.optics.start_time.isoformat(),
        })

    return info_tfs


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
    ) -> TrimHistories:
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
    for knob in machine_info.trim_histories.trims:
        try:
            defs[knob] = get_knob_definition(lsa_client, knob, optics)
        except ValueError as e:
            LOGGER.error(e.args[0])
    return defs


# Script Mode ##################################################################


if __name__ == "__main__":
    get_info()
