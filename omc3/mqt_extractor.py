r"""
MQT Extractor
-------------

Entrypoint to extract MQT (Quadrupole Trim) knob values from ``NXCALS`` at a given time,
and eventually write them out to a file.
The data is fetched from ``NXCALS`` through PySpark queries and converted to K-values using ``LSA``.

MQT magnets are trim quadrupoles located in the LHC arcs, used for fine-tuning the optics.
There are two types per arc (focusing 'f' and defocusing 'd'), resulting in 16 MQT magnets
per beam (8 arcs x 2 types).

.. note::
    Please note that access to the GPN is required to use this functionality.

**Arguments:**

*--Required--*

- **beam** *(int)*:

    The beam number (1 or 2).


- **time** *(str)*:

    At what time to extract the MQT knobs. Accepts ISO-format (YYYY-MM-
    DDThh:mm:ss), timestamp or 'now'. The default timezone for the ISO-
    format is local time, but you can force e.g. UTC by adding +00:00.

    default: ``now``


*--Optional--*

- **output** *(PathOrStr)*:

    Specify user-defined output path. This should probably be
    `model_dir/mqts.madx`


- **timedelta** *(str)*:

    Add this timedelta to the given time. The format of timedelta is
    '((\d+)(\w))+' with the second token being one of s(seconds),
    m(minutes), h(hours), d(days), w(weeks), M(months) e.g 7m = 7 minutes,
    1d = 1day, 7m30s = 7 min 30 secs. A prefix '_' specifies a negative
    timedelta. This allows for easily getting the setting e.g. 2h ago:
    '_2h' while setting the `time` argument to 'now' (default).


- **delta_days** *(float)*:

    Number of days to look back for data in NXCALS.

    default: ``0.25`` (e.g. 6 hours)

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from generic_parser import EntryPointParameters, entrypoint

from omc3.nxcals.mqt_extraction import get_mqt_vals, knobs_to_madx
from omc3.utils.iotools import PathOrStr
from omc3.utils.logging_tools import get_logger
from omc3.utils.mock import cern_network_import
from omc3.utils.time_tools import parse_time

if TYPE_CHECKING:
    from datetime import datetime

spark_session_builder = cern_network_import("nxcals.spark_session_builder")

LOGGER = get_logger(__name__)

USAGE_EXAMPLES = """Usage Examples:

python -m omc3.mqt_extractor --beam 1 --time 2022-05-04T14:00
    extracts the MQT knobs for beam 1 at 14h on May 4th 2022

python -m omc3.mqt_extractor --beam 2 --time now --timedelta _2h
    extracts the MQT knobs for beam 2 as of 2 hours ago

python -m omc3.mqt_extractor --beam 1 --time now --output model_dir/mqts.madx
    extracts the current MQT settings for beam 1 and writes to file
"""


def get_params():
    return EntryPointParameters(
        beam={
            "type": int,
            "help": "The beam number (1 or 2).",
        },
        time={
            "type": str,
            "help": (
                "At what time to extract the MQT knobs. "
                "Accepts ISO-format (YYYY-MM-DDThh:mm:ss), timestamp or 'now'. "
                "The default timezone for the ISO-format is local time, "
                "but you can force e.g. UTC by adding +00:00."
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
        output={
            "type": PathOrStr,
            "help": (
                "Specify user-defined output path. This should probably be `model_dir/mqts.madx`"
            ),
        },
        delta_days={
            "type": float,
            "help": "Number of days to look back for data in NXCALS.",
            "default": 0.25,
        },
    )


@entrypoint(
    get_params(),
    strict=True,
    argument_parser_args={
        "epilog": USAGE_EXAMPLES,
        "formatter_class": argparse.RawDescriptionHelpFormatter,
        "prog": "MQT Extraction Tool.",
    },
)
def main(opt) -> tfs.TfsDataFrame:
    """
    Main MQT extracting function.

    Retrieves MQT (Quadrupole Trim) knob values from NXCALS for a specific time and beam,
    and optionally saves them to a file.

    Returns:
        tfs.TfsDataFrame: DataFrame containing the extracted MQT knob values with columns
            for madx name, value, timestamp, and power converter name.
    """
    spark = spark_session_builder.get_or_create(conf={"spark.ui.showConsoleProgress": "false"})
    time = parse_time(opt.time, opt.timedelta)

    LOGGER.info(f"---- EXTRACTING MQT KNOBS @ {time} for Beam {opt.beam} ----")
    mqt_vals = get_mqt_vals(spark, time, opt.beam, delta_days=opt.delta_days)

    # Convert to TfsDataFrame for consistency with knob_extractor
    mqt_df = tfs.TfsDataFrame(
        index=[result.name for result in mqt_vals],
        columns=["madx", "value", "timestamp", "pc_name"],
        headers={"EXTRACTION_TIME": time, "BEAM": opt.beam},
    )
    for result in mqt_vals:
        mqt_df.loc[result.name, "madx"] = result.name
        mqt_df.loc[result.name, "value"] = result.value
        mqt_df.loc[result.name, "timestamp"] = result.timestamp
        mqt_df.loc[result.name, "pc_name"] = result.pc_name

    if opt.output:
        _write_mqt_file(opt.output, mqt_vals, time, opt.beam)

    return mqt_df


def _write_mqt_file(output: Path | str, mqt_vals, time: datetime, beam: int):
    """Write MQT knobs to a MAD-X file."""
    with Path(output).open("w") as outfile:
        outfile.write("!! --- MQT knobs extracted by mqt_extractor\n")
        outfile.write(f"!! --- extracted MQT knobs for time {time}, beam {beam}\n\n")
        outfile.write(knobs_to_madx(mqt_vals))


if __name__ == "__main__":
    main()
