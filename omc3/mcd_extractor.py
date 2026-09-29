r"""
MCD Extractor
-------------

Entrypoint to extract MCD (arc decapole corrector, "b5 corrector") knob values from ``NXCALS``
at a given time, and eventually write them out to a file.
The data is fetched from ``NXCALS`` through PySpark queries and converted to K-values using ``LSA``.

MCD magnets are decapole correctors located in the LHC arcs, used e.g. to correct the
f_jklm resonance driving terms driven by the arc decapole (b5) error. There is one MCD
circuit per arc (no focusing/defocusing split, unlike MQT), resulting in 8 MCD circuits
per beam.

.. note::
    Please note that access to the GPN is required to use this functionality.

**Arguments:**

*--Required--*

- **beam** *(int)*:

    The beam number (1 or 2).


- **time** *(str)*:

    At what time to extract the MCD knobs. Accepts ISO-format (YYYY-MM-
    DDThh:mm:ss) with timezone, timestamp or 'now'. Timezone must be
    specified for ISO-format (e.g. +00:00 for UTC).

    default: ``now``


*--Optional--*

- **output** *(PathOrStr)*:

    Specify user-defined output path. This should probably be
    `model_dir/mcds.madx`. If not provided, the data is only returned
    and not written to a file.

    default: ``None``


- **timedelta** *(str)*:

    Add this timedelta to the given time. The format of timedelta is
    '((\d+)(\w))+' with the second token being one of s(seconds),
    m(minutes), h(hours), d(days), w(weeks), M(months) e.g 7m = 7 minutes,
    1d = 1day, 7m30s = 7 min 30 secs. A prefix '_' specifies a negative
    timedelta. This allows for easily getting the setting e.g. 2h ago:
    '_2h' while setting the `time` argument to 'now' (default).

    default: ``None``


- **data_retrieval_days** *(float)*:

    Number of days to look back for data in NXCALS. Will always take the latest available data within this window.

    default: ``0.25`` (e.g. 6 hours)


"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from generic_parser import EntryPointParameters, entrypoint

from omc3.machine_data_extraction.mcd_extraction import get_mcd_vals
from omc3.machine_data_extraction.nxcals_knobs import NXCALSResult
from omc3.utils.iotools import PathOrStr
from omc3.utils.logging_tools import get_logger
from omc3.utils.mock import cern_network_import
from omc3.utils.time_tools import parse_time

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd


spark_session_builder = cern_network_import("nxcals.spark_session_builder")

LOGGER = get_logger(__name__)

USAGE_EXAMPLES = """Usage Examples:

python -m omc3.mcd_extractor --beam 1 --time 2022-04-24T16:10:00+00:00
    extracts the MCD knobs for beam 1 at 16:10 UTC on 2022-04-24

python -m omc3.mcd_extractor --beam 2 --time now --timedelta _2h
    extracts the MCD knobs for beam 2 as of 2 hours ago

python -m omc3.mcd_extractor --beam 1 --time now --output model_dir/mcds.madx
    extracts the current MCD settings for beam 1 and writes to file
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
                "At what time to extract the MCD knobs. "
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
        output={
            "type": PathOrStr,
            "help": (
                "Specify user-defined output path. This should probably be `model_dir/mcds.madx`"
            ),
        },
        data_retrieval_days={
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
        "prog": "MCD Extraction Tool.",
    },
)
def main(opt) -> pd.DataFrame:
    """
    Main MCD extracting function.

    Retrieves MCD (arc decapole corrector) knob values from NXCALS for a specific time and
    beam, and optionally saves them to a file.

    Returns:
        tfs.TfsDataFrame: DataFrame containing the extracted MCD knob values with columns
            for madx name, value, timestamp, and power converter name.
    """
    spark = spark_session_builder.get_or_create(conf={"spark.ui.showConsoleProgress": "false"})
    time = parse_time(opt.time, opt.timedelta)

    LOGGER.info(f"---- EXTRACTING MCD KNOBS @ {time.isoformat()} for Beam {opt.beam} ----")
    mcd_vals = get_mcd_vals(spark, time, opt.beam, data_retrieval_days=opt.data_retrieval_days)

    if opt.output:
        _write_mcd_file(opt.output, mcd_vals, time, opt.beam)

    return NXCALSResult.to_tfs(mcd_vals, time, opt.beam)


def _write_mcd_file(output: Path | str, mcd_vals: list[NXCALSResult], time: datetime, beam: int):
    """Write MCD knobs to a MAD-X file."""
    with Path(output).open("w") as outfile:
        outfile.write("!! --- MCD knobs extracted by mcd_extractor\n")
        outfile.write(f"!! --- extracted MCD knobs for time {time}, beam {beam}\n\n")
        outfile.write("\n".join([res.to_madx() for res in mcd_vals]))
        outfile.write("\n")


if __name__ == "__main__":
    main()
