"""
MQT Extraction
---------------

Extract MQT (Quadrupole Trim) knob values from NXCALS for a specific time and beam.

**Arguments:**

*--Required--*

- **time** *(datetime|str)*:

    The timestamp for which to retrieve the data (timezone-aware recommended). If a string is provided, it will be parsed to a datetime object, assuming UTC timezone.

- **beam** *(int)*:

    The beam number (1 or 2).

*--Optional--*

- **output_path** *(str|Path)*:
    Path to the output file or directory. If a directory is given, the file will be named 'extracted_mqts.str'. If not provided, no output will be written.
"""

from pathlib import Path

from generic_parser import DotDict, EntryPointParameters, entrypoint

from omc3.nxcals.constants import EXTRACTED_MQTS_FILENAME
from omc3.nxcals.knob_extraction import NXCalResult
from omc3.nxcals.mqt_extraction import get_mqt_vals, knobs_to_madx
from omc3.utils.iotools import DateOrStr, PathOrStr
from omc3.utils.mock import cern_network_import

spark_session_builder = cern_network_import("nxcals.spark_session_builder")


def _get_params() -> EntryPointParameters:
    """
    Define the parameters for the MQT retrieval entry point.
    """
    return EntryPointParameters(
        time={
            "type": DateOrStr,
            "help": "The timestamp for which to retrieve the data (timezone-aware recommended).",
        },
        beam={"type": int, "help": "The beam number (1 or 2)."},
        output_path={
            "type": PathOrStr,
            "default": EXTRACTED_MQTS_FILENAME,
            "help": f"Path to the output file or directory. Defaults to '{EXTRACTED_MQTS_FILENAME}'. If a directory is given, the file will be named '{EXTRACTED_MQTS_FILENAME}'.",
        },
    )


@entrypoint(_get_params(), strict=True)
def retrieve_mqts(opt: DotDict) -> list[NXCalResult]:
    """
    Retrieve MQT (Quadrupole Trim) knob values from NXCALS for a specific time and beam,
    and optionally save them to a file.

    Returns:
        The list of retrieved MQT knob values.
    """
    spark = spark_session_builder.get_or_create()
    mqt_vals = get_mqt_vals(spark, opt.time, opt.beam)

    if opt.output_path is not None:
        output_path = Path(opt.output_path)
        if output_path.is_dir():
            output_path = output_path / EXTRACTED_MQTS_FILENAME
        with output_path.open("w") as f:
            f.write(knobs_to_madx(mqt_vals))

    return mqt_vals


if __name__ == "__main__":
    retrieve_mqts()
