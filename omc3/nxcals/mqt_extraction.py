"""
Extraction of MQT (Quadrupole Trim) knob values from NXCALS.
------------------------------------------------------------

This module provides functions to retrieve MQT knob values for the LHC for a specified beam
and time using NXCALS and LSA.

**Arguments:**

*--Required--*

- **time** *(datetime|str)*:

    The timestamp for which to retrieve the data (timezone-aware recommended). If a string is provided, it will be parsed to a datetime object, assuming UTC timezone.

- **beam** *(int)*:

    The beam number (1 or 2).

- **output_dir** *(str|Path)*:
    Path to the output directory where the MQT knob values will be saved in.
"""

import logging
from datetime import datetime
from pathlib import Path

from generic_parser import DotDict, EntryPointParameters, entrypoint

# import jpype
from pyspark.sql import SparkSession

from omc3.nxcals.constants import EXTRACTED_MQTS_FILENAME
from omc3.nxcals.knob_extraction import NXCalResult, get_knob_vals
from omc3.utils.iotools import DateOrStr, PathOrStr
from omc3.utils.mock import cern_network_import

spark_session_builder = cern_network_import("nxcals.spark_session_builder")

logger = logging.getLogger(__name__)


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
        output_dir={
            "type": PathOrStr,
            "help": "Path to the output directory where the MQT knob values will be saved in.",
        },
    )


@entrypoint(_get_params(), strict=True)
def retrieve_mqts(opt: DotDict) -> None:
    """
    Retrieve MQT (Quadrupole Trim) knob values from NXCALS for a specific time and beam,
    and save them to a file in the specified output directory.
    """
    spark = spark_session_builder.get_or_create()
    mqt_vals = get_mqt_vals(spark, opt.time, opt.beam)
    output_path = Path(opt.output_dir) / EXTRACTED_MQTS_FILENAME
    with output_path.open("w") as f:
        for result in mqt_vals:
            timestamp_str = f"{result.timestamp:%Y-%m-%d %H:%M:%S%z}"
            value_str = f"{result.value:.10E}".replace("E+", "E")
            f.write(
                f"{result.name:<15}= {value_str}; ! powerconverter: {result.pc_name} at {timestamp_str}\n"
            )


def get_mqts(beam: int) -> set[str]:
    """
    Generate the set of MAD-X MQT (Quadrupole Trim) variable names for a given beam.

    Args:
        beam (int): The beam number (1 or 2).

    Returns:
        set[str]: A set of MAD-X variable names for MQT magnets, e.g., 'kqt12.a12b1'.

    Raises:
        ValueError: If beam is not 1 or 2.

    Examples:
        >>> get_mqts(1)
        {'kqt12.a12b1', 'kqt12.a23b1', ..., 'kqtd.a81b1'}
    """
    if beam not in (1, 2):
        raise ValueError("Beam must be 1 or 2")

    types = ["f", "d"]
    arcs = [12, 23, 34, 45, 56, 67, 78, 81]
    return {f"kqt{t}.a{a}b{beam}" for t in types for a in arcs}


def get_mqt_vals(spark: SparkSession, time: datetime, beam: int) -> list[NXCalResult]:
    """
    Retrieve MQT (Quadrupole Trim) knob values from NXCALS for a specific time and beam.

    This function queries NXCALS for current measurements of MQT power converters,
    calculates the corresponding K-values (integrated quadrupole strengths) using LSA,
    and returns them in MAD-X format with timestamps.

    Args:
        spark (SparkSession): Active Spark session for NXCALS queries.
        time (datetime): The timestamp for which to retrieve the data (timezone-aware recommended).
        beam (int): The beam number (1 or 2).

    Returns:
        list[NXCalResult]: List of NXCalResult objects containing the MAD-X knob names, K-values, and timestamps.

    Raises:
        ValueError: If beam is not 1 or 2 (propagated from get_mqts).
        RuntimeError: If no data is found in NXCALS or LSA calculations fail.
    """
    madx_mqts = get_mqts(beam)
    pattern = f"RPMBB.UA%.RQT%.A%B{beam}:I_MEAS"
    patterns = [pattern]
    return get_knob_vals(spark, time, beam, patterns, madx_mqts, "MQT: ")
