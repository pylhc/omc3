"""
Extraction of MQT knobs from NXCALS.
------------------------------------

This module provides functions to retrieve MQT (Quadrupole Trim) knob values for the LHC
for a specified beam and time using NXCALS and LSA.

There are two types per arc (focusing 'f' and defocusing 'd'), resulting in 16 MQT circuits
per beam (8 arcs * 2 planes).

The extraction uses the underlying `knob_extraction` module with MQT-specific patterns.
"""

import logging
from datetime import datetime

# import jpype
from pyspark.sql import SparkSession

from omc3.nxcals.knob_extraction import NXCALSResult, get_knob_vals

logger = logging.getLogger(__name__)


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


def get_mqt_vals(
    spark: SparkSession, time: datetime, beam: int, delta_days: int = 1
) -> list[NXCALSResult]:
    """
    Retrieve MQT (Quadrupole Trim) knob values from NXCALS for a specific time and beam.

    This function queries NXCALS for current measurements of MQT power converters,
    calculates the corresponding K-values (integrated quadrupole strengths) using LSA,
    and returns them in MAD-X format with timestamps.

    Args:
        spark (SparkSession): Active Spark session for NXCALS queries.
        time (datetime): The timestamp for which to retrieve the data (timezone-aware recommended).
        beam (int): The beam number (1 or 2).
        delta_days (int): Number of days to look back for data. Default is 1.

    Returns:
        list[NXCalResult]: List of NXCalResult objects containing the MAD-X knob names, K-values, and timestamps.

    Raises:
        ValueError: If beam is not 1 or 2 (propagated from get_mqts).
        RuntimeError: If no data is found in NXCALS or LSA calculations fail.
    """
    madx_mqts = get_mqts(beam)
    pattern = f"RPMBB.UA%.RQT%.A%B{beam}:I_MEAS"
    patterns = [pattern]
    return get_knob_vals(spark, time, beam, patterns, madx_mqts, "MQT: ", delta_days)


def knobs_to_madx(mqt_vals: list[NXCALSResult]) -> str:
    """
    Convert a list of NXCalResult objects to a MAD-X script string.

    Args:
        mqt_vals: List of NXCalResult objects containing knob values.

    Returns:
        A string containing the MAD-X script with knob assignments.
    """
    lines = []
    for result in mqt_vals:
        timestamp_str = f"{result.timestamp:%Y-%m-%d %H:%M:%S%z}"
        value_str = f"{result.value:.10E}".replace("E+", "E")
        lines.append(
            f"{result.name:<15}= {value_str}; ! powerconverter: {result.pc_name} at {timestamp_str}\n"
        )
    return "".join(lines)
