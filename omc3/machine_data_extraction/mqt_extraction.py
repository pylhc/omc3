"""
MQT knobs extraction
--------------------

This module provides functions to retrieve MQT (Quadrupole Trim) knob values for the LHC
for a specified beam and time using NXCALS and LSA.

There are two types per arc (focusing 'f' and defocusing 'd'), resulting in 16 MQT circuits
per beam (8 arcs * 2 planes).

The extraction uses the underlying `knob_extraction` module with MQT-specific patterns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omc3.machine_data_extraction.nxcals_knobs import NXCALSResult, get_knob_vals

if TYPE_CHECKING:
    from datetime import datetime

    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def generate_mqt_names(beam: int) -> set[str]:
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
    spark: SparkSession,
    time: datetime,
    beam: int,
    data_retrieval_days: float = 0.25,
    energy: float | None = None,
) -> list[NXCALSResult]:
    """
    Retrieve MQT (Quadrupole Trim) knob values from NXCALS for a specific time and beam.

    This function queries NXCALS for current measurements of MQT power converters,
    calculates the corresponding K-values (integrated quadrupole strengths) using LSA,
    and returns them in MAD-X format with timestamps.

    Args:
        spark (SparkSession): Active Spark session for NXCALS queries.
        time (datetime): The timestamp for which to retrieve the data (timezone-aware required).
        beam (int): The beam number (1 or 2).
        data_retrieval_days (float): Number of days to look back for data in NXCALS. Will always take the latest available data within this window.
            default: ``0.25`` (e.g. 6 hours)
        energy (float | None): Beam energy in GeV. If None, the energy is retrieved from the HX:ENG variable.

    Returns:
        list[NXCalResult]: List of NXCalResult objects containing the MAD-X knob names, K-values, and timestamps.

    Raises:
        ValueError: If beam is not 1 or 2 (propagated from get_mqts), or if time is not timezone-aware.
        RuntimeError: If no data is found in NXCALS or LSA calculations fail.
    """
    if time.tzinfo is None:
        raise ValueError("Time must be timezone-aware.")

    madx_mqts = generate_mqt_names(beam)
    pattern = f"RPMBB.UA%.RQT%.A%B{beam}:I_MEAS"
    patterns = [pattern]
    return get_knob_vals(spark, time, beam, patterns, madx_mqts, "MQT: ", data_retrieval_days)
