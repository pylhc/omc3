"""
MCD knobs extraction
---------------------

This module provides functions to retrieve MCD (arc decapole corrector, "b5 corrector") knob
values for the LHC for a specified beam and time using NXCALS and LSA.

There is one MCD circuit per arc (no focusing/defocusing split, unlike MQT), resulting in 8
MCD circuits per beam (8 arcs), each powering a series string of individual MCD magnets
spread around that arc.

The extraction uses the underlying `nxcals_knobs.get_knob_vals` function with MCD-specific
patterns, exactly as `mqt_extraction.get_mqt_vals` does for MQT.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omc3.machine_data_extraction.nxcals_knobs import NXCALSResult, get_knob_vals

if TYPE_CHECKING:
    from datetime import datetime

    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def generate_mcd_names(beam: int) -> set[str]:
    """
    Generate the set of MAD-X MCD (arc decapole corrector) variable names for a given beam.

    Args:
        beam (int): The beam number (1 or 2).

    Returns:
        set[str]: A set of MAD-X variable names for MCD circuits, e.g., 'kcd.a12b1'.

    Raises:
        ValueError: If beam is not 1 or 2.

    Examples:
        >>> generate_mcd_names(1)
        {'kcd.a12b1', 'kcd.a23b1', ..., 'kcd.a81b1'}
    """
    if beam not in (1, 2):
        raise ValueError("Beam must be 1 or 2")

    arcs = [12, 23, 34, 45, 56, 67, 78, 81]
    return {f"kcd.a{a}b{beam}" for a in arcs}


def get_mcd_vals(
    spark: SparkSession,
    time: datetime,
    beam: int,
    data_retrieval_days: float = 0.25,
    energy: float | None = None,
) -> list[NXCALSResult]:
    """
    Retrieve MCD (arc decapole corrector) knob values from NXCALS for a specific time and beam.

    This function queries NXCALS for current measurements of MCD power converters,
    calculates the corresponding K-values (integrated decapole strengths) using LSA,
    and returns them in MAD-X format with timestamps.

    Args:
        spark (SparkSession): Active Spark session for NXCALS queries.
        time (datetime): The timestamp for which to retrieve the data (timezone-aware required).
        beam (int): The beam number (1 or 2).
        data_retrieval_days (float): Number of days to look back for data in NXCALS. Will always take the latest available data within this window.
            default: ``0.25`` (e.g. 6 hours)
        energy (float | None): Beam energy in GeV. If None, the energy is retrieved from the HX:ENG variable.

    Returns:
        list[NXCALSResult]: List of NXCALSResult objects containing the MAD-X knob names, K-values, and timestamps.

    Raises:
        ValueError: If beam is not 1 or 2 (propagated from generate_mcd_names), or if time is not timezone-aware.
        RuntimeError: If no data is found in NXCALS or LSA calculations fail.
    """
    if time.tzinfo is None:
        raise ValueError("Time must be timezone-aware.")

    madx_mcds = generate_mcd_names(beam)
    pattern = f"RPMBB.UA%.RCD.A%B{beam}:I_MEAS"
    patterns = [pattern]
    return get_knob_vals(
        spark,
        time,
        beam,
        patterns,
        madx_mcds,
        "MCD: ",
        data_retrieval_days,
        energy,
    )
