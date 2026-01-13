"""
Knob Extraction
---------------

This module provides functionality to extract knob values from NXCALS for the LHC and
convert them to MAD-X compatible format using LSA services.

It handles retrieval of raw variable data from NXCALS, conversion of power converter
currents to K-values, and mapping of power converter names to MAD-X naming
conventions.

This module requires the installation of `jpype`, `pyspark`, and access to the
CERN network to connect to NXCALS and LSA services. You can install the required
packages via pip:
```
python -m pip install omc3[cern]
```

See the [NXCALS documentation](https://nxcals-docs.web.cern.ch/current/user-guide/data-access/quickstart/)
for more information on getting access and using the Python API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pandas as pd

from omc3.nxcals import lsa_utils
from omc3.nxcals.madx_conversion import map_pc_name_to_madx
from omc3.nxcals.utils import strip_i_meas
from omc3.utils.mock import cern_network_import

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

pjlsa = cern_network_import("pjlsa")
builders = cern_network_import("nxcals.api.extraction.data.builders")
functions = cern_network_import("pyspark.sql.functions")
window = cern_network_import("pyspark.sql.window")

LOGGER = logging.getLogger(__name__)

# Constants --------------------------------------------------------------------

VARIABLE_NAME: str = "nxcals_variable_name"
VALUE: str = "nxcals_value"
TIMESTAMP: str = "nxcals_timestamp"

@dataclass
class NXCALSResult:
    name: str
    value: float
    datetime: pd.Timestamp | datetime  # pd.Timestamp inherits from datetime
    pc_name: str


# High-level Knob Extraction ---------------------------------------------------


def get_knob_vals(
    spark: SparkSession,
    time: datetime,
    beam: int,
    patterns: list[str],
    expected_knobs: set[str] | None = None,
    log_prefix: str = "",
    delta_days: float = 0.25,
) -> list[NXCALSResult]:
    """
    Retrieve knob values for a given beam and time using specified patterns for the LHC.

    This is the main entry point for extracting magnet knob values from NXCALS. The function
    performs a complete workflow:

    1. Queries NXCALS for power converter current measurements (I_MEAS) using variable patterns
    2. Retrieves the beam energy at the specified time
    3. Converts currents to K-values (integrated quadrupole strengths) using LSA
    4. Maps power converter names to MAD-X naming conventions
    5. Returns knob values with their timestamps

    The difference between patterns and knob names:

    - **patterns**: NXCALS variable patterns (e.g., "RPMBB.UA%.RQT%.A%B1:I_MEAS") used to query
      raw power converter current measurements. These follow CERN naming conventions and may
      include wildcards (%).
    - **expected_knobs**: MAD-X element names (e.g., "kqt12.a12b1") representing the final
      knob names as used in MAD-X scripts. These are lowercase, simplified names.

    Args:
        spark (SparkSession): Active Spark session for NXCALS queries.
        time (datetime): Time to retrieve data for.
        beam (int): Beam number (1 or 2).
        patterns (list[str]): List of NXCALS variable patterns to query for power converter
            currents. Patterns can include SQL-like wildcards (%). Example:
            "RPMBB.UA%.RQT%.A%B1:I_MEAS" matches all MQT quadrupole trim magnets for beam 1.
        expected_knobs (set[str] | None): Set of expected MAD-X knob names to validate and filter
            results. If None, returns all found knobs without validation.
        log_prefix (str): Prefix for logging messages to distinguish different extraction runs.
        delta_days (float): Number of days to look back for data. Default is 0.25.

    Returns:
        list[NXCalResult]: List of NXCalResult objects containing MAD-X knob names, K-values,
            timestamps, and power converter names.
    """
    LOGGER.info(f"{log_prefix}Starting data retrieval for beam {beam} at time {time}")

    # Retrieve raw current measurements from NXCALS
    combined_vars: list[NXCALSResult] = []
    for pattern in patterns:
        LOGGER.info(f"{log_prefix}Getting currents for pattern {pattern} at {time}")
        raw_vars = get_raw_vars(spark, time, pattern, delta_days)
        combined_vars.extend(raw_vars)

    # Get beam energy for K-value calculations
    energy, _ = get_energy(spark, time)

    # Prepare currents dict for LSA
    currents = {strip_i_meas(var.name): var.value for var in combined_vars}

    LOGGER.info(
        f"{log_prefix}Calculating K values for {len(currents)} power converters with energy {energy} GeV"
    )

    # Calculate K values using LSA
    lsa_client = pjlsa.LSAClient()
    k_values = lsa_utils.calc_k_from_iref(lsa_client, currents, energy)

    # Transform K values keys to MAD-X format
    k_values_madx = {map_pc_name_to_madx(key): value for key, value in k_values.items()}

    found_keys = set(k_values_madx.keys())

    if expected_knobs is None:
        expected_knobs = found_keys

    if missing_keys := expected_knobs - found_keys:
        LOGGER.warning(f"{log_prefix}Missing K-values for knobs: {missing_keys}")

    if unknown_keys := found_keys - expected_knobs:
        LOGGER.warning(f"{log_prefix}Unknown K-values found: {unknown_keys}")

    # Build result list with timestamps
    timestamps = {map_pc_name_to_madx(var.name): var.datetime for var in combined_vars}
    pc_names = {map_pc_name_to_madx(var.name): var.pc_name for var in combined_vars}
    results = []
    for madx_name in expected_knobs:
        value = k_values_madx.get(madx_name)
        timestamp = timestamps.get(madx_name)
        pc_name = pc_names.get(madx_name)

        if value is not None and timestamp is not None and pc_name is not None:
            results.append(NXCALSResult(madx_name, value, timestamp, pc_name))
        else:
            LOGGER.warning(f"{log_prefix}Missing data for {madx_name}")

    return results


# NXCALS Data Retrieval --------------------------------------------------------


def get_raw_vars(
    spark: SparkSession, time: datetime, var_name: str, delta_days: float = 0.25, latest_only: bool = True,
) -> list[NXCALSResult]:
    """
    Retrieve raw variable values from NXCALS.

    Args:
        spark (SparkSession): Active Spark session.
        time (datetime): Python datetime (timezone-aware recommended).
        var_name (str): Name or pattern of the variable(s) to retrieve.
        delta_days (float): Number of days to look back for data. Default is 0.25.
        latest_only (bool): If True, only the latest sample for each variable is returned. Default is True.

    Returns:
        list[NXCalResult]: List of NXCalResult containing variable name, value, timestamp,
        and power converter name for the latest sample of each matching variable at the given time,
        or all samples if so required.

    Raises:
        RuntimeError: If no data is found for the variable in the given interval.
        You may need to increase the delta_days if necessary.
    """

    # Ensure time is in UTC
    if time.tzinfo is None:
        raise ValueError("Datetime object must be timezone-aware")
    time = time.astimezone(timezone.utc)

    # Look back delta_days, may need up to 1 day to find data
    start_time = time - timedelta(days=delta_days)
    end_time = time
    LOGGER.info(f"Retrieving raw variables {var_name} from {start_time} to {end_time}")

    df = (
        builders.DataQuery.builder(spark)
        .variables()
        .system("CMW")
        .nameLike(var_name)
        .timeWindow(start_time, end_time)
        .build()
    )

    # Avoid full count() – just check if we have at least one row
    if df is None or not df.take(1):
        raise RuntimeError(
            f"No data found for {var_name} in {start_time} to {end_time}. "
            f"You may need to increase the delta_days from {delta_days} days if necessary."
        )

    LOGGER.info(f"Raw variables {var_name} retrieved successfully.")


    if latest_only:
        # Get the latest sample for each variable
        window_spec = window.Window.partitionBy(VARIABLE_NAME).orderBy(
            functions.col(TIMESTAMP).desc()
        )
        df = (
            df.withColumn("row_num", functions.row_number().over(window_spec))
            .filter(functions.col("row_num") == 1)
        )

    results = []
    for row in df.select(VARIABLE_NAME, VALUE, TIMESTAMP).collect():
        full_varname = row[VARIABLE_NAME]
        raw_val = float(row[VALUE])
        ts = pd.to_datetime(row[TIMESTAMP], unit="ns", utc=True)
        results.append(NXCALSResult(full_varname, raw_val, ts, strip_i_meas(full_varname)))
        LOGGER.info(f"LHC value retrieved: {full_varname} = {raw_val:.2f} at {ts}")

    return results


def get_energy(spark: SparkSession, time: datetime) -> tuple[float, pd.Timestamp]:
    """
    Retrieve the beam energy of the LHC from NXCALS.

    Args:
        spark (SparkSession): Active Spark session.
        time (datetime): Python datetime (timezone-aware recommended).

    Returns:
        tuple[float, pd.Timestamp]: Beam energy in GeV and its timestamp.

    Raises:
        RuntimeError: If no energy data is found.
    """
    scale = 0.120
    raw_vars = get_raw_vars(spark, time, "HX:ENG")
    if not raw_vars:
        raise RuntimeError("No energy data found.")
    return raw_vars[0].value * scale, raw_vars[0].timestamp
