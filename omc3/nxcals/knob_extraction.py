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

from omc3.utils.mock import cern_network_import

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

jpype = cern_network_import("jpype")
pjlsa = cern_network_import("pjlsa")
builders = cern_network_import("nxcals.api.extraction.data.builders")
functions = cern_network_import("pyspark.sql.functions")
window = cern_network_import("pyspark.sql.window")

LOGGER = logging.getLogger(__name__)


@dataclass
class NXCALSResult:
    name: str
    value: float
    timestamp: pd.Timestamp
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
    k_values = calc_k_from_iref(lsa_client, currents, energy)

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
    timestamps = {map_pc_name_to_madx(var.name): var.timestamp for var in combined_vars}
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
    spark: SparkSession, time: datetime, var_name: str, delta_days: float = 0.25
) -> list[NXCALSResult]:
    """
    Retrieve raw variable values from NXCALS.

    Args:
        spark (SparkSession): Active Spark session.
        time (datetime): Python datetime (timezone-aware recommended).
        var_name (str): Name or pattern of the variable(s) to retrieve.
        delta_days (float): Number of days to look back for data. Default is 0.25.

    Returns:
        list[NXCalResult]: List of NXCalResult containing variable name, value, timestamp,
        and power converter name for the latest sample of each matching variable at the given time.

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

    # Get the latest sample for each variable
    window_spec = window.Window.partitionBy("nxcals_variable_name").orderBy(
        functions.col("nxcals_timestamp").desc()
    )
    latest_df = (
        df.withColumn("row_num", functions.row_number().over(window_spec))
        .filter(functions.col("row_num") == 1)
        .select("nxcals_variable_name", "nxcals_value", "nxcals_timestamp")
    )

    results = []
    for row in latest_df.collect():
        full_varname = row["nxcals_variable_name"]
        raw_val = float(row["nxcals_value"])
        ts = pd.to_datetime(row["nxcals_timestamp"], unit="ns", utc=True)
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


# LSA K-value Calculation ------------------------------------------------------


def calc_k_from_iref(lsa_client, currents: dict[str, float], energy: float) -> dict[str, float]:
    """
    Calculate K values in the LHC from IREF using the LSA service.

    Args:
        lsa_client: The LSA client instance.
        currents (dict[str, float]): Dictionary of current values keyed by variable name.
        energy (float): The beam energy in GeV.

    Returns:
        dict[str, float]: Dictionary of K values keyed by variable name.
    """
    # 1) Use the **instance** PJLSA already created
    lhc_service = lsa_client._lhcService

    # 2) Build a java.util.HashMap<String, Double> (not a Python dict)
    j_hash_map = jpype.JClass("java.util.HashMap")
    j_double = jpype.JClass("java.lang.Double")

    jmap = j_hash_map()
    for power_converter, current in currents.items():
        if current is None:
            raise ValueError(f"Current for {power_converter} is None")
        # ensure primitive-compatible double values
        jmap.put(power_converter, j_double.valueOf(current))  # boxed; Java unboxes internally

    # 3) Call: Map<String, Double> calculateKfromIREF(Map<String, Double>, double)
    out = lhc_service.calculateKfromIREF(jmap, float(energy))

    # 4) Convert java.util.Map -> Python dict
    res = {}
    it = out.entrySet().iterator()
    while it.hasNext():
        e = it.next()
        res[str(e.getKey())] = float(e.getValue())
    return res


# Utility Functions ------------------------------------------------------------


def strip_i_meas(text: str) -> str:
    """
    Remove the I_MEAS suffix from a variable name.

    Args:
        text (str): The variable name possibly ending with ':I_MEAS'.

    Returns:
        str: The variable name without the ':I_MEAS' suffix.
    """
    return text.removesuffix(":I_MEAS")


# Note: this will have to be updated if we ever want to support other magnet types
# such as dipoles, sextupoles, octupoles, etc.
def map_pc_name_to_madx(pc_name: str) -> str:
    """
    Convert an LHC power converter name or circuit name to its corresponding MAD-X name.

    This function processes the input name by removing the ':I_MEAS' suffix if present,
    extracting the circuit name from full power converter names (starting with 'RPMBB' or 'RPL'),
    applying specific string replacements for MAD-X compatibility, and converting the result to lowercase.

    Args:
        pc_name (str): The power converter or circuit name to convert.

    Returns:
        str: The converted MAD-X name.

    Examples:
        >>> map_pc_name_to_madx("RPMBB.UJ33.RCBH10.L1B1:I_MEAS")
        'acb10.l1b1'
        >>> map_pc_name_to_madx("RQT12.L5B1")
        'kqt12.l5b1'
    """
    # Remove the ':I_MEAS' suffix if it exists
    pc_name = strip_i_meas(pc_name)

    # Extract circuit name from full power converter names
    if "." in pc_name:
        parts = pc_name.split(".")
        # For full names like 'RPMBB.UJ33.RCBH10.L1B1', take from the third part onward, else use pc_name
        circuit_name = ".".join(parts[2:]) if len(parts) > 2 else pc_name
    else:
        circuit_name = pc_name

    # Apply MAD-X specific replacements
    replacements = {"RQ": "KQ", "RCB": "ACB"}
    for old, new in replacements.items():
        circuit_name = circuit_name.replace(old, new)
    return circuit_name.lower()
