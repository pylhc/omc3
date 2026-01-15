"""
NXCals Results to MAD-X Conversion
----------------------------------

This module provides functions to convert NXCALS knob extraction results into
a MAD-X script format for further processing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omc3.machine_data_extraction.utils import strip_i_meas
from omc3.utils.mock import cern_network_import

jpype = cern_network_import("jpype")
pjlsa = cern_network_import("pjlsa")


if TYPE_CHECKING:
    from pjlsa import LSAClient

    from omc3.machine_data_extraction.nxcals_knobs import NXCALSResult

LOGGER = logging.getLogger(__name__)


def knobs_to_madx(nxcals_results: list[NXCALSResult]) -> str:
    """
    Convert a list of NXCalResult objects to a MAD-X script string.

    Args:
        nxcals_results: List of NXCalResult objects containing knob values.

    Returns:
        A string containing the MAD-X script with knob assignments.
    """
    lines = []
    for result in nxcals_results:
        timestamp_str = f"{result.datetime:%Y-%m-%d %H:%M:%S%z}"
        value_str = f"{result.value:.10E}".replace("E+", "E")
        lines.append(
            f"{result.name:<15}= {value_str}; ! powerconverter: {result.pc_name} at {timestamp_str}\n"
        )
    return "".join(lines)


def map_pc_name_to_madx(pc_name: str) -> str:
    """
    Convert an LHC power converter name or circuit name to its corresponding MAD-X name.

    This function processes the input name by removing the ':I_MEAS' suffix if present,
    extracting the circuit name from full power converter names (starting with 'RPMBB' or 'RPL'),
    applying specific string replacements for MAD-X compatibility, and converting the result to lowercase.

    ..note::
        This function currently supports only some of the LHC power converters:
        Corrector bends (RCB), Sextupoles (RS), Quadrupoles (RQ), and Bends (RB).
        Additional logic may be required to handle other magnet types.

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

    # Convert to lowercase
    circuit_name = circuit_name.lower()

    # Apply MAD-X specific replacements
    if circuit_name.startswith("rcb"):
        return f"a{circuit_name[1:]}"
    if circuit_name.startswith("r"):
        return f"k{circuit_name[1:]}"

    LOGGER.warning(f"Power converter name {pc_name} does not start with an 'R'.")
    return circuit_name


def map_lsa_name_to_madx(lsa_client: LSAClient, name: str):
    """Returns the ``MAD-X`` name (Circuit/Knob) from the given circuit name in LSA."""
    logical_name = name.split("/")[0]
    slist = jpype.java.util.Collections.singletonList(  # python lists did not work (jdilly)
        logical_name
    )
    madx_name_map = lsa_client._deviceService.findMadStrengthNamesByLogicalNames(
        slist
    )  # returns a map
    madx_name = madx_name_map[logical_name]
    LOGGER.debug(f"Name conversion: {name} -> {logical_name} -> {madx_name}")
    return madx_name
