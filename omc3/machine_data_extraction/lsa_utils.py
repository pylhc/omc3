"""
LSA Utilities
-------------

This module contains utility functions for working with LSA and data extracted from it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omc3.utils import logging_tools
from omc3.utils.mock import cern_network_import

jpype = cern_network_import("jpype")

if TYPE_CHECKING:
    from pjlsa import LSAClient


LOGGER = logging_tools.get_logger(__name__)


def calc_k_from_iref(
    lsa_client: LSAClient, currents: dict[str, float], energy: float
) -> dict[str, float]:
    """
    Calculate K values in the LHC from IREF using the LSA service.

    Args:
        lsa_client (LSAClient): The LSA client instance.
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
