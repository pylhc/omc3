"""
LSA Beamprocesses
-----------------

This module contains functions to extract beamprocess - and related - information
from LSA.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import dateutil.tz as tz
import numpy as np

from omc3.machine_data_extraction.data_classes import BeamProcessInfo, FillInfo
from omc3.machine_data_extraction.nxcals_knobs import NXCALSResult, get_raw_vars
from omc3.utils import logging_tools
from omc3.utils.mock import cern_network_import

pjlsa = cern_network_import("pjlsa")

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cern.lsa.domain.settings.spi import StandAloneBeamProcessImpl
    from pjlsa import LSAClient
    from pyspark.sql import SparkSession


LOGGER = logging_tools.get_logger(__name__)

RELEVANT_BP_CONTEXTS: tuple[str, ...] = ("OPERATIONAL", "MD")
RELEVANT_BP_CATEGORIES: tuple[str, ...] = ("DISCRETE",)
KNOBS_BP_GROUP: str = "POWERCONVERTERS"  # the Beamprocesses relevant for OMC
BP_CONTEXT_FAMILY: str = "beamprocess"

FILL_VARIABLE = "HX:FILLN"


def get_active_beamprocess_at_time(
    lsa_client: LSAClient,
    time: datetime,
    accelerator: str = "lhc",
    bp_group: str = KNOBS_BP_GROUP,
    ) -> BeamProcessInfo:
    """
    Find the active beam process at the time given.

    Note: This function has been derived from the original KnobExtractor implementation in the Java Online Model.

    Args:
        lsa_client (LSAClient): The LSA client instance.
        time (datetime): The time at which to find the active beam process.
        accelerator (str): Name of the accelerator.
        bp_group (str): BeamProcess Group, choices : 'POWERCONVERTERS', 'ADT', 'KICKERS', 'SPOOLS', 'COLLIMATORS'


    Returns:
        BeamProcessInfo: The corresponding BeamProcessInfo dataclass instance.
    """
    if accelerator != "lhc":
        raise NotImplementedError("Active-Beamprocess retrieval is only implemented for LHC")

    beamprocessmap = lsa_client._lhcService.findResidentStandAloneBeamProcessesByTime(
        int(time.timestamp() * 1000)  # java timestamps are in milliseconds
    )
    beamprocess = beamprocessmap.get(bp_group)
    if beamprocess is None:
        raise ValueError(
            f"No active BeamProcess found for group '{bp_group}' at time {time.isoformat()}."
        )
    LOGGER.debug(f"Active BeamProcess at time '{time.isoformat()}': {str(beamprocess)}")
    return BeamProcessInfo.from_java_beamprocess(beamprocess)


def get_beamprocess_from_name(
    lsa_client: LSAClient,
    name: str,
    ) -> BeamProcessInfo:
    """
    Get the BeamProcess by its name.

    Args:
        lsa_client (LSAClient): The LSA client instance.
        name (str): The name of the beam process.

    Returns:
        BeamProcessInfo: The corresponding BeamProcessInfo dataclass instance.
    """

    LOGGER.debug(f"Extracting Beamprocess {name} from ContextService")
    beamprocess = lsa_client._contextService.findStandAloneBeamProcess(name)
    return BeamProcessInfo.from_java_beamprocess(beamprocess)


def get_beamprocess_with_fill_at_time(
    lsa_client: LSAClient,
    spark: SparkSession,
    time: datetime,
    data_retrieval_days: float = 1,  # assumes fills are not longer than 1 day
    accelerator: str = "lhc",
) -> tuple[FillInfo, BeamProcessInfo]:
    """Get the info about the active beamprocess at ``time``."""

    # Beamprocess -
    bp_info = get_active_beamprocess_at_time(lsa_client, time, accelerator=accelerator)

    # Fill -
    fills = get_beamprocesses_for_fills(
        lsa_client,
        spark,
        time=time,
        data_retrieval_days=data_retrieval_days,
        accelerator=accelerator,
    )
    fill_info = fills[-1]  # last fill before time

    # Get start time directly from the fill timestamps,
    # as the time in the extracted beamprocess is often 01/01/1970.
    try:
        start_time = _find_beamprocess_start(fill_info.beamprocesses, time, bp_info.name)
    except ValueError as e:
        raise ValueError(f"In fill {fill_info.no} the {str(e)}") from e
    LOGGER.debug(
        f"Beamprocess {bp_info.name} in fill {fill_info.no} started at time {start_time.isoformat()}."
    )
    LOGGER.debug(f"Replacing beamprocess start time {bp_info.start_time.isoformat()} with {start_time.isoformat()}.")
    bp_info.start_time = start_time

    return fill_info, bp_info


def beamprocess_to_dict(bp: StandAloneBeamProcessImpl) -> dict:
    """Convert all ``get``- fields of the beamprocess (java) to a dictionary.

    Will contain the original Beamprocess object under the key 'object'
    and its string representation under 'name'.

    Args:
        bp (StandAloneBeamProcessImpl): Beamprocess object (Java).

    Returns:
        dict: Dictionary representation of the beamprocess.
    """
    bp_dict = {"name": bp.toString(), "object": bp}
    bp_dict.update(
        {
            getXY[3:].lower(): str(bp.__getattribute__(getXY)())  # note: __getattr__ does not exist
            for getXY in dir(bp)
            if getXY.startswith("get") and "Attribute" not in getXY
        }
    )
    return bp_dict


# By Fills --------------------------

def get_beamprocesses_for_fills(
    lsa_client: LSAClient,
    spark: SparkSession,
    time: datetime,
    data_retrieval_days: float,
    accelerator: str = "lhc",
) -> list[FillInfo]:
    """
    Finds the BeamProcesses between t_start and t_end and sorts then by fills.
    Adapted from pjlsa's FindBeamProcessHistory.

    Args:

    Returns:
        List of FillInfo objects, each containing the fill number, accelerator, start time, and beam processes.
        They are sorted by fill number.
    """
    # get fill numbers from nxcals
    fills_results: list[NXCALSResult] = get_raw_vars(
        spark,
        time=time,
        var_name=FILL_VARIABLE,
        data_retrieval_days=data_retrieval_days,
        latest_only=False,
    )
    LOGGER.debug(f"{len(fills_results)} fills aqcuired.")
    map_fill_times = {fr.datetime: fr.value for fr in fills_results}
    fill_times = np.array(sorted(map_fill_times.keys()))

    # retrieve beamprocess history from LSA
    beamprocess_history = lsa_client.findUserContextMappingHistory(
        # use timestamp below as pjlsa ignores timezone
        t1=(time - timedelta(days=data_retrieval_days)).timestamp(),
        t2=time.timestamp(),
        accelerator=accelerator,
        contextFamily=BP_CONTEXT_FAMILY,
    )

    # map beam-processes to fills
    fills_and_beamprocesses = {}
    for bp_timestamp, bp_name in zip(beamprocess_history.timestamp, beamprocess_history.name):
        bp_datetime = datetime.fromtimestamp(bp_timestamp, tz=tz.UTC)
        idx = fill_times.searchsorted(bp_datetime) - 1  # last fill before bp time
        fill_no = int(map_fill_times[fill_times[idx]])
        if fill_no in fills_and_beamprocesses:
            fills_and_beamprocesses[fill_no].beamprocesses.append((bp_datetime, bp_name))
        else:
            fills_and_beamprocesses[fill_no] = FillInfo(
                no=fill_no,
                accelerator=accelerator,
                start_time=fill_times[idx],
                beamprocesses=[(bp_datetime, bp_name)]
            )

    LOGGER.debug("Beamprocess History extracted.")
    return sorted(fills_and_beamprocesses.values(), key=lambda fi: fi.start_time)


def _find_beamprocess_start(
    beamprocesses: Iterable[tuple[datetime, str]], time: datetime, bp_name: str
) -> datetime:
    """
    Get the last time the given beamprocess occurs in the list of beamprocesses before the given time.
    """
    LOGGER.debug(f"Looking for beamprocess '{bp_name}' in fill before '{time.isoformat()}'")
    for ts, name in sorted(beamprocesses, reverse=True):
        if ts <= time and name == bp_name:
            LOGGER.debug(f"Found start for beamprocess '{bp_name}' at {ts.isoformat()}.")
            return ts
    raise ValueError(f"Beamprocess '{bp_name}' was not found.")
