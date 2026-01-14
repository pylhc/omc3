from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import dateutil.tz as tz

from omc3.machine_data_extraction.data_classes import OpticsInfo
from omc3.utils import logging_tools
from omc3.utils.mock import cern_network_import

jpype = cern_network_import("jpype")
pjlsa = cern_network_import("pjlsa")

if TYPE_CHECKING:
    from pjlsa import LSAClient

    from omc3.machine_data_extraction.lsa_beamprocesses import BeamProcessInfo


LOGGER = logging_tools.get_logger(__name__)


def get_optics_for_beamprocess_at_time(
    lsa_client: LSAClient,
    time: datetime,
    beamprocess: BeamProcessInfo
    ) -> OpticsInfo:
    """Get the optics information for the given beamprocess at the specified time.

    Note:
       This function has been derived from old implementations and returns the right optics,
       but in my tests the optics table only ever had one entry with time 0. (jdilly, 2026)

    Args:
        lsa_client (LSAClient): The LSA client instance.
        time (datetime): The time at which to get the optics.
        beamprocess (BeamProcessInfo): The beamprocess information.

    Returns:
        OpticsInfo: The corresponding OpticsInfo dataclass instance.
    """
    LOGGER.debug(f"Getting optics for {beamprocess.name} at time {time.isoformat()}.")
    optics_table = lsa_client.getOpticTable(beamprocess.name)

    # Note: Times in optics table are relative to beamprocess start time
    if beamprocess.start_time.timestamp() == 0:
        raise ValueError(f"Beamprocess {beamprocess.name} has no valid start time.")

    time_rel = time.timestamp() - beamprocess.start_time.timestamp()
    for item in reversed(optics_table):
        if item.time <= time_rel:
            break
    else:
        raise ValueError(f"No optics found for beamprocess {beamprocess.name} at time {time.isoformat()}.")

    optics_info = OpticsInfo(
        name=item.name,
        id=item.id,
        accelerator=beamprocess.accelerator,
        start_time=datetime.fromtimestamp(item.time + beamprocess.start_time.timestamp(), tz=tz.UTC),
        beamprocess=beamprocess
    )

    LOGGER.debug(f"Optics {optics_info.name} extracted.")
    return optics_info
