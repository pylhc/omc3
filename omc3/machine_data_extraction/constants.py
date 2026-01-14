"""
Constants
---------

Specific constants relating to the retrieval of machine settings information, to help with consistency.
"""

from omc3.optics_measurements.constants import EXT as TFS_SUFFIX
from omc3.utils.misc import StrEnum


class MSISummaryColumn(StrEnum):
    """
    TFS columns for machine settings information.
    """
    KNOB = "KNOB"
    TIME = "TIME"
    TIMESTAMP = "TIMESTAMP"
    VALUE = "VALUE"

class MSISummaryHeader(StrEnum):
    """
    TFS headers for machine settings information.
    """
    ACCEL = "ACCELERATOR"
    TIME = "TIME"
    START_TIME = "START_TIME"
    END_TIME = "END_TIME"
    BEAMPROCESS = "BEAMPROCESS"
    FILL = "FILL"
    BEAMPROCESS_START = "BEAMPROCESS_START"
    CONTEXT_CATEGORY = "CONTEXT_CATEGORY"
    BEAMPROCESS_DESCRIPTION = "BEAMPROCESS_DESCRIPTION"
    OPTICS = "OPTICS"
    OPTICS_START = "OPTICS_START"

# Filenames
MADX_SUFFIX = ".madx"
MSI_SUMMARY_FILENAME: str = f"machine_settings{TFS_SUFFIX}"

KNOB_DEFINITION_ID: str = "_definition"
KNOB_DEFINITION_TFS: str = f"{KNOB_DEFINITION_ID}{TFS_SUFFIX}"
KNOB_DEFINITION_MADX: str = f"{KNOB_DEFINITION_ID}{MADX_SUFFIX}"

TRIM_HISTORY_ID: str = "_trims"
TRIM_HISTORY_TFS: str = f"{TRIM_HISTORY_ID}{TFS_SUFFIX}"
TRIM_HISTORY_MADX: str = f"{TRIM_HISTORY_ID}{MADX_SUFFIX}"