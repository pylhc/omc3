"""
Formats
-------

Recurring formats are defined here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

TIME = "%Y_%m_%d@%H_%M_%S_%f"  # CERN default
CONFIG_FILENAME = "{script:s}_{time:s}.ini"
BACKUP_FILENAME = "{basefile}_backup{counter}"


def get_config_filename(script):
    """Default Filename for config-files. Call from script with ``__file__``."""
    return CONFIG_FILENAME.format(
        script=Path(script).stem,
        time=datetime.now(timezone.utc).strftime(TIME)
    )
