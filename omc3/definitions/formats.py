"""
Formats
-------------

Recurring formats are defined here.

:module: omc3.definitions.formats

"""
from pathlib import Path
from datetime import datetime

TIME = "%Y_%m_%d@%H_%M_%S_%f"  # CERN default
CONFIG_FILENAME = "{script:s}_{time:s}.ini"


def get_config_filename(script):
    """ Default Filename for config-files. Call from script with '__file__'."""
    return CONFIG_FILENAME.format(
        script=Path(script).name.split('.')[0],
        time=datetime.utcnow().strftime(TIME)
    )

