"""
Constants
---------

Specific constants relating to the retrieval of machine settings information, to help with consistency.
"""

from omc3.optics_measurements.constants import EXT as TFS_SUFFIX

# Filename
info_name: str = f"machine_settings{TFS_SUFFIX}"
knobdef_suffix: str = f"_definition{TFS_SUFFIX}"
trimhistory_suffix: str = f"_trims{TFS_SUFFIX}"

# Columns
column_knob: str = "KNOB"
column_time: str = "TIME"
column_timestamp: str = "TIMESTAMP"
column_value: str = "VALUE"

# Headers
head_accel: str = "Accelerator"
head_time: str = "Time"
head_start_time: str = "StartTime"
head_end_time: str = "EndTime"
head_beamprocess: str = "Beamprocess"
head_fill: str = "Fill"
head_beamprocess_start: str = "BeamprocessStart"
head_context_category: str = "ContextCategory"
head_beamprcess_description: str = "Description"
head_optics: str = "Optics"
head_optics_start: str = "OpticsStart"
