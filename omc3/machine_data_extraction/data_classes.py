"""
DataClasses for Machine Data
----------------------------

This module contains the dataclasses that are used to hold
and pass on the extracted machine data.

Most of them are purely to hold the data in a structured way,
some have minor functionality to convert the data into different formats
such as TFS and MADX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import dateutil.tz as tz
import tfs

from omc3.machine_data_extraction.utils import knob_to_output_name, timestamp_to_utciso
from omc3.utils.misc import StrEnum
from omc3.utils.mock import cern_network_import

jpype = cern_network_import("jpype")
pjlsa = cern_network_import("pjlsa")

if TYPE_CHECKING:
    from cern.lsa.domain.settings.spi import StandAloneBeamProcessImpl
    from pjlsa._pjlsa import TrimTuple


@dataclass
class MachineSettingsInfo:
    """Dataclass to hold the extracted Machine Settings Info."""
    time: datetime
    accelerator: str
    fill: FillInfo | None = None
    beamprocess: BeamProcessInfo | None = None
    optics: OpticsInfo | None = None
    trim_histories: TrimHistories | None = None
    trims: dict[str, float] | None = None
    knob_definitions: dict[str, KnobDefinition] | None = None


# Fill ----

@dataclass
class FillInfo:
    """Dataclass to hold Fill information.

    This contains only the relevant fields for OMC,
    extracted from the Java Fill object.
    Add more fields if needed.
    """
    no: int
    accelerator: str
    start_time: datetime
    beamprocesses: list[tuple[datetime, str]] | None = None

    def __hash__(self) -> int:
        return hash((self.no, self.accelerator, self.start_time))


# BeamProcess ----

@dataclass
class BeamProcessInfo:
    """Dataclass to hold BeamProcess information.

    This contains only the relevant fields for OMC, extracted from the Java BeamProcess object.
    Add more fields if needed.
    """
    name: str
    accelerator: str
    context_category: str
    start_time: datetime
    category: str
    description: str

    @classmethod
    def from_java_beamprocess(
        cls, bp: StandAloneBeamProcessImpl
    ) -> BeamProcessInfo:
        """Create a BeamProcessInfo from a StandAloneBeamProcessImpl object.

        Args:
            bp (StandAloneBeamProcessImpl): The BeamProcess object (Java).

        Returns:
            BeamProcessInfo: The corresponding BeamProcessInfo dataclass instance.
        """
        return cls(
            name=bp.getName(),
            accelerator=bp.getAccelerator().getName(),
            context_category=bp.getContextCategory().toString(),
            category=bp.getCategory().toString(),
            start_time=datetime.fromtimestamp(bp.getStartTime() / 1000, tz=tz.UTC),  # Note: might be wrong, better to get from Fill info
            description=bp.getDescription(),
        )


# Optics ----

@dataclass
class OpticsInfo:
    """Dataclass to hold Optics information."""
    name: str
    id: str
    start_time: datetime
    accelerator: str | None = None
    beamprocess: BeamProcessInfo | None = None


# Trims ----

class TrimHistoryColumn(StrEnum):
    """Columns for Trim History TFS files."""
    TIME = "TIME"
    TIMESTAMP = "TIMESTAMP"
    DATA = "DATA"


class TrimHistoryHeader(StrEnum):
    """Headers for Trim History TFS files."""
    ACCEL = "ACCELERATOR"
    BEAMPROCESS = "BEAMPROCESS"
    START_TIME = "START_TIME"
    END_TIME = "END_TIME"
    KNOB_NAME = "KNOB_NAME"
    OPTICS = "OPTICS"  # set in machine settings info
    FILL = "FILL"      # set in machine settings info


@dataclass
class TrimHistories:
    """Dataclass to hold Knob Trim History information."""
    beamprocess: str
    start_time: datetime | None
    end_time: datetime | None
    accelerator: str
    trims: dict[str, TrimTuple]

    def __post_init__(self):
        self.headers: dict[str, str | float | int] = self._create_tfs_header()

    def _create_tfs_header(self) -> dict[str, str]:
        """Generate common TFS headers for TrimHistory."""
        headers = {
            TrimHistoryHeader.BEAMPROCESS: self.beamprocess,
            TrimHistoryHeader.ACCEL: self.accelerator,
        }
        if self.start_time is not None:
            headers[TrimHistoryHeader.START_TIME] = self.start_time.isoformat()
        if self.end_time is not None:
            headers[TrimHistoryHeader.END_TIME] = self.end_time.isoformat()
        return headers

    def trim_tuple_to_tfs(self, knob_name: str, trim_tuple: TrimTuple) -> tfs.TfsDataFrame:
        """Convert KnobTrimHistory to TfsDataFrame."""
        df = tfs.TfsDataFrame(
            {
                TrimHistoryColumn.TIMESTAMP: trim_tuple.time,
                TrimHistoryColumn.TIME: map(timestamp_to_utciso, trim_tuple.time),
                TrimHistoryColumn.DATA: trim_tuple.data,
            },
            headers=self.headers.copy(),
        )
        df.headers[TrimHistoryHeader.KNOB_NAME] = knob_name
        return df

    def to_tfs_dict(self) -> dict[str, tfs.TfsDataFrame]:
        """Convert all trims to a dictionary of TfsDataFrames."""
        return {
            knob_to_output_name(knob_name): self.trim_tuple_to_tfs(knob_name, trim_tuple)
            for knob_name, trim_tuple in self.trims.items()
        }

# Knobs ----

class DefinitionColumn(StrEnum):
    """Columns for Knob Definition TFS files."""
    CIRCUIT = "CIRCUIT"
    TYPE = "TYPE"
    FACTOR = "FACTOR"
    MADX_NAME = "NAME"


class DefinitionHeader(StrEnum):
    """Headers for Knob Definition TFS files."""
    NAME = "KNOB_NAME"
    OPTICS = "OPTICS"
    OUTPUT_NAME = "OUTPUT_NAME"


@dataclass
class KnobPart:
    """Dataclass to hold Knob Part information."""
    circuit: str
    type: str
    factor: float
    madx_name: str | None

    def __str__(self) -> str:
        return f"{self.circuit}<{self.madx_name}, factor={self.factor}, type={self.type}>"


@dataclass
class KnobDefinition:
    """Dataclass to hold Knob Definition information."""
    name: str
    optics: str
    parts: list[KnobPart] = field(default_factory=list)

    @property
    def output_name(self) -> str:
        """Generates a safe knob name for output files."""
        return knob_to_output_name(self.name)

    def to_madx(self, value: float = 0.0, template: str = "add2expr") -> str:
        """Converts the knob definition to madx code."""
        if not self.parts:
            raise ValueError(f"Knob {self.name} has no parts defined!")

        line_template = {
            "assign": "{var} = {var} + {value:e} * {knob};",
            "add2expr": "add2expr, var={var}, expr={value:e} * {knob};",
            "defer": "{var} := {value:e} * {knob};",
        }[template]

        string_parts = [
            line_template.format(
                var=part.madx_name,
                value=part.factor,
                knob=self.output_name
            )
            if part.madx_name else
            f"! WARNING: No MAD-X name for circuit {part.circuit} (factor={part.factor}) in knob {self.name}"
            for part in self.parts
        ]

        return "\n".join([
            f"! Knob Definition: {self.name} for optics {self.optics}",
            f"{self.output_name} = {value};",
            ] + string_parts)


    def to_tfs(self) -> tfs.TfsDataFrame:
        """Convert KnobDefinition to TfsDataFrame."""
        data = {
            DefinitionColumn.CIRCUIT: [part.circuit for part in self.parts],
            DefinitionColumn.MADX_NAME: [part.madx_name for part in self.parts],
            DefinitionColumn.FACTOR: [part.factor for part in self.parts],
            DefinitionColumn.TYPE: [part.type for part in self.parts],
        }
        df = tfs.TfsDataFrame(data)
        df.headers = {
            DefinitionHeader.NAME: self.name,
            DefinitionHeader.OPTICS: self.optics,
            DefinitionHeader.OUTPUT_NAME: self.output_name,
        }
        return df
