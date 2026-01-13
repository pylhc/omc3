from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import dateutil.tz as tz
import numpy as np

from omc3.nxcals.knob_extraction import NXCALSResult, get_raw_vars
from omc3.nxcals.madx_conversion import map_lsa_name_to_madx
from omc3.utils import logging_tools
from omc3.utils.mock import cern_network_import

jpype = cern_network_import("jpype")
pjlsa = cern_network_import("pjlsa")

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cern.lsa.domain.settings.spi import StandAloneBeamProcessImpl
    from pjlsa import LSAClient
    from pjlsa._pjlsa import TrimTuple
    from pyspark.sql import SparkSession


LOGGER = logging_tools.get_logger(__name__)

RELEVANT_BP_CONTEXTS: tuple[str, ...] = ("OPERATIONAL", "MD")
RELEVANT_BP_CATEGORIES: tuple[str, ...] = ("DISCRETE",)
KNOBS_BP_GROUP: str = "POWERCONVERTERS"  # the Beamprocesses relevant for OMC
BP_CONTEXT_FAMILY: str = "beamprocess"

PARAMETER_KNOB: str = "KNOB"
FILL_VARIABLE = "HX:FILLN"

def calc_k_from_iref(
    lsa_client: LSAClient,
    currents: dict[str, float],
    energy: float) -> dict[str, float]:
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


# Beamprocess ##################################################################

@dataclass
class BeamProcessInfo:
    """Dataclass to hold BeamProcess information.

    This contains only the relevant fields for OMC,
    extracted from the Java BeamProcess object.
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
    delta_days: float = 1,  # assumes fills are not longer than 1 day
    accelerator: str = "lhc",
) -> tuple[FillInfo, BeamProcessInfo]:
    """Get the info about the active beamprocess at ``time``."""

    # Beamprocess -
    bp_info = get_active_beamprocess_at_time(lsa_client, time, accelerator=accelerator)

    # Fill -
    fills = get_beamprocesses_for_fills(
        lsa_client, spark, time=time, delta_days=delta_days, accelerator=accelerator
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


def get_beamprocesses_for_fills(
    lsa_client: LSAClient,
    spark: SparkSession,
    time: datetime,
    delta_days: float,
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
    fills_results: list[NXCALSResult]  = get_raw_vars(
        spark,
        time=time,
        var_name=FILL_VARIABLE,
        delta_days=delta_days,
        latest_only=False
    )
    LOGGER.debug(f"{len(fills_results)} fills aqcuired.")
    map_fill_times = {fr.datetime: fr.value for fr in fills_results}
    fill_times = np.array(sorted(map_fill_times.keys()))

    # retrieve beamprocess history from LSA
    beamprocess_history = lsa_client.findUserContextMappingHistory(
        t1=(time-timedelta(days=delta_days)).timestamp(),  # timestamp as pjlsa ignores timezone
        t2=time.timestamp(),                               # timestamp as pjlsa ignores timezone
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


# Optics #######################################################################

@dataclass
class OpticsInfo:
    """Dataclass to hold Optics information."""
    name: str
    id: str
    start_time: datetime
    accelerator: str | None = None
    beamprocess: BeamProcessInfo | None = None


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

# Knobs ########################################################################

def find_knob_names(lsa_client: LSAClient, accelerator: str = "lhc", regexp: str | None = None) -> list:
    """
    Return knobs for accelerator.

    Args:
        lsa_client: The LSA client instance.
        accelerator (str): Accelerator name.
        regexp (str): Regular Expression to filter knob names.

    Returns:
        Sorted list of knob names.
    """
    LOGGER.debug(f"Getting knobs for {accelerator}.")

    # Prepare request
    request_builder = lsa_client._ParametersRequestBuilder()
    request_builder.setAccelerator(lsa_client._getAccelerator(accelerator))
    request_builder.setParameterTypeName(PARAMETER_KNOB)

    # Get parameters and filter
    parameters: Iterable = lsa_client._parameterService.findParameters(request_builder.build())

    LOGGER.debug(f"{len(parameters)} Knobs extracted for {accelerator}.")
    param_names: list[str] = [param.getName() for param in parameters]

    if regexp is not None:
        LOGGER.debug(f"Selecting Knobs containing expression: {regexp}")
        pattern = re.compile(regexp, re.IGNORECASE)
        return sorted(filter(pattern.search, param_names))  # type: ignore[arg-type]

    return sorted(param_names)


def filter_non_existing_knobs(lsa_client: LSAClient, knobs: list[str]) -> list[str]:
    """
    Return only the knobs that exist from the given list.
    This function was created out of the need to filter these first,
    as knobs that exist but do not belong to a beamprocess return noting in
    _getTrimsByBeamprocess, while knobs that do not exist at all crashed pjlsa.
    This filter should probably have been in pjlsa's _buildParameterList.

    Args:
        knobs (list): List of strings of the knobs to check.

    Returns:
        A list of the knob names that actually exist.
    """
    LOGGER.debug(f"Checking if the following knobs exist: {knobs}")
    dont_exist = [k for k in knobs if lsa_client._getParameter(k) is None]
    if len(dont_exist):
        LOGGER.warning(f"The following knobs do not exist and will be filtered: {dont_exist}.")
        knobs = [k for k in knobs if k not in dont_exist]
    return knobs


def get_trim_history(
    lsa_client: LSAClient,
    beamprocess: str,
    knobs: list | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    accelerator: str = "lhc",
) -> dict[str, TrimTuple]:
    """
    Get trim history for knobs between specified times.
    If any of the times are not given, all available data in that time-direction
    is extracted.

    Args:
        beamprocess (str): Name of the beamprocess.
        knobs (list): List of strings of the knobs to check.
        start_time (datetime): Earliest time to extract trims from.
        end_time (datetime): Latest time to extract trims to.
        accelerator (str): Name of the accelerator.

    Returns:
        Dictionary of trims and their data (as TrimTuples, i.e. NamedTuple of lists of time and data).
    """
    LOGGER.debug("Extracting Trim history.")
    if not knobs:
        knobs = find_knob_names(lsa_client, accelerator=accelerator)
    else:
        knobs = filter_non_existing_knobs(lsa_client, knobs)
        if not knobs:
            raise ValueError("None of the given knobs exist!")

    LOGGER.debug(f"Getting trims for {len(knobs)} knobs.")
    try:
        trims = lsa_client.getTrims(
            parameter=knobs,
            beamprocess=beamprocess,
            start=start_time.timestamp() if start_time is not None else None,  # avoid timezone issues
            end=end_time.timestamp() if end_time is not None else None,        # avoid timezone issues
        )
    except jpype.java.lang.NullPointerException as e:
        # In the past this happened, when a knob was not defined, but
        # this should have been caught by the filter_existing_knobs above
        raise ValueError(
            f"Something went wrong when extracting trims for the knobs: {knobs}"
        ) from e

    LOGGER.debug(f"{len(trims)} trims extracted.")
    trims_not_found = [k for k in knobs if k not in trims]
    if len(trims_not_found):
        LOGGER.warning(
            f"The following knobs were not found in '{beamprocess}' "
            f"or had no trims during the given time: {trims_not_found}"
        )
    return trims


def get_last_trim(trims: dict[str, TrimTuple]) -> dict[str, float]:
    """Returns the last trim in the trim history.

    Args:
        trims (dict): trim history as extracted via LSA.get_trim_history()

    Returns:
        Dictionary of knob names and their values.
    """
    LOGGER.debug("Extracting last trim from found trim histories.")
    trim_dict = {trim: value.data[-1] for trim, value in trims.items()}  # return last set value
    for trim, value in trim_dict.items():
        try:
            trim_dict[trim] = value.flatten()[-1]  # the very last entry ...
        except AttributeError:
            continue  # single value, as expected
        else:
            LOGGER.warning(f"Trim {trim} hat multiple data entries {value}, taking only the last one.")
    return trim_dict


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

    def to_madx(self, value: float = 0.0) -> str:
        """Converts the knob definition to madx code."""
        if not self.parts:
            raise ValueError(f"Knob {self.name} has no parts defined!")

        knob_name = self.name.replace("/", "_").replace("-", "_")

        string_parts = [
            f"{part.madx_name} = {part.madx_name} + {part.factor:.7e} * {knob_name};"
            if part.madx_name else
            f"! WARNING: No MAD-X name for circuit {part.circuit} (factor={part.factor}) in knob {self.name}"
            for part in self.parts
        ]

        return "\n".join([
            "! Knob Definition: {self.name} for optics {self.optics}",
            f"{knob_name} = {value};",
            ] + string_parts)


def get_knob_definition(lsa_client: LSAClient, knob: str, optics: str) -> KnobDefinition:
    """
    Get a dataframe of the structure of the knob. Similar to online model extractor
    (KnobExtractor.getKnobHiercarchy)

    Args:
        lsa_client (LSAClient): The LSA client instance.
        knob (str): The name of the knob.
        optics (str): The optics name.

    Returns:
        KnobDefinition: The definition of the knob.
    """
    LOGGER.debug(f"Getting knob defintions for '{knob}', optics '{optics}'")

    knob_def = KnobDefinition(name=knob, optics=optics)

    lsa_knob = lsa_client._knobService.findKnob(knob)
    if lsa_knob is None:
        raise ValueError(f"Knob '{knob}' does not exist")

    try:
        knob_settings = lsa_knob.getKnobFactors().getFactorsForOptic(optics)
    except jpype.java.lang.IllegalArgumentException:
        raise ValueError(f"Knob '{knob}' not available for optics '{optics}'")

    for knob_setting in knob_settings:
        circuit = knob_setting.getComponentName()
        param = lsa_client._parameterService.findParameterByName(circuit)

        madx_name = map_lsa_name_to_madx(lsa_client, circuit)
        if madx_name is None:
            LOGGER.error(
                f"Circuit '{circuit}' could not be resolved to a MADX name in LSA!"
            )

        part = KnobPart(
            circuit=circuit,
            type=param.getParameterType().getName(),
            factor=knob_setting.getFactor(),
            madx_name=madx_name
        )
        LOGGER.debug(f"Found component {part}")
        knob_def.parts.append(part)
    return knob_def
