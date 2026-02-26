"""
LSA Knobs
---------

This module contains functions to extract knob information from LSA.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from omc3.machine_data_extraction.data_classes import KnobDefinition, KnobPart, TrimHistories
from omc3.machine_data_extraction.madx_conversion import map_lsa_name_to_madx
from omc3.utils import logging_tools
from omc3.utils.mock import cern_network_import

jpype = cern_network_import("jpype")
pjlsa = cern_network_import("pjlsa")

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from pjlsa import LSAClient
    from pjlsa._pjlsa import TrimTuple


LOGGER = logging_tools.get_logger(__name__)

PARAMETER_KNOB: str = "KNOB"


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
    existing = []
    missing = []

    for k in knobs:
        if lsa_client._getParameter(k) is None:
            missing.append(k)
        else:
            existing.append(k)

    if missing:
        LOGGER.warning(f"The following knobs do not exist and will be filtered: {missing}")

    return existing


def get_trim_history(
    lsa_client: LSAClient,
    beamprocess: str,
    knobs: list | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    accelerator: str = "lhc",
) -> TrimHistories:
    """
    Get trim history for knobs between specified times.
    If any of the times are not given, all available data in that time-direction
    is extracted for that beamprocess.

    Args:
        beamprocess (str): Name of the beamprocess.
        knobs (list): List of strings of the knobs to check.
        start_time (datetime): Earliest time to extract trims from.
        end_time (datetime): Latest time to extract trims to.
        accelerator (str): Name of the accelerator.

    Returns:
        TrimHistory object containing trim extraction and history information.
        The actual trims are in the 'trims' attribute as a dictionary
        of knob names and their data (as TrimTuples, i.e. NamedTuple of lists of time and data).
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

    return TrimHistories(
        beamprocess=beamprocess,
        start_time=start_time,
        end_time=end_time,
        accelerator=accelerator,
        trims=trims
    )


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
