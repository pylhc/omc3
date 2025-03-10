""" Knob Manipulation Utilities """
from __future__ import annotations
from typing import TYPE_CHECKING

from omc3.utils import logging_tools


if TYPE_CHECKING:
    from collections.abc import Iterable

LOGGER = logging_tools.get_logger(__name__)


def get_vars_by_classes(classes_or_knobs: Iterable[str] | None, all_knobs_by_class: dict[str, list[str]]) -> list[str]:
    """ Returns the variables that should be used based on the class names.

    Args:
        classes_or_knobs (Iterable[str] | None): The classes or variable names to use.
                                                 Variables are names that are not found as keys in the `all_knobs_by_class` dictionary.
                                                 Variables or Classes with a "-" prefix will be removed from the list.
        all_knobs_by_class (dict[str, list[str]]): The dictionary of all classes and their variables.
    """
    if classes_or_knobs is None:
        return list(set(_flatten_list([knob for knob in all_knobs_by_class.values()])))

    if isinstance(classes_or_knobs, str):
        # if not checked, lead to each char being treates as a knob. 
        raise TypeError(f"Classes must be an iterable, not a string: {classes_or_knobs}")  

    add_classes = set(c for c in classes_or_knobs if c in all_knobs_by_class)
    
    remove_classes = set(c[1:] for c in classes_or_knobs if c[0] == "-" and c[1:] in all_knobs_by_class)
    if remove_classes:
        LOGGER.info(f"The following classes will be removed from the correctors:\n{remove_classes!s}")

    add_knobs = set(knob for knob in classes_or_knobs if knob[0] != "-" and knob not in all_knobs_by_class)
    if add_knobs:
        LOGGER.info("The following names are not found as corrector/variable classes and "
                    f"are assumed to be the variable names directly instead:\n{add_knobs!s}")

    remove_knobs = set(knob[1:] for knob in classes_or_knobs if knob[0] == "-" and knob[1:] not in all_knobs_by_class)
    if remove_knobs:
        LOGGER.info(f"The following names will not be used as correctors, as requested:\n{remove_knobs!s}")
    
    knobs_to_add = set(_flatten_list(all_knobs_by_class[corr_cls] for corr_cls in add_classes)) | add_knobs
    knobs_to_remove = set(_flatten_list(all_knobs_by_class[corr_cls] for corr_cls in remove_classes)) | remove_knobs
    knobs_to_use = list(knobs_to_add - knobs_to_remove)
    
    LOGGER.debug(f"All knobs to use:\n{knobs_to_use!s}")
    return knobs_to_use


def _flatten_list(my_list: Iterable) -> list:
    return [item for sublist in my_list for item in sublist]

