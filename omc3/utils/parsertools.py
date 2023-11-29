"""
ParserTools
-----------

This module provides tools for the parsing of parameters.
Maybe merge this into `generic_parser` one day.
"""
from textwrap import wrap


def print_help(parameters):
    """
    Pretty prints the help information for the given parameters.

    (copied from generic parser and modified to fit in here)
    """
    for name in sorted(parameters.keys()):
        print(_get_help_str(name, parameters))


def require_param(name: str, parameters: dict, options: dict):
    """
    Guard for a missing parameter.

    This function is meant to be used if a parameter is required from a certain line on, but not before
    (because we want to get some debug info, help output, etc up until that point even if the required parameter is
    not given).

    The presence of the parameter in the parse options is checked and an `AttributeError` is thrown if it is missing.

    Args:
        name: the name of the parameter
        parameters: the list of all parameters
        options: the parsed options

    """
    if options[name] is None:
        raise AttributeError(f"Missing flag `{name}`.\nUsage:\n{_get_help_str(name, parameters)}")


def _maybe_get(space: str, category: str, map: dict)  -> str:
    if category in map:
        return f"{space}{category}: {map[category]}\n"
    return ""


def _get_help_str(name: str, parameters: dict) -> str:
    """
    Gets the parameter's help string.

    Args:
        name: the name of the parameter
        parameters: the list of all parameters

    Returns:
        The help string of the parameter `name`

    """
    space = " " * 4
    item = parameters[name]

    try:
        name_and_type = f"{_fmt_name(name)} ({_fmt_type(item['type'].__name__)}):\n"
    except KeyError:
        name_and_type = f"{_fmt_name(name)}:\n"

    try:
        help_str = f"{item['help']}"
    except KeyError:
        help_str = "-No info available-"

    help_str = "\n".join([f"{space}{line}" for line in wrap(help_str, 70)])
    help_str = f"{help_str}\n"

    flags = _maybe_get(space, "flags", item)
    choices = _maybe_get(space, "choices", item)
    default = _maybe_get(space, "default", item)
    action = _maybe_get(space, "action", item)

    return f"{name_and_type}{help_str}{flags}{choices}{default}{action}"


# ---- formatting ----------------------------------------------------------------------------------
def _fmt_name(name) -> str:
    return name
    # # if color terminal:
    # return f"\33[1m{name}\33[22m"


def _fmt_type(name) -> str:
    return name
    # # if color terminal:
    # return f"\33[33m{name}\33[0m"
