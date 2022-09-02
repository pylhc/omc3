"""
ParserTools
-----------

This module provides tools for the parsing of parameters.
Maybe merge this into `generic_parser` one day.
"""
from textwrap import wrap

def print_help(parameters):
    """ copied from generic parser and modified to fit in here"""
    optional_params = []
    required_params = []
    space = " " * 4

    for name in sorted(parameters.keys()):
        item = parameters[name]
        try:
            name_and_type = f"\33[1m{name}\33[22m (\33[33m{item['type'].__name__}\33[0m):\n"
        except KeyError:
            name_and_type = f"\33[1m{name}\33[22m:\n"

        try:
            help_str = f"{item['help']}"
        except KeyError:
            help_str = "-No info available-"

        help_str = "\n".join([f"{space}{line}" for line in wrap(help_str, 70)])
        help_str = f"{help_str}\n"

        try:
            flags = f"{space}flags: {item['flags']}\n"
        except KeyError:
            flags = ''

        try:
            choices = f"{space}choices: {item['choices']}\n"
        except KeyError:
            choices = ''

        try:
            default = f"{space}default: {item['default']}\n"
        except KeyError:
            default = ''

        try:
            action = f"{space}action: \33[35m{item['action']}\33[0m\n"
        except KeyError:
            action = ''

        item_str = f"{name_and_type}{help_str}{flags}{choices}{default}{action}"

        if item.get("required", False):
            required_params.append(item_str)
        else:
            optional_params.append(item_str)

    if required_params:
        print("Required:\n")
        print("".join(required_params))

    if optional_params:
        print("Optional:\n")
        print("".join(optional_params))


