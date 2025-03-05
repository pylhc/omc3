""" 
Debugging tools
---------------

Tools that can help with debugging.
"""
import os
import sys

PYTEST_VARIABLE: str = "PYTEST_CURRENT_TEST"


def is_pytest() -> bool:
    """ Check if we are in a pytest environment. """
    return PYTEST_VARIABLE in os.environ


def is_debug() -> bool:
    """ Check if we are running in debug mode. """
    return sys.flags.debug