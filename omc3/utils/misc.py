"""
Miscellaneous Tools
-------------------

Miscellaneous tools for the `omc3` package.
"""
from __future__ import annotations

from enum import Enum


# TODO: remove this and replace with StrEnum from stdlib when min supported Python version is 3.11
class StrEnum(str, Enum):
    """ Enum for strings.

    From python 3.11 there will be a built-in StrEnum type,
    with the same, plus additional, functionality.

    See: https://docs.python.org/3/library/enum.html#enum.StrEnum
    """
    def __str__(self):
        return self.value
