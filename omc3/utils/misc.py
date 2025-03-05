""" 
Miscellaneous Tools
-------------------

Miscellaneous tools for the `omc3` package.
"""
from enum import Enum

class StrEnum(str, Enum):
    """ Enum for strings.
    
    From python 3.11 there will be a built-in StrEnum type,
    with the same, plus additional, functionality.

    See: https://docs.python.org/3/library/enum.html#enum.StrEnum
    """
    def __str__(self):
        return self.value 
