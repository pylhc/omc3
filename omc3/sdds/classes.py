"""
Module sdds.classes
----------------------

This module holds the classes handled by the sdds handler.

Most of the documentation comes from:
https://ops.aps.anl.gov/manuals/SDDStoolkit/SDDStoolkitsu2.html

"""
from typing import Any, Tuple, List, Iterator, Optional, Dict


ENCODING = "utf-8"
ENCODING_LEN = 1

NUMTYPES =       {"float": ">f",  "double": ">d",  "short": ">i2", "long": ">i4", "char": ">i1"}
NUMTYPES_SIZES = {"float": 4,     "double": 8,     "short": 2,     "long": 4,     "char": 1}
NUMTYPES_CAST =  {"float": float, "double": float, "short": int,   "long": int,   "char": str}


class Description:
    """Description (&description) command container.

    This optional command describes the data set in terms of two strings.
    The first, text, is an informal description that is intended principally for human consumption.
    The second, contents, is intended to formally specify the type of data stored in a data set.
    Most frequently, the contents field is used to record the name of the program that created
    or most recently modified the file.
    """
    TAG: str = "&description"
    text: Optional[str]
    contents: Optional[str]

    def __init__(self, text: Optional[str] = None, contents: Optional[str] = None) -> None:
        self.text = text
        self.contents = contents


class Include:
    """Include (&include) command container.

    This optional command directs that SDDS header lines be read from the file named
    by the filename field. These commands may be nested.
    """
    filename: str

    def __init__(self, filename: str) -> None:
        self.filename = filename


class Definition:
    """Abstract class for the common behaviour of the data definition commands.

    The name field must be supplied, as must the type field. The type must be
    one of short, long, float, double, character, or string.

    The optional symbol field allows specification of a symbol to represent the parameter;
    it may contain escape sequences, for example, to produce Greek or mathematical characters.
    The optional units field allows specification of the units of the parameter.
    The optional description field provides for an informal description of the parameter.
    The optional format field allows specification of the print format string to be used to
    print the data (e.g., for ASCII in SDDS or other formats).

    The Column, Array and Parameter definitions inherit from this class. They can be created just by
    passing name and type and optionally more parameters that depend on the actual definition type.

    Raises:
        AssertionError: If an invalid argument for the definition type is passed.
    """
    name: str
    type: str
    symbol: Optional[str] = None
    units: Optional[str] = None
    description: Optional[str] = None
    format_string: Optional[str] = None

    def __init__(self, name: str, type_: str, **kwargs) -> None:
        self.name = name
        self.type = type_
        for argname in kwargs:
            assert hasattr(self, argname),\
                   f"Unknown name {argname} for data type "\
                   f"{self.__class__.__name__}"
            setattr(self, argname, kwargs[argname])


class Column(Definition):
    """Column (&column) command container, a data definition.

    This optional command defines a column that will appear in the tabular data
    section of each data page.
    """
    TAG: str = "&column"


class Parameter(Definition):
    """Parameter (&parameter) command container, a data definition.

    This optional command defines a parameter that will appear along with the
    tabular data section of each data page.

    The optional fixed_value field allows specification of a constant value for a given parameter.
    This value will not change from data page to data page, and is not specified along
    with non-fixed parameters or tabular data. This feature is for convenience only;
    the parameter thus defined is treated like any other.
    """
    TAG: str = "&parameter"
    fixed_value: Optional[str] = None


class Array(Definition):
    """Array (&array) command container, a data definition.

    This optional command defines an array that will appear along with the
    tabular data section of each data page.

    The optional group_name field allows specification of a string giving the name of the array
    group to which the array belongs; such strings may be defined by the user to indicate
    that different arrays are related (e.g., have the same dimensions, or parallel elements).
    The optional dimensions field gives the number of dimensions in the array.
    """
    TAG: str = "&array"
    field_length: int = 0
    group_name: Optional[str] = None
    dimensions: int = 1


class Data:
    """Data (&data) command container.

    This command is optional unless parameter commands without fixed_value
    fields, array commands, or column commands have been given.

    The mode field is required, and it must be “binary”, the only supported mode.
    """
    TAG: str = "&data"

    def __init__(self, mode: str) -> None:
        self.mode = mode


class SddsFile:
    """Holds the contents of the SDDS file as a pair of dictionaries.

    The first dictionary "definitions" has the form: name (as a str) ->
    Definition, containing an object of each field in the SDDS file (of type
    Parameter, Array or Column). The "values" dictionary has the form:
    name (as a str) -> value. To access them:
    sdds_file = SddsFile(...)

    .. code-block:: python

        def_ = sdds_file.definitions["name"]
        val = sdds_file.values["name"]
        # The definitions and values can also be accessed like:
        def_, val = sdds_file["name"]

    """
    version: str  # This should always be "SDDS1"
    description: Optional[Description]
    definitions: Dict[str, Definition]
    values: Dict[str, Any]

    def __init__(self, version: str, description: Optional[Description],
                 definitions_list: List[Definition],
                 values_list: List[Any]) -> None:
        self.version = version
        self.description = description
        self.definitions = {definition.name: definition for definition in definitions_list}
        self.values = {definition.name: value for definition, value
                       in zip(definitions_list, values_list)}

    def __getitem__(self, name: str) -> Tuple[Definition, Any]:
        return self.definitions[name], self.values[name]

    def __iter__(self) -> Iterator[Tuple[Definition, Any]]:
        for def_name in self.definitions:
            yield self[def_name]
