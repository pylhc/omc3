"""This module holds the classes handled by the sdds handler.

Most of the documentation comes from:
https://ops.aps.anl.gov/manuals/SDDStoolkit/SDDStoolkitsu2.html
"""
from typing import Any, Tuple, List, Iterator, Optional, Dict, NewType


ENCODING = "utf-8"
ENCODING_LEN = 1

NUMTYPES = {"float": ">f", "double": ">d", "short": ">i2",
            "long": ">i4", "char": ">i1"}
NUMTYPES_SIZES = {"float": 4, "double": 8, "short": 2,
                  "long": 4, "char": 1}
NUMTYPES_CAST = {"float": float, "double": float, "short": int,
                 "long": int, "char": str}


class Description:
    """Description (&description) command container.

    This optional command describes the data set in terms of two strings. The
    first, text, is an informal description that is intended principly for
    human consumption. The second, contents, is intended to formally specify
    the type of data stored in a data set. Most frequently, the contents field
    is used to record the name of the program that created or most recently
    modified the file.
    """
    TAG: str = "&description"
    text: Optional[str]
    contents: Optional[str]

    def __init__(self,
                 text: Optional[str] = None,
                 contents: Optional[str] = None) -> None:
        self.text = text
        self.contents = contents


class Include:
    """Include (&include) command container.

    This optional command directs that SDDS header lines be read from the file
    named by the filename field. These commands may be nested.
    """
    filename: str

    def __init__(self, filename: str) -> None:
        self.filename = filename


class Definition:
    """Abstract class for the common behaviour of the data definition commands.

    The name field must be supplied, as must the type field. The type must be
    one of short, long, float, double, character, or string.

    The optional symbol field allows specification of a symbol to represent the
    parameter; it may contain escape sequences, for example, to produce Greek
    or mathematical characters.
    The optional units field allows specification of the units of the
    parameter.
    The optional description field provides for an informal description of the
    parameter.
    The optional format field allows specification of the printf format string
    to be used to print the data (e.g., for ASCII in SDDS or other formats).

    The Column, Array and Parameter definitions inherit from this class. They
    can be created just by passing name and type and optionaly more parameters
    that depend on the actual definition type.

    Raises:
        AssertionError: If an invalid argument for the definition type is
            passed.
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

    For ASCII data, the optional field_length field specifies the number of
    characters occupied by the data for the column. If zero, the data is assumed to
    be bounded by whitespace characters. If negative, the absolute value is taken
    as the field length, but leading and trailing whitespace characters will be
    deleted from string data. This feature permits reading fixed-field-length
    FORTRAN output without modification of the data to include separators.
    """
    TAG: str = "&column"
    field_length: int = 0


class Parameter(Definition):
    """Parameter (&parameter) command container, a data definition.

    This optional command defines a parameter that will appear along with the
    tabular data section of each data page.

    The optional fixed_value field allows specification of a constant value for
    a given parameter. This value will not change from data page to data page,
    and is not specified along with non-fixed parameters or tabular data. This
    feature is for convenience only; the parameter thus defined is treated like
    any other.
    """
    TAG: str = "&parameter"
    fixed_value: Optional[str] = None


class Array(Definition):
    """Array (&array) command container, a data definition.

    This optional command defines an array that will appear along with the
    tabular data section of each data page.

    The optional group_name field allows specification of a string giving the
    name of the array group to which the array belongs; such strings may be
    defined by the user to indicate that different arrays are related (e.g.,
    have the same dimensions, or parallel elements).
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

    The mode field is required, and may have one of the values “ascii” or
    “binary”. If binary mode is specified, the other entries of the command are
    irrelevant and are ignored.

    In ASCII mode, these entries are optional.
    In ASCII mode, each row of the tabular data occupies lines_per_row rows in
    the file.
    If lines_per_row is zero, however, the data is assumed to be in “stream”
    format, which means that line breaks are irrelevant.
    Each line is processed until it is consumed, at which point the next line
    is read and processed.  Normally, each data page includes an integer
    specifying the number of rows in the tabular data section. This allows for
    preallocation of arrays for data storage, and obviates the need for an
    end-of-page indicator.
    However, if no_row_counts is set to a non-zero value, the number of rows
    will be determined by looking for the occurence of an empty line. A comment
    line does not qualify as an empty line in this sense.
    If additional_header_lines is set to a non-zero value, it gives the number
    of non-SDDS data lines that follow the data command. Such lines are treated
    as comments.
    """
    TAG: str = "&data"

    def __init__(self, mode: str,
                 lines_per_row: int = 1, no_row_count: int = 0,
                 additional_header_lines: int = 0) -> None:
        self.mode = mode
        self.lines_per_row = lines_per_row
        self.no_row_count = no_row_count
        self.additional_header_lines = additional_header_lines


class SddsFile:
    """Holds the contents of the SDDS file as a pair of dictionaries.

    The first dictionary "definitions" contains the 
    """
    version: str
    description: Optional[Description]
    definitions: Dict[str, Definition]
    values: Dict[str, Any]

    def __init__(self, version: str, description: Optional[Description],
                 definitons_list: List[Definition],
                 values_list: List[Any]) -> None:
        self.version = version
        self.description = description
        self.definitions = {definition.name: definition
                            for definition in definitons_list}
        self.values = {definition.name: value
                       for definition, value in
                       zip(definitons_list, values_list)}

    def __getitem__(self, name: str) -> Tuple[Definition, Any]:
        return self.definitions[name], self.values[name]

    def __iter__(self) -> Iterator[Tuple[Definition, Any]]:
        for def_name in self.definitions:
            yield self[def_name]
