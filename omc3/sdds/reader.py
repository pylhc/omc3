"""
reader
-------------------

Read sdds files.

"""
from typing import IO, Any, List, Optional, Generator, Dict, Union, Tuple, Callable, Type
import numpy as np
from sdds.classes import (SddsFile, Column, Parameter, Definition, Array, Data, Description,
                          ENCODING, NUMTYPES, NUMTYPES_CAST, NUMTYPES_SIZES)


def read_sdds(file_path: str) -> SddsFile:
    """
    Reads SDDS file from specified file_path

    Args:
        file_path: path to SDDS file

    Returns:
        SddsFile object
    """
    with open(file_path, "rb") as inbytes:
        version, definition_list, description, data = _read_header(inbytes)
        data_list = _read_data(data, definition_list, inbytes)
        return SddsFile(version, description, definition_list, data_list)


##############################################################################
# Common reading of header and data.
##############################################################################

def _read_header(inbytes: IO[bytes]) ->Tuple[str, List[Definition], Optional[Description], Data]:
    word_gen = _gen_words(inbytes)
    version = next(word_gen)  # First token is the SDDS version
    assert version == "SDDS1",\
        "This module is compatible with SDDS v1 only... are there really other versions?"
    definitions: List[Definition] = []
    description: Optional[Description] = None
    data: Optional[Data] = None
    for word in word_gen:
        def_dict: Dict[str, str] = _get_def_as_dict(word_gen)
        if word in (Column.TAG, Parameter.TAG, Array.TAG):
            definitions.append({
                Column.TAG: Column,
                Parameter.TAG: Parameter,
                Array.TAG: Array}[word](name=def_dict.pop("name"),
                                        type_=def_dict.pop("type"),
                                        **def_dict))
            continue
        if word == Description.TAG:
            if description is not None:
                raise ValueError("Two &description tags found.")
            description = Description(**def_dict)
            continue
        if word == "&include":
            # TODO: This should be easy but I will not support it for now.
            raise NotImplementedError
        if word == Data.TAG:
            data = Data(mode=def_dict.pop("mode"))
            break
        raise ValueError(f"Unknown token: {word} encountered.")
    if data is None:
        raise ValueError("Found end of file while looking for &data tag.")
    definitions = _sort_definitions(definitions)
    return version, definitions, description, data


def _sort_definitions(orig_defs: List[Definition]) -> List[Definition]:
    """Sorts the definitions in the parameter, array, column order.

    According to the specification parameters appear first in data pages then arrays
    and then columns. Inside each group they follow the order of appearance in the header.
    """
    definitions: List[Definition] = [definition for definition in orig_defs
                                     if isinstance(definition, Parameter)]
    definitions.extend([definition for definition in orig_defs if isinstance(definition, Array)])
    definitions.extend([definition for definition in orig_defs if isinstance(definition, Column)])
    return definitions


def _read_data(data: Data, definitions: List[Definition], inbytes: IO[bytes]) -> List[Any]:
    if data.mode == "binary":
        return _read_data_binary(definitions, inbytes)
    raise ValueError(f"Unsupported data mode {data.mode}.")


##############################################################################
# Binary data reading
##############################################################################

def _read_data_binary(definitions: List[Definition], inbytes: IO[bytes]) -> List[Any]:
    row_count: int = _read_bin_int(inbytes)  # First int in bin data
    functs_dict: Dict[Type[Definition], Callable] = {
        Parameter: _read_bin_param,
        Column: lambda x, y: _read_bin_column(x, y, row_count),
        Array: _read_bin_array
    }
    return [functs_dict[definition.__class__](inbytes, definition) for definition in definitions]


def _read_bin_param(inbytes: IO[bytes], definition: Parameter) -> Union[int, float, str]:
    try:
        if definition.fixed_value is not None:
            if definition.type == "string":
                return definition.fixed_value
            return NUMTYPES_CAST[definition.type](definition.fixed_value)
    except AttributeError:
        pass
    if definition.type == "string":
        str_len: int = _read_bin_int(inbytes)
        return inbytes.read(str_len).decode(ENCODING)
    return NUMTYPES_CAST[definition.type](
        _read_bin_numeric(inbytes, definition.type, 1)
    )


def _read_bin_column(inbytes: IO[bytes], definition: Column, row_count: int):
    # TODO: This columns things might be interesting to implement.
    raise NotImplementedError("")


def _read_bin_array(inbytes: IO[bytes], definition: Array) -> Any:
    dims, total_len = _read_bin_array_len(inbytes, definition.dimensions)
    if definition.type == "string":
        len_type: str = "long"\
                        if not hasattr(definition, "modifier")\
                        else {"u1": "char", "i2": "short"}\
                             .get(definition.modifier, "long")
        str_array = []
        for _ in range(total_len):
            str_len = int(_read_bin_numeric(inbytes, len_type, 1))
            str_array.append(inbytes.read(str_len).decode(ENCODING))
        return str_array
    return _read_bin_numeric(inbytes, definition.type, total_len).reshape(dims)


def _read_bin_array_len(inbytes: IO[bytes], num_dims: int) -> Tuple[List[int], int]:
    dims = [_read_bin_int(inbytes) for _ in range(num_dims)]
    return dims, int(np.prod(dims))


def _read_bin_numeric(inbytes: IO[bytes], type_: str, count: int) -> Any:
    return np.frombuffer(inbytes.read(count * NUMTYPES_SIZES[type_]),
                         dtype=np.dtype(NUMTYPES[type_]))


def _read_bin_int(inbytes: IO[bytes]) -> int:
    return int(_read_bin_numeric(inbytes, "long", 1))


##############################################################################
# Helper generators to consume the input bytes
##############################################################################

def _gen_real_lines(inbytes: IO[bytes]) -> Generator[str, None, None]:
    """No comments and stripped lines."""
    while True:
        line = inbytes.readline().decode(ENCODING)
        if not line:
            return  # TODO is this ok?
        if line != "\n" and not line.strip().startswith("!"):
            yield line.strip()


def _gen_words(inbytes: IO[bytes]) -> Generator[str, None, None]:
    for line in _gen_real_lines(inbytes):
        for word in line.split():
            yield word
    return


def _get_def_as_dict(word_gen: Generator[str, None, None]) -> Dict[str, str]:
    raw_str: List[str] = []
    for word in word_gen:
        if word.strip() == "&end":
            recomposed: str = " ".join(raw_str)
            parts = [assign for assign in recomposed.split(",") if assign]
            return {key.strip(): value.strip() for (key, value) in
                    [assign.split("=") for assign in parts]}
        raw_str.append(word.strip())
    raise ValueError("EOF found while looking for &end tag.")
