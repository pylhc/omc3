"""
writer
-------------------

Write sdds files.

"""
from typing import IO, List, Union, Iterable
import numpy as np
from sdds.classes import (SddsFile, Column, Parameter, Definition, Array, Data, Description,
                          ENCODING, NUMTYPES)


def write_sdds(sdds_file: SddsFile, output_path: str) -> None:
    """
        Writes SddsFile object into output_path

    Args:
        sdds_file: SddsFile object to write
        output_path: path, where to write sdds_file
    """
    with open(output_path, "wb") as outbytes:
        names = _write_header(sdds_file, outbytes)
        _write_data(names, sdds_file, outbytes)


def _write_header(sdds_file: SddsFile, outbytes: IO[bytes]) -> List[str]:
    outbytes.writelines(("SDDS1\n".encode(ENCODING),
                         "!# big-endian\n".encode(ENCODING)))
    names = []
    if sdds_file.description is not None:
        outbytes.write(_sdds_def_as_str(sdds_file.description).encode(ENCODING))
    for def_name in sdds_file.definitions:
        names.append(def_name)
        definition = sdds_file.definitions[def_name]
        outbytes.write(_sdds_def_as_str(definition).encode(ENCODING))
    outbytes.write("&data mode=binary, &end\n".encode(ENCODING))
    return names


def _sdds_def_as_str(definition: Union[Description, Definition, Data]) -> str:
    start = definition.TAG + " "
    things = ", ".join([f"{key}={definition.__dict__[key]}"
                        for key in definition.__dict__ if "__" not in key])
    end = " &end\n"
    return start + things + end


def _write_data(names: List[str], sdds_file: SddsFile, outbytes: IO[bytes])-> None:
    # row_count:
    outbytes.write(np.array(0, dtype=NUMTYPES["long"]).tobytes())
    _write_parameters((sdds_file[name] for name in names
                       if isinstance(sdds_file.definitions[name], Parameter)),
                      outbytes)
    _write_arrays((sdds_file[name] for name in names
                   if isinstance(sdds_file.definitions[name], Array)),
                  outbytes)
    _write_columns((sdds_file[name] for name in names
                    if isinstance(sdds_file.definitions[name], Column)),
                   outbytes)


def _write_parameters(param_gen: Iterable[Parameter], outbytes: IO[bytes]):
    for param_def, value in param_gen:
        if param_def.type == "string":
            _write_string(value, outbytes)
        else:
            outbytes.write(np.array(value, dtype=NUMTYPES[param_def.type]).tobytes())


def _write_arrays(array_gen: Iterable[Array], outbytes: IO[bytes]):
    for array_def, value in array_gen:
        outbytes.write(np.array(len(value), dtype=NUMTYPES["long"]).tobytes())
        if array_def.type == "string":
            for string in value:
                _write_string(string, outbytes)
        else:
            outbytes.write(np.array(value, dtype=NUMTYPES[array_def.type]).tobytes())


def _write_columns(col_gen: Iterable[Column], outbytes: IO[bytes]):
    # TODO: Implement the columns thing.
    pass


def _write_string(string: str, outbytes: IO[bytes]):
    outbytes.write(np.array(len(string), dtype=NUMTYPES["long"]).tobytes())
    outbytes.write(string.encode(ENCODING))
