import numpy as np

from sdds_files import sdds_reader

# We will only use big endian.
HARDCODED_HEAD = "SDDS1\n!# big-endian\n"


TYPES = {
    "boolean": np.dtype(">i1"),
    "char": np.dtype(">i1"),
    "double": np.dtype(">d"),
    "float": np.dtype(">f"),
    # Long means 32bits in these sdds...
    "long": np.dtype(">i"),
    "int": np.dtype(">i"),
    "short": np.dtype(">i2"),
}


def write_sdds_file(sdds_file, output_file, binary=True):
    """Writes the sdds_file SddsFile object into output_file.

    The SDDS specification accepts a non-binary mode (ASCII), but this has not
    been implemented yet (TODO?).
    """
    content = get_sdds_binary(sdds_file, binary=binary)
    with open(output_file, "wb") as outdata:
        outdata.write(content)


def get_sdds_binary(sdds_file, binary=True):
    """Reads sdds_file and returns its content as a binary string.
    """
    header = _compute_header(sdds_file)
    if binary:
        header += "&data mode=binary, " + sdds_reader.END_TAG + "\n"
        data = _compute_data_binary(sdds_file)
    else:
        raise NotImplementedError("Only binary mode for now.")
    return header + data


# Building the ASCII header ###################################################

def _compute_header(sdds_file):
    header = HARDCODED_HEAD
    header += _compute_params_head(sdds_file)
    header += _compute_arrays_head(sdds_file)
    header += _compute_cols_head(sdds_file)
    return header


def _compute_params_head(sdds_file):
    header = ""
    for param_name in sdds_file.get_parameters():
        param = sdds_file.get_parameters()[param_name]
        header += sdds_reader.PARAMETER_TAG + " "
        header += _common_headers(param)
        header += sdds_reader.END_TAG + "\n"
    return header


def _compute_arrays_head(sdds_file):
    header = ""
    for array_name in sdds_file.get_arrays():
        array = sdds_file.get_arrays()[array_name]
        header += sdds_reader.ARRAY_TAG + " "
        header += _common_headers(array)
        header += _add(sdds_reader.GROUP, array.group)
        header += sdds_reader.END_TAG + "\n"
    return header


def _compute_cols_head(sdds_file):
    header = ""
    for col_name in sdds_file.get_columns():
        col = sdds_file.get_columns()[col_name]
        header += sdds_reader.COLUMN_TAG + " "
        header += _common_headers(col)
        header += sdds_reader.END_TAG + "\n"
    return header


def _common_headers(thing):
    header = ""
    header += _add(sdds_reader.NAME, thing.name)
    header += _add(sdds_reader.TYPE, thing.type_name)
    header += _add(sdds_reader.UNITS, thing.units)
    header += _add(sdds_reader.SYMBOL, thing.symbol)
    header += _add(sdds_reader.MODIFIER, thing.modifier)
    header += _add(sdds_reader.FORMAT_STRING, thing.format_string)
    header += _add(sdds_reader.DESCRIPTION, thing.description)
    return header


def _add(name, value):
    return "{}={}, ".format(name, value) if value else ""


###############################################################################


# Adding binary data ##########################################################

def _compute_data_binary(sdds_file):
    # This 0 is called row_count in the reader... not sure of its purpose
    data = np.array(0, dtype=TYPES["int"]).tobytes()
    data += _compute_params_bin(sdds_file)
    data += _compute_arrays_bin(sdds_file)
    data += _compute_cols_bin(sdds_file)
    return data


def _compute_params_bin(sdds_file):
    data = ""
    for param_name in sdds_file.get_parameters():
        param = sdds_file.get_parameters()[param_name]
        if param.type_name == "string":
            data += _compute_string(param, param.value)
        else:
            data += np.array(param.value, dtype=TYPES[param.type_name]).tobytes()
    return data


def _compute_arrays_bin(sdds_file):
    data = ""
    for array_name in sdds_file.get_arrays():
        array = sdds_file.get_arrays()[array_name]
        data += np.array(len(array.values), dtype=TYPES["int"]).tobytes()
        if array.type_name == "string":
            for string in array.values:
                data += _compute_string(array, string)
        else:
            data += np.array(array.values, dtype=TYPES[array.type_name]).tobytes()
    return data


def _compute_cols_bin(sdds_file):
    # TODO: I dont know what these columns things are...
    return ""


def _compute_string(thing, string):
    data = ""
    type_ = TYPES["int"]
    if thing.modifier == "u1":
        type_ = TYPES["byte"]
    elif thing.modifier == "i2":
        type_ = TYPES["short"]
    data += np.array(len(string), dtype=type_).tobytes()
    data += string.encode("utf-8")
    return data


###############################################################################
