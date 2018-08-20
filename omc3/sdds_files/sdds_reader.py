import sys
import re
import logging
import numpy as np
from collections import OrderedDict

LOGGER = logging.getLogger(__name__)

# Constants #
NAME = "name"
TYPE = "type"
FORMAT_STRING = "format_string"
UNITS = "units"
DESCRIPTION = "description"
SYMBOL = "symbol"
DIMENSIONS = "dimensions"
MODIFIER = "modifier"
GROUP = "group_name"

PARAMETER_TAG = "&parameter"
ARRAY_TAG = "&array"
COLUMN_TAG = "&column"
END_TAG = "&end"
#############

DEBUG = False


def read_sdds_file(file_path):
    return SddsReader(file_path).sdds_file


class SddsTypes(object):
    class Types(object):
        (BOOLEAN, BYTE, CHAR,
         DOUBLE, FLOAT, INT,
         LONG, SHORT, STRING) = range(9)

    TYPE_IDS = {
        "boolean": Types.BOOLEAN,
        "java.lang.Boolean": Types.BOOLEAN,
        "byte": Types.BYTE,
        "sbyte": Types.BYTE,
        "java.lang.Byte": Types.BYTE,
        "character": Types.CHAR,
        "java.lang.Character": Types.CHAR,
        "char": Types.CHAR,
        "double": Types.DOUBLE,
        "java.lang.Double": Types.DOUBLE,
        "float": Types.FLOAT,
        "java.lang.Float": Types.FLOAT,
        "int": Types.INT,
        "java.lang.Integer": Types.INT,
        "long": Types.INT,  # Apparently this long means 32bits = int
        "llong": Types.LONG,
        "java.lang.Long": Types.LONG,
        "short": Types.SHORT,
        "java.lang.Short": Types.SHORT,
        "string": Types.STRING,
        "java.lang.String": Types.STRING,
        "String": Types.STRING,
    }

    TYPES_SIZES = {
        Types.BYTE: 1,
        Types.BOOLEAN: 1,
        Types.CHAR: 1,
        Types.SHORT: 2,
        Types.INT: 4,
        Types.FLOAT: 4,
        Types.DOUBLE: 8,
        Types.LONG: 8,
    }

    def __init__(self):
        pass

    @staticmethod
    def identify_type(name):
        return SddsTypes.TYPE_IDS[name]


class SddsElementDescriptor(object):

    def __init__(self):
        self.name = None
        self.type = None
        self.type_name = None
        self.units = None
        self.symbol = None
        self.modifier = None
        self.format_string = None
        self.description = None
        self.group = None
        self.dimensions = [0]


class SddsReader(object):

    def __init__(self, file_path):
        self._line_num = 0
        self._sdds_file = SddsFile()
        self._data_tag_read = False
        with open(file_path, "rb") as lines:
            self._lines = lines
            self._read_version()
            self._read_header()
            self._read_data()

    @property
    def sdds_file(self):
        return self._sdds_file

    def _read_line(self):
        self._line_num += 1
        next_line = self._lines.readline().decode("latin-1").rstrip()
        return next_line

    def _read_version(self):
        self._sdds_file.version = self._read_line()

    def _read_byte_order(self, line):
        if line.startswith("!#"):
            if "big-endian" in line:
                self._sdds_file.big_endian = True
                return True
            elif "little-endian" in line:
                self._sdds_file.big_endian = False
                return True
            return False

    def _read_header(self):
        line = self._read_line()
        if self._read_byte_order(line):
            line = self._read_line()
        while line.startswith("&"):
            if line.startswith(PARAMETER_TAG):
                self._parse_parameter_definition(line)
            elif line.startswith(ARRAY_TAG):
                self._parse_array_definition(line)
            elif line.startswith(COLUMN_TAG):
                self._parse_column_definition(line)
            elif line.startswith("&data"):
                self._parse_data_definition(line)
                break
            line = self._read_line()
        if not self._data_tag_read:
            raise IOError("&data tag hasn't been defined!")

    def _parse_parameter_definition(self, line):
        sdds_element_descriptor = self._parse_element_definition(line)
        self._sdds_file.define_parameter(
            sdds_element_descriptor.name,
            sdds_element_descriptor.type,
            sdds_element_descriptor.type_name,
            sdds_element_descriptor.units,
            sdds_element_descriptor.symbol,
            sdds_element_descriptor.modifier,
            sdds_element_descriptor.format_string,
            sdds_element_descriptor.description,
        )
        LOGGER.debug(" ".join(["Read parameter", sdds_element_descriptor.name,
                               "type", sdds_element_descriptor.type_name]))

    def _parse_array_definition(self, line):
        sdds_element_descriptor = self._parse_element_definition(line)
        self._sdds_file.define_array(
            sdds_element_descriptor.name,
            sdds_element_descriptor.type,
            sdds_element_descriptor.type_name,
            sdds_element_descriptor.dimensions,
            sdds_element_descriptor.units,
            sdds_element_descriptor.symbol,
            sdds_element_descriptor.modifier,
            sdds_element_descriptor.format_string,
            sdds_element_descriptor.description,
            sdds_element_descriptor.group,
        )
        LOGGER.debug(" ".join(["Read array", sdds_element_descriptor.name,
                               "type", sdds_element_descriptor.type_name]))

    def _parse_column_definition(self, line):
        sdds_element_descriptor = self._parse_element_definition(line)
        self._sdds_file.define_column(
            sdds_element_descriptor.name,
            sdds_element_descriptor.type,
            sdds_element_descriptor.type_name,
            sdds_element_descriptor.units,
            sdds_element_descriptor.symbol,
            sdds_element_descriptor.modifier,
            sdds_element_descriptor.format_string,
            sdds_element_descriptor.description,
        )
        LOGGER.debug(" ".join(["Read column", sdds_element_descriptor.name,
                               "type", sdds_element_descriptor.type_name]))

    def _parse_data_definition(self, line):
        tokens_iter = iter(self._tokenize(line))
        # Get rid of first token
        tokens_iter.__next__()
        should_be_mode = tokens_iter.__next__()
        if not should_be_mode == "mode":
            raise IOError("Data tag doesn't contain mode attribute! Line: " + str(self._line_num))
        mode = tokens_iter.__next__()
        self._sdds_file.is_binary = mode == "binary"
        if self._sdds_file.is_binary:
            should_be_endian = tokens_iter.__next__()
            if should_be_endian == "endian":
                self._sdds_file.big_endian = tokens_iter.__next__() == "big"
        self._data_tag_read = True

    def _parse_element_definition(self, line):
        sdds_element_descriptor = SddsElementDescriptor()
        tokens_iter = iter(self._tokenize(line))
        # Get rid of first token
        tokens_iter.__next__()
        token = tokens_iter.__next__()
        while token is not None and not token == END_TAG:
            if token == NAME:
                token = tokens_iter.__next__()
                sdds_element_descriptor.name = token
            elif token == TYPE:
                token = tokens_iter.__next__()
                sdds_element_descriptor.type = SddsTypes.identify_type(token)
                sdds_element_descriptor.type_name = token
            elif token == DIMENSIONS:
                token = tokens_iter.__next__()
                sdds_element_descriptor.dimensions = int(token)
            elif token == MODIFIER:
                token = tokens_iter.__next__()
                sdds_element_descriptor.modifier = token
            elif token == FORMAT_STRING:
                token = tokens_iter.__next__()
                sdds_element_descriptor.format_string = token
            elif token == UNITS:
                token = tokens_iter.__next__()
                sdds_element_descriptor.units = token
            elif token == DESCRIPTION:
                token = tokens_iter.__next__()
                sdds_element_descriptor.description = token
            elif token == GROUP:
                token = tokens_iter.__next__()
                sdds_element_descriptor.group = token
            token = tokens_iter.__next__()
        if (not hasattr(sdds_element_descriptor, "name") or
                sdds_element_descriptor.name is None):
            raise IOError("Header without name.")
        if (hasattr(sdds_element_descriptor, "format_string")):
            try:
                type = SddsTypes.identify_type(sdds_element_descriptor.format_string)
                sdds_element_descriptor.type = type
            except KeyError:
                pass
        return sdds_element_descriptor

    def _tokenize(self, line):
        return re.split(" |=|, ", line)

    def _read_data(self):
        if self._sdds_file.is_binary:
            self._read_binary_data()
        else:
            self._read_ascii_data()

    def _read_binary_data(self):
        self._sdds_file.row_count = self._read_binary_int()
        for key in self._sdds_file.get_parameters().keys():
            self._read_binary_parameter_value(self._sdds_file.get_parameters()[key])
        for key in self._sdds_file.get_arrays().keys():
            self._read_binary_array_values(self._sdds_file.get_arrays()[key])
        for key in self._sdds_file.get_columns().keys():
            pass  # TODO

    def _binary_read_functions(self):
        return {
            SddsTypes.Types.BOOLEAN: self._read_binary_boolean,
            SddsTypes.Types.BYTE: self._read_binary_byte,
            SddsTypes.Types.CHAR: self._read_binary_char,
            SddsTypes.Types.DOUBLE: self._read_binary_double,
            SddsTypes.Types.FLOAT: self._read_binary_float,
            SddsTypes.Types.INT: self._read_binary_int,
            SddsTypes.Types.LONG: self._read_binary_long,
            SddsTypes.Types.SHORT: self._read_binary_short,
            SddsTypes.Types.STRING: self._read_binary_string,
        }

    def _read_binary_parameter_value(self, parameter):
        read_function = self._binary_read_functions()[parameter.type]
        parameter.value = read_function()[0]
        LOGGER.debug(" ".join(["Value for parameter",
                               parameter.name, str(parameter.value)]))

    # TODO: > or < depending on the endianess in the header
    # > Big-endian
    # < Little-endian
    def _read_binary_boolean(self, bytes=SddsTypes.TYPES_SIZES[SddsTypes.Types.BOOLEAN]):
        return np.frombuffer(self._lines.read(bytes), dtype=np.bool)

    def _read_binary_byte(self):
        return np.frombuffer(self._lines.read(bytes), dtype=np.dtype(">i1"))

    def _read_binary_char(self, bytes=SddsTypes.TYPES_SIZES[SddsTypes.Types.CHAR]):
        return self._lines.read(bytes)

    def _read_binary_double(self, bytes=SddsTypes.TYPES_SIZES[SddsTypes.Types.DOUBLE]):
        return np.frombuffer(self._lines.read(bytes), dtype=np.dtype(">d"))

    def _read_binary_float(self, bytes=SddsTypes.TYPES_SIZES[SddsTypes.Types.FLOAT]):
        return np.frombuffer(self._lines.read(bytes), dtype=np.dtype(">f"))

    def _read_binary_int(self, bytes=SddsTypes.TYPES_SIZES[SddsTypes.Types.INT]):
        return np.frombuffer(self._lines.read(bytes), dtype=np.dtype(">i"))

    def _read_binary_long(self, bytes=SddsTypes.TYPES_SIZES[SddsTypes.Types.LONG]):
        return np.frombuffer(self._lines.read(bytes), dtype=np.dtype(">i8"))

    def _read_binary_short(self, bytes=SddsTypes.TYPES_SIZES[SddsTypes.Types.SHORT]):
        return np.frombuffer(self._lines.read(bytes), dtype=np.dtype(">i2"))

    def _read_binary_string(self, bytes=SddsTypes.TYPES_SIZES[SddsTypes.Types.CHAR]):
        return self._lines.read(bytes)

    def _read_binary_array_values(self, array):
        dimensions = array.dimensions
        for i in range(len(dimensions)):
            dimensions[i] = self._read_binary_int()[0]
        array_size = 1
        for dimension in dimensions:
            array_size *= dimension
        if array.type == SddsTypes.Types.STRING:
            values = []
            for str_index in range(array_size):
                if "u1" == array.modifier:
                    str_len = self._read_binary_byte()
                elif "i2" == array.modifier:
                    str_len = self._read_binary_short()
                str_len = self._read_binary_int()[0]
                values.append(self._read_binary_string(bytes=str_len))
            array.values = values
        else:
            bytes = array_size * SddsTypes.TYPES_SIZES[array.type]
            values = self._binary_read_functions()[array.type](bytes=bytes)
            array.values = values
        LOGGER.debug(" ".join(["Values for array", array.name,
                               "length", str(len(array.values)),
                               str(array.values)]))

    def _read_ascii_data(self):
        raise NotImplementedError("ASCII file reading has not been implemented...")


class SddsParameter(object):
    def __init__(self, name, type, type_name, units, symbol, modifier, format_string, description):
        self.name = name
        self.type = type
        self.type_name = type_name
        self.units = units
        self.symbol = symbol
        self.modifier = modifier
        self.format_string = format_string
        self.description = description


class SddsArray(object):
    def __init__(self, name, type, type_name, dimensions, units, symbol, modifier, format_string, description, group):
        self.name = name
        self.type = type
        self.type_name = type_name
        self.dimensions = dimensions
        self.units = units
        self.symbol = symbol
        self.modifier = modifier
        self.format_string = format_string
        self.description = description
        self.group = group


class SddsColumn(object):
    def __init__(self, name, type, type_name, units, symbol, modifier, format_string, description):
        self.name = name
        self.type = type
        self.type_name = type_name
        self.units = units
        self.symbol = symbol
        self.modifier = modifier
        self.format_string = format_string
        self.description = description


class SddsFile(object):

    def __init__(self):
        self._parameters = OrderedDict()
        self._arrays = OrderedDict()
        self._columns = OrderedDict()

    def define_parameter(
            self, name, type, type_name,
            units, symbol, modifier, format_string, description
    ):
        self._parameters[name] = SddsParameter(
            name, type, type_name,
            units, symbol, modifier, format_string, description
        )

    def get_parameters(self):
        return self._parameters

    def define_array(
            self, name, type, type_name, dimensions,
            units, symbol, modifier, format_string, description, group
    ):
        self._arrays[name] = SddsArray(
            name, type, type_name, dimensions,
            units, symbol, modifier, format_string, description, group
        )

    def get_arrays(self):
        return self._arrays

    def define_column(
            self, name, type, type_name,
            units, symbol, modifier, format_string, description
    ):
        self._columns[name] = SddsColumn(
            name, type, type_name,
            units, symbol, modifier, format_string, description
        )

    def get_columns(self):
        return self._columns


if __name__ == "__main__":
    SddsReader(sys.argv[1])
