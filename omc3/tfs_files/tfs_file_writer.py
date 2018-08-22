import numpy as np
from utils import logging_tools


LOG = logging_tools.get_logger(__name__)
# TODO check if directory, where to write exists?
FLOAT_PARENTS = (float, np.floating)
INT_PARENTS = (int, np.integer, bool, np.bool_)
_TYPE_CHAR = {"STRING": "%s", "FLOAT": "%le", "INT": "%d"}
_DEFAULT_PRECISION = 13


def significant_numbers(value, uncertainty):
    digits = -int(np.floor(np.log10(uncertainty)))
    sig_uncertainty = round(uncertainty, digits)
    if np.floor(uncertainty / 10 ** np.floor(np.log10(sig_uncertainty))) == 1:
        digits = digits + 1
        sig_uncertainty = round(uncertainty, digits)
    sig_value = round(value, digits)
    return f"{sig_value:.{max(digits, 0)}f}", f"{sig_uncertainty:.{max(digits, 0)}f}"


class TfsFileWriter(object):
    """
    This class represents a TFS file. It stores all header lines and the table and write
    all the content formatted at once by calling the write function.
    """
    # Indicates width of columns in output file.
    DEFAULT_COLUMN_WIDTH = 20
    MIN_COLUMN_WIDTH = 10

    def __init__(self, file_name, headers=None, column_width=DEFAULT_COLUMN_WIDTH):
        """
        Constructor

        Args:
            file_name (str): full path to file where it will be written
            headers (dict): dictionary of headers
            column_width (int): Indicates the width of each column in the file.
        """
        if not isinstance(file_name, str) or 0 == len(file_name):
            raise ValueError("File name is not valid: " + file_name)
        self._file_name = file_name

        if not isinstance(column_width, int) or column_width < TfsFileWriter.MIN_COLUMN_WIDTH:
            column_width = TfsFileWriter.DEFAULT_COLUMN_WIDTH
        if headers is not None:
            self._tfs_header_lines = [self.get_header_line(head_name, headers[head_name])
                                      for head_name in headers]
        else:
            self._tfs_header_lines = []
        self._column_width = column_width

        self._num_of_columns = 0
        self._list_of_column_data_types = []
        self._list_of_column_names = []
        self._list_of_table_rows = []

    def add_column_names(self, list_names):
        if len(self._list_of_column_data_types):
            if self.has_different_length(list_names):
                raise AttributeError(f"Number of columns does not match length of the given list "
                                     f"({self._file_name}).")
        else:
            self._num_of_columns = len(list_names)

        self._list_of_column_names.extend(list_names)

    def add_column_datatypes(self, list_datatypes):
        if len(self._list_of_column_names):
            if self.has_different_length(list_datatypes):
                raise AttributeError(f"Number of columns does not match length of the given list "
                                     f"({self._file_name}).")
        else:
            self._num_of_columns = len(list_datatypes)

        self._list_of_column_data_types.extend(list_datatypes)

    def add_table_row(self, list_row_entries):
        if not self.contains_names_and_types():
            raise TypeError(f"Before filling the table, set the names and datatypes"
                            f"({self._file_name}).")
        elif self.has_different_length(list_row_entries):
                raise TypeError(f"Number of entries does not match the number of columns"
                                f"({self._file_name}).")
        self._list_of_table_rows.append(list_row_entries)

    def write_to_file(self):
        """ Writes the stored data to the file with the given filename. """
        if not self.contains_names_and_types():
            LOG.error(f"{self._file_name}: Abort writing file. "
                      f"Cannot write file until column names and types are set.")
            return
        if 0 == len(self._list_of_table_rows):
            LOG.error(f"{self._file_name}: Abort writing file. No rows in table.")
            return

        # Header
        lines = self._tfs_header_lines[:]

        LOG.debug(f"{len(self._list_of_table_rows)} lines in tfs table")
        LOG.debug("{} rows in tfs table: {}".format(len(self._list_of_column_names),
                                                    " ".join(self._list_of_column_names)))
        format_for_titles = self._get_column_formatter(self._list_of_column_names, with_type=False)
        lines.append("* " + format_for_titles.format(*self._list_of_column_names))
        lines.append("$ " + format_for_titles.format(*self._list_of_column_data_types))
        data_format = self._get_column_formatter(self._list_of_column_data_types, with_type=True)
        for table_line in self._list_of_table_rows:
            lines.append("  " + data_format.format(*table_line))
        with open(self._file_name, 'w') as tfs_file:
            tfs_file.write("\n".join(lines))

    def _get_column_formatter(self, list_of_names, with_type):
        def type_fmt(s):
            if with_type:
                return self.get_type_format(s, self._column_width)
            return f"{self._column_width:d}"

        return " ".join("{" + "{:d}:>".format(indx) + type_fmt(ctype) + "}"
                        for indx, ctype in enumerate(list_of_names))

    def has_different_length(self, a_list):
        return self._num_of_columns != len(a_list)

    def contains_names_and_types(self):
        return len(self._list_of_column_names) and len(self._list_of_column_data_types)

    @staticmethod
    def get_header_line(name, value):
        # TODO types can be a global dictionary
        if not isinstance(name, str):
            raise ValueError(f"{name} is not a string")
        if isinstance(value, INT_PARENTS):
            return f"@ {name} %d {value}"
        elif isinstance(value, FLOAT_PARENTS):
            return f"@ {name} %le {value}"
        elif isinstance(value, str):
            return f"@ {name} %s \"{value}\""
        else:
            raise ValueError(f"{value} does not correspond to any _TfsDataType")

    @staticmethod
    def get_type_format(tfs_type, width=None):
        if tfs_type not in _TYPE_CHAR.values():
            raise ValueError(f"Invalid type {tfs_type}")
        if width is None:
            width, precision = "", _DEFAULT_PRECISION
        else:
            precision = width - 7
        if tfs_type == _TYPE_CHAR["STRING"]:
            return f"{width}s"
        return f"{width}.{precision}g"
