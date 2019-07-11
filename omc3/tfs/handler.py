"""
handler
-------------------

Basic tfs-to-pandas io-functionality.
"""
from collections import OrderedDict
from os.path import basename, dirname
import logging
import pandas
import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

HEADER = "@"
NAMES = "*"
TYPES = "$"
COMMENTS = "#"
INDEX_ID = "INDEX&&&"
FLOAT_PARENTS = (float, np.floating)
INT_PARENTS = (int, np.integer, bool, np.bool_)
ID_TO_TYPE = {
    "%s": np.str,
    "%bpm_s": np.str,
    "%le": np.float64,
    "%f": np.float64,
    "%hd": np.int,
    "%d": np.int,
}
DEFAULT_COLUMN_WIDTH = 20
MIN_COLUMN_WIDTH = 10


class TfsDataFrame(pandas.DataFrame):
    """
    Class to hold the information of the built Pandas DataFrame,
    together with a way of getting the headers of the TFS file.
    To get a header value do: data_frame["header_name"] or
    data_frame.header_name.
    """
    _metadata = ["headers", "indx"]

    def __init__(self, *args, **kwargs):
        self.headers = kwargs.pop("headers", {})
        self.indx = _Indx(self)
        super(TfsDataFrame, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        try:
            return super(TfsDataFrame, self).__getitem__(key)
        except KeyError as e:
            try:
                return self.headers[key]
            except KeyError:
                raise KeyError(f"{key} is neither in the DataFrame nor in headers.")
            except TypeError:
                raise e

    def __getattr__(self, name):
        try:
            return super(TfsDataFrame, self).__getattr__(name)
        except AttributeError:
            try:
                return self.headers[name]
            except KeyError:
                raise AttributeError(f"{name} is neither in the DataFrame nor in headers.")

    @property
    def _constructor(self):
        return TfsDataFrame


class _Indx(object):
    """
    Helper class to mock the metaclass twiss.indx["element_name"]
    behaviour.
    """
    def __init__(self, tfs_data_frame):
        self._tfs_data_frame = tfs_data_frame

    def __getitem__(self, key):
        name_series = self._tfs_data_frame.NAME
        return name_series[name_series == key].index[0]


def read_tfs(tfs_path, index=None):
    """
    Parses the TFS table present in tfs_path and returns a custom Pandas DataFrame (TfsDataFrame).

    Args:
        tfs_path: path to the input TFS file
        index: Name of the column to set as index. If not given looks for INDEX_ID-column

    Returns:
        TfsDataFrame object
    """
    LOGGER.debug(f"Reading path: {tfs_path}")
    headers = OrderedDict()
    column_names = column_types = None
    rows_list = []
    with open(tfs_path, "r") as tfs_data:
        for line in tfs_data:
            parts = line.split()
            if len(parts) == 0:
                continue
            if parts[0] == HEADER:
                name, value = _parse_header(parts[1:])
                headers[name] = value
            elif parts[0] == NAMES:
                LOGGER.debug("Setting column names.")
                column_names = np.array(parts[1:])
            elif parts[0] == TYPES:
                LOGGER.debug("Setting column types.")
                column_types = _compute_types(parts[1:])
            elif parts[0] == COMMENTS:
                continue
            else:
                if column_names is None:
                    raise TfsFormatError("Column names have not been set.")
                if column_types is None:
                    raise TfsFormatError("Column types have not been set.")
                parts = [part.strip('"') for part in parts]
                rows_list.append(parts)
    data_frame = _create_data_frame(column_names, column_types, rows_list, headers)

    if index is not None:  # Use given column as index
        data_frame = data_frame.set_index(index)
    else:  # Try to find Index automatically
        index_column = [c for c in data_frame.columns if c.startswith(INDEX_ID)]
        if len(index_column) > 0:
            data_frame = data_frame.set_index(index_column)
            idx_name = index_column[0].replace(INDEX_ID, "")
            if idx_name == "":
                idx_name = None  # to remove it completely (Pandas makes a difference)
            data_frame = data_frame.rename_axis(idx_name)

    _validate(data_frame, f"from file {tfs_path:s}")
    return data_frame


def write_tfs(tfs_path, data_frame, headers_dict=None,
              save_index=False, colwidth=DEFAULT_COLUMN_WIDTH):
    """
    Writes the DataFrame into tfs_path with the headers_dict as
    headers dictionary. If you want to keep the order of the headers, use collections.OrderedDict.

    Args:
        tfs_path: path to the output TFS file
        data_frame: TfsDataFrame or pandas.DataFrame to save
        headers_dict: Headers of the data_frame, if empty tries to use data_frame.headers
        save_index: bool or string. Default: False
            If True, saves the index of the data_frame to a column identifiable by INDEX_ID.
            If string, it saves the index of the data_frame to a column named by string.
        colwidth: Column width
    """
    _validate(data_frame, f"to be written in {tfs_path:s}")
    if save_index:
        if isinstance(save_index, str):
            # saves index into column by name given
            idx_name = save_index
        else:
            # saves index into column, which can be found by INDEX_ID
            try:
                idx_name = INDEX_ID + data_frame.index.name
            except TypeError:
                idx_name = INDEX_ID
        data_frame.insert(0, idx_name, data_frame.index)
    LOGGER.debug(f"Attempting to write file: {basename(tfs_path)} in {dirname(tfs_path)}")

    if headers_dict is None:  # Tries to get headers from TfsDataFrame
        try:
            headers_dict = data_frame.headers
        except AttributeError:
            headers_dict = {}

    colwidth = max(MIN_COLUMN_WIDTH, colwidth)
    headers_str = _get_headers_str(headers_dict)
    colnames_str = _get_colnames_str(data_frame.columns, colwidth)
    coltypes_str = _get_coltypes_str(data_frame.dtypes, colwidth)
    data_str = _get_data_str(data_frame, colwidth)
    with open(tfs_path, "w") as tfs_data:
        tfs_data.write("\n".join((
            headers_str, colnames_str, coltypes_str, data_str
        )))


def _get_headers_str(headers_dict):
    return "\n".join(_get_header_line(name, headers_dict[name])
                     for name in headers_dict)


def _get_header_line(name, value):
    if not isinstance(name, str):
        raise ValueError(f"{name} is not a string")
    if isinstance(value, INT_PARENTS):
        return f"@ {name} %d {value}"
    elif isinstance(value, FLOAT_PARENTS):
        return f"@ {name} %le {value}"
    elif isinstance(value, str):
        return f"@ {name} %s \"{value}\""
    else:
        raise ValueError(f"{value} does not correspond to recognized types (string, float and int)")


def _get_colnames_str(colnames, colwidth):
    fmt = _get_row_fmt_str([None] * len(colnames), colwidth)
    return "* " + fmt.format(*colnames)


def _get_coltypes_str(types, colwidth):
    fmt = _get_row_fmt_str([str] * len(types), colwidth)
    return "$ " + fmt.format(*[_dtype_to_str(type_) for type_ in types])


def _get_data_str(data_frame, colwidth):
    format_strings = "  " + _get_row_fmt_str(data_frame.dtypes, colwidth)
    return "\n".join(
        data_frame.apply(lambda series: format_strings.format(*series), axis=1)
    )


def _get_row_fmt_str(dtypes, colwidth):
    return " ".join(
        "{" + f"{indx:d}:>{_dtype_to_format(type_, colwidth)}" + "}"
        for indx, type_ in enumerate(dtypes)
    )


class TfsFormatError(Exception):
    """Raised when wrong format is detected in the TFS file."""
    pass


def _create_data_frame(column_names, column_types, rows_list, headers):
    data_frame = TfsDataFrame(data=np.array(rows_list),
                              columns=column_names,
                              headers=headers)
    _assign_column_types(data_frame, column_names, column_types)
    return data_frame


def _assign_column_types(data_frame, column_names, column_types):
    names_to_types = dict(zip(column_names, column_types))
    for name in names_to_types:
        data_frame[name] = data_frame[name].astype(names_to_types[name])


def _compute_types(str_list):
    return [_id_to_type(string) for string in str_list]


def _parse_header(str_list):
    type_idx = next((idx for idx, part in enumerate(str_list) if part.startswith("%")), None)
    if type_idx is None:
        raise TfsFormatError("No data type found in header: '{}'".format(" ".join(str_list)))

    name = " ".join(str_list[0:type_idx])
    value_str = " ".join(str_list[(type_idx+1):])
    return name, _id_to_type(str_list[type_idx])(value_str.strip('"'))


def _id_to_type(type_str):
    try:
        return ID_TO_TYPE[type_str]
    except KeyError:
        if type_str.startswith("%") and type_str.endswith("s"):
            return str
        raise TfsFormatError(f"Unknown data type: {type_str}")


def _dtype_to_str(type_):
    if np.issubdtype(type_, np.integer) or np.issubdtype(type_, np.bool_):
        return "%d"
    elif np.issubdtype(type_, np.floating):
        return "%le"
    else:
        return "%s"


def _dtype_to_format(type_, colsize):
    if type_ is None:
        return f"{colsize}"
    if np.issubdtype(type_, np.integer) or np.issubdtype(type_, np.bool_):
        return f"{colsize}d"
    if np.issubdtype(type_, np.floating):
        return f"{colsize}.{colsize - len('-0.e-000')}g"
    return f"{colsize}s"


def _validate(data_frame, info_str=""):
    """ 
    Check if Dataframe contains finite values only 
    and both indices and columns are unique.  
    """
    def isnotfinite(x):
        try:
            return ~np.isfinite(x)
        except TypeError:  # most likely string
            try:
                return np.zeros(x.shape, dtype=bool)
            except AttributeError:  # single entry
                return np.zeros(1, dtype=bool)

    bool_df = data_frame.apply(isnotfinite)
    if bool_df.values.any():
        LOGGER.warning(f"DataFrame {info_str:s} contains non-physical values at Index: "
                       f"{bool_df.index[bool_df.any(axis='columns')].tolist()}")

    if not len(set(data_frame.index)) == len(data_frame.index):
        raise TfsFormatError("Indices are not unique.")

    if not len(set(data_frame.columns)) == len(data_frame.columns):
        raise TfsFormatError("Column names are not unique.")

    LOGGER.debug(f"DataFrame {info_str:s} validated.")
