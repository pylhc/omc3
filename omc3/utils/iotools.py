"""
IO Tools
--------

Helper functions for input/output issues.
"""
import sys
from typing import Iterable, Any, Union

import re

import json
import os
import shutil
from pathlib import Path

from generic_parser.entry_datatypes import get_instance_faker_meta, get_multi_class
from generic_parser.entrypoint_parser import save_options_to_config
from pandas import DataFrame
from tfs import TfsDataFrame

from omc3.definitions import formats
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


def copy_content_of_dir(src_dir, dst_dir):
    """Copies all files and directories from ``src_dir`` to ``dst_dir``."""
    if not os.path.isdir(src_dir):
        return

    create_dirs(dst_dir)

    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dst_dir, item)
        copy_item(src_item, dst_item)


def create_dirs(path_to_dir):
    """Creates all dirs to ``path_to_dir`` if not exists."""
    path_to_dir = Path(path_to_dir)
    if not path_to_dir.exists():
        path_to_dir.mkdir(parents=True)
        LOG.debug("Created directory structure: '{}'".format(path_to_dir))


def copy_item(src_item, dest):
    """
    Copies a file or a directory to ``dest``, which may be a directory.
    If ``src_item`` is a directory then all containing files and dirs will be copied into ``dest``.
    """
    try:
        if os.path.isfile(src_item):
            shutil.copy2(src_item, dest)
        elif os.path.isdir(src_item):
            copy_content_of_dir(src_item, dest)
        else:
            raise IOError
    except IOError:
        LOG.error("Could not copy item because of IOError. Item: '{}'".format(src_item))


def glob_regex(path: Path, pattern: str) -> "filter object":
    """ Do a glob on the given `path` based on the regular expression `pattern`.

    Args:
        path (Path): Folder path to look in.
        pattern (str): Pattern to match.

    """
    return filter(re.compile(pattern).match, (str(p) for p in path.glob("*")))


class PathOrStr(metaclass=get_instance_faker_meta(Path, str)):
    """A class that behaves like a Path when possible, otherwise like a string."""
    def __new__(cls, value):
        return Path(strip_quotes(value))


class PathOrStrOrDataFrame(metaclass=get_instance_faker_meta(TfsDataFrame, Path, str)):
    """A class that behaves like a Path when possible, otherwise like a string."""
    def __new__(cls, value):
        value = strip_quotes(value)
        try:
            return Path(value)
        except TypeError:
            TfsDataFrame(value)


class UnionPathStr(metaclass=get_instance_faker_meta(Path, str)):
    """A class that can be used as Path and string parser input, but does not convert."""
    def __new__(cls, value):
        return


class UnionPathStrInt(metaclass=get_instance_faker_meta(Path, str, int)):
    """A class that can be used as Path, string, int parser input, but does not convert.
    Very special and used e.g. in the BBQ Input."""
    def __new__(cls, value):
        return strip_quotes(value)


class OptionalStr(metaclass=get_instance_faker_meta(str, type(None))):
    """A class that allows `str` or `None`.
    Can be used in string-lists when individual entries can be `None`."""
    def __new__(cls, value):
        return strip_quotes(value)


"""A class that allows `float`, 'int' or `None`.
Can be used in numeric-lists when individual entries can be `None`."""
OptionalFloat = get_multi_class(float, int, type(None))


def strip_quotes(value: Any) -> Any:
    """Strip quotes around string-objects. If not a string, nothing
    is changed. This is because the input from commandline or json files
    could be surrounded by quotes (if they are strings).
    The dict-parser removes them automatically as well.
    This behaviour is important for basically every string-faker!

    Args:
        value (Any): The input value that goes into the function.
                     Can be of any type.

    Returns:
        If the input was a string, then it will be the string with stripped
        quotes (if there were any). Otherwise just the value.

    """
    if isinstance(value, str):
        value = value.strip("\'\"")  # behavior like dict-parser, IMPORTANT FOR EVERY STRING-FAKER
    return value


def convert_paths_in_dict_to_strings(dict_: dict) -> dict:
    """Converts all Paths in the dict to strings, including those in iterables."""
    dict_ = dict_.copy()
    for key, value in dict_.items():
        if isinstance(value, Path):
            dict_[key] = str(value)
        else:
            try:
                list_ = list(value)
            except TypeError:
                pass
            else:
                has_changed = False
                for idx, item in enumerate(list_):
                    if isinstance(item, Path):
                        list_[idx] = str(item)
                        has_changed = True
                if has_changed:
                    dict_[key] = list_
    return dict_


def remove_none_dict_entries(dict_: dict) -> dict:
    """
    Removes ``None`` entries from dict. This can be used as a workaround to
    https://github.com/pylhc/generic_parser/issues/26.
    """
    return {key: value for key, value in dict_.items() if value is not None}


def maybe_add_command(opt: dict, script: str) -> dict:
    """ Add a comment ';command' to the opt-dict,
    which is gotten from sys.argv but only if the executed file
    equals the given script.

    Args:
        opt (dict): Options datastructure
        script (str): Name of the script that called save_config.

    Returns:
        Updated dict with ;command entry, or
        if the script names were different, just the original opt.
    """
    if script == sys.argv[0]:
        opt[";command"] = " ".join([sys.executable] + sys.argv)  # will be sorted to the beginning below
    return opt


def save_config(output_dir: Path, opt: dict, script: str,
                unknown_opt: Union[dict, list] = None):
    """
    Quick wrapper for ``save_options_to_config``.

    Args:
        output_dir (Path): Path to the output directory (does not need to exist).
        opt (dict): opt-structure to be saved.
        script (str): path/name of the invoking script (becomes name of the .ini) usually
            ``__file__``.
        unknown_opt (dict|list): un-parsed opt-structure to be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # opt = remove_none_dict_entries(opt)  # fixed in 2020
    opt = convert_paths_in_dict_to_strings(opt)
    opt = maybe_add_command(opt, script)
    save_options_to_config(
        output_dir / formats.get_config_filename(script),
        dict(sorted(opt.items())),
        unknown=unknown_opt
    )
