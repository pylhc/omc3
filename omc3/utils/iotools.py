"""
IO Tools
--------

Helper functions for input/output issues.
"""
import json
import os
import shutil
from pathlib import Path

from generic_parser.entry_datatypes import get_instance_faker_meta
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


class PathOrStr(metaclass=get_instance_faker_meta(Path, str)):
    """A class that behaves like a Path when possible, otherwise like a string."""
    def __new__(cls, value):
        if isinstance(value, str):
            value = value.strip("\'\"")  # behavior like dict-parser, IMPORTANT FOR EVERY STRING-FAKER
        return Path(value)


class PathOrStrOrDataFrame(metaclass=get_instance_faker_meta(TfsDataFrame, Path, str)):
    """A class that behaves like a Path when possible, otherwise like a string."""
    def __new__(cls, value):
        if isinstance(value, str):
            value = value.strip("\'\"")  # behavior like dict-parser, IMPORTANT FOR EVERY STRING-FAKER
        try:
            return Path(value)
        except TypeError:
            TfsDataFrame(value)


class UnionPathStr(metaclass=get_instance_faker_meta(Path, str)):
    """A class that can be used as Path and string parser input, but does not convert."""
    def __new__(cls, value):
        if isinstance(value, str):
            value = value.strip("\'\"")  # behavior like dict-parser, IMPORTANT FOR EVERY STRING-FAKER
        return value


class UnionPathStrInt(metaclass=get_instance_faker_meta(Path, str, int)):
    """A class that can be used as Path, string, int parser input, but does not convert.
    Very special and used e.g. in the BBQ Input."""
    def __new__(cls, value):
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


def save_config(output_dir: Path, opt: dict, script: str):
    """
    Quick wrapper for ``save_options_to_config``.

    Args:
        output_dir (Path): Path to the output directory (does not need to exist).
        opt (dict): opt-structure to be saved.
        script (str): path/name of the invoking script (becomes name of the .ini) usually
            ``__file__``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    opt = remove_none_dict_entries(opt)  # temporary fix (see docstring)
    opt = convert_paths_in_dict_to_strings(opt)
    save_options_to_config(output_dir / formats.get_config_filename(script),
                           dict(sorted(opt.items()))
                           )
