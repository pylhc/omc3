"""
Module utils.iotools
---------------------

Created on 1 Jul 2013

utils.iotools.py holds helper functions for input/output issues. This module is not intended to
be executed.

Feel free to use and extend this module.

.. moduleauthor:: vimaier

"""

import json
import os
import shutil
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path

from generic_parser.entry_datatypes import get_instance_faker_meta
from generic_parser.entrypoint_parser import save_options_to_config

from omc3.definitions import formats
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


def delete_content_of_dir(path_to_dir):
    """
    Deletes all folders, files and symbolic links in given directory.
    :param string path_to_dir:
    """
    if not os.path.isdir(path_to_dir):
        return

    for item in os.listdir(path_to_dir):
        item_path = os.path.join(path_to_dir, item)
        delete_item(item_path)


def delete_item(path_to_item):
    """ Deletes the item given by path_to_item. It distinguishes between a file, a directory and a
    symbolic link.
    """
    try:
        if os.path.isfile(path_to_item):
            os.unlink(path_to_item)
        elif os.path.isdir(path_to_item):
            shutil.rmtree(path_to_item)
        elif os.path.islink(path_to_item):
            os.unlink(path_to_item)
    except OSError:
        LOG.error("Could not delete item because of OSError. Item: '{}'".format(path_to_item))


def copy_content_of_dir(src_dir, dst_dir):
    """ Copies all files and directories from src_dir to dst_dir. """
    if not os.path.isdir(src_dir):
        return

    create_dirs(dst_dir)

    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dst_dir, item)
        copy_item(src_item, dst_item)


def create_dirs(path_to_dir):
    """ Creates all dirs to path_to_dir if not exists. """
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
        LOG.debug("Created directory structure: '{}'".format(path_to_dir))


def copy_item(src_item, dest):
    """ Copies a file or a directory to dest. dest may be a directory.
    If src_item is a directory then all containing files and dirs will be copied into dest. """
    try:
        if os.path.isfile(src_item):
            shutil.copy2(src_item, dest)
        elif os.path.isdir(src_item):
            copy_content_of_dir(src_item, dest)
        else:
            raise IOError
    except IOError:
        LOG.error("Could not copy item because of IOError. Item: '{}'".format(src_item))


def deleteFilesWithoutGitignore(pathToDirectory):
    """
    Deletes all files in the given pathToDirectory except of the file with the name '.gitignore'

    :returns: bool -- True if the directory exists and the files are deleted otherwise False.
    """
    if not os.path.exists(pathToDirectory):
        return False

    filenames_list = os.listdir(pathToDirectory)

    for filename in filenames_list:
        if ".gitignore" != filename:
            os.remove( os.path.join(pathToDirectory,filename) )

    return True


def exists_directory(path_to_dir):
    return os.path.isdir(path_to_dir)


def not_exists_directory(path_to_dir):
    return not exists_directory(path_to_dir)


def no_dirs_exist(*dirs):
    return not dirs_exist(*dirs)


def dirs_exist(*dirs):
    for d in dirs:
        if not os.path.isdir(d):
            return False
    return True


def get_all_filenames_in_dir_and_subdirs(path_to_dir):
    """ Looks for files(not dirs) in dir and subdirs and returns them as a list.  """
    if not os.path.isdir(path_to_dir):
        return []
    result = []
    for root, sub_folders, files in os.walk(path_to_dir):  # @UnusedVariable
        result += files
    return result


def get_all_absolute_filenames_in_dir_and_subdirs(path_to_dir):
    """ Looks for files(not dirs) in dir and subdirs and returns them as a list.  """
    if not os.path.isdir(path_to_dir):
        return []
    abs_path_to_dir = os.path.abspath(path_to_dir)
    result = []
    for root, sub_folders, files in os.walk(abs_path_to_dir):  # @UnusedVariable
        for file_name in files:
            result.append(os.path.join(root, file_name))
    return result


def get_all_filenames_in_dir(path_to_dir):
    """ Looks for files in dir(not subdir) and returns them as a list """
    if not os.path.isdir(path_to_dir):
        return []
    result = []
    for item in os.listdir(path_to_dir):
        item_path = os.path.join(path_to_dir, item)
        if os.path.isfile(item_path):
            result.append(item)
    return result


def get_all_dir_names_in_dir(path_to_dir):
    """ Looks for directories in dir and returns them as a list """
    if not os.path.isdir(path_to_dir):
        return []
    result = []
    for item in os.listdir(path_to_dir):
        item_path = os.path.join(path_to_dir, item)
        if os.path.isdir(item_path):
            result.append(item)
    return result


def is_empty_dir(directory):
    return 0 == os.listdir(directory)


def is_not_empty_dir(directory):
    return not is_empty_dir(directory)


def read_all_lines_in_textfile(path_to_textfile):
    if not os.path.exists(path_to_textfile):
        LOG.error("File does not exist: '{}'".format(path_to_textfile))
        return ""
    with open(path_to_textfile) as textfile:
        return textfile.read()


def append_string_to_textfile(path_to_textfile, str_to_append):
    """ If file does not exist, a new file will be created. """
    with open(path_to_textfile, "a") as file_to_append:
        file_to_append.write(str_to_append)


def write_string_into_new_file(path_to_textfile, str_to_insert):
    """ An existing file will be truncated. """
    with open(path_to_textfile, "w") as new_file:
        new_file.write(str_to_insert)


def replace_keywords_in_textfile(path_to_textfile, dict_for_replacing, new_output_path=None):
    """
    This function replaces all keywords in a textfile with the corresponding values in the dictionary.
    E.g.: A textfile with the content "%(This)s will be replaced!" and the dict {"This":"xyz"} results
    to the change "xyz will be replaced!" in the textfile.

    Args:
        new_output_path: If new_output_path is None, then the source file will be replaced.
    """
    if new_output_path is None:
        destination_file = path_to_textfile
    else:
        destination_file = new_output_path

    all_lines = read_all_lines_in_textfile(path_to_textfile)
    lines_with_replaced_keys = all_lines % dict_for_replacing
    write_string_into_new_file(destination_file, lines_with_replaced_keys)


def json_dumps_readable(json_outfile, object_to_dump):
    """ This is how you write a beautiful json file
    
    Args:
        json_outfile: File to write
        object_to_dump: object to dump to json format
    """
    object_to_dump = json.dumps(object_to_dump).replace(", ", ",\n    "
                                                        ).replace("[", "[\n    "
                              ).replace("],\n    ", "],\n\n"
                              ).replace("{", "{\n"
                              ).replace("}", "\n}")
    with open(json_outfile, "w") as json_file:
        json_file.write(object_to_dump)


class PathOrStr(metaclass=get_instance_faker_meta(Path, str)):
    """ A class that behaves like a Path when possible, otherwise like a string."""
    def __new__(cls, value):
        if isinstance(value, str):
            value = value.strip("\'\"")  # behavior like dict-parser, IMPORTANT FOR EVERY STRING-FAKER
        return Path(value)


def convert_paths_in_dict_to_strings(dict_: dict) -> dict:
    """ Converts all Paths in the dict to strings. """
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
    """ Removes None entries from dict.
    This can be used as a workaround to
    https://github.com/pylhc/generic_parser/issues/26 """
    for key, value in list(dict_.items()):
        if value is None:
            del dict_[key]
    return dict_


def save_config(output_dir: Path, opt: dict, script: str):
    """  Quick wrapper for save_options_to_config.

    Args:
        output_dir (Path): Path to the output directory (does not need to exist)
        opt (dict): opt-structure to be saved
        script (str): path/name of the invoking script (becomes name of the .ini)
                      usually ``__file__``
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    opt = opt.copy()
    opt = remove_none_dict_entries(opt)  # temporary fix
    opt = convert_paths_in_dict_to_strings(opt)
    save_options_to_config(output_dir / formats.get_config_filename(script),
                           OrderedDict(sorted(opt.items()))
                           )
