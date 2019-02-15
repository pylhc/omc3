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
from utils import logging_tools

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
    except IOError:
        LOG.error("Could not delete item because of IOError. Item: '{}'".format(path_to_item))
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


def get_absolute_path_to_betabeat_root():
    return os.path.abspath(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)
                    )


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


def json_dumps_readable(json_outfile, object):
    """ This is how you write a beautiful json file
    
    Args:
        json_outfile: File to write
        object: object to dump
    """
    object = json.dumps(object).replace(", ", ",\n    "
                              ).replace("[", "[\n    "
                              ).replace("],\n    ", "],\n\n"
                              ).replace("{", "{\n"
                              ).replace("}", "\n}")
    with open(json_outfile, "w") as json_file:
        json_file.write(object)
