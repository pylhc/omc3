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

from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


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
