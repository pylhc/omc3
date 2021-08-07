import sys
import os
import logging
from logging import StreamHandler, FileHandler


def set_up_console_logger(logger):
    logger.setLevel(logging.DEBUG)

    console_handler = StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    return logger


def add_file_handler(log_dir):
    logger = logging.getLogger("")
    log_file = os.path.join(log_dir, "log.txt")
    file_handler = FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_get_formatter())
    logger.addHandler(file_handler)
    logger.info("Set up debug log file in: " +
                os.path.abspath(log_file))


def _get_formatter():
    formatter = logging.Formatter(
        "%(name)s %(asctime)s %(levelname)s %(message)s"
    )
    formatter.datefmt = '%d/%m/%Y %H:%M:%S'
    return formatter
