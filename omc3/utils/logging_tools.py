"""
Logging Tools
-------------

Functions for easier use of logging, like automatic logger setup
(see: :meth:`~utils.logging_tools.get_logger`).
"""
import datetime
import inspect
import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from io import StringIO
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL  # make them available directly

import numpy as np
import pandas as pd

DIVIDER = "|"
NEWLINE = "\n" + " " * 10  # levelname + divider + 2
BASIC_FORMAT = '%(levelname)7s {div:s} %(message)s {div:s} %(name)s'.format(div=DIVIDER)
COLOR_LEVEL = '\33[0m\33[38;2;150;150;255m'
COLOR_MESSAGE = '\33[0m'
COLOR_MESSAGE_LOW = '\33[0m\33[38;2;140;140;140m'
COLOR_WARN = '\33[0m\33[38;2;255;161;53m'
COLOR_ERROR = '\33[0m\33[38;2;216;31;42m'
COLOR_NAME = '\33[0m\33[38;2;80;80;80m'
COLOR_DIVIDER = '\33[0m\33[38;2;127;127;127m'
COLOR_RESET = '\33[0m'

MADX = DEBUG + 3

# Classes and Contexts #########################################################


class MaxFilter(object):
    """To get messages only up to a certain level."""
    def __init__(self, level):
        self.__level = level

    def filter(self, log_record):
        return log_record.levelno <= self.__level


class DebugMode(object):
    """
    Context Manager for the debug mode.
    Hint: Does not work with ``@contextmanager`` from contextlib (even though nicer code),
    as the ``_get_caller`` would find **contextlib.py**.

    Args:
        active (bool): Defines if this manager is doing anything. Defaults to ``True``.
        log_file (str): File to log into.
    """
    def __init__(self, active=True, log_file=None):
        self.active = active
        if active:
            # get current logger
            caller_file = _get_caller()
            current_module = _get_current_module(caller_file)

            self.logger = logging.getLogger(".".join([current_module, os.path.basename(caller_file)]))

            # set level to debug
            self.current_level = self.logger.getEffectiveLevel()
            self.logger.setLevel(DEBUG)
            self.logger.debug("Running in Debug-Mode.")

            # create logfile name:
            now = "{:s}_".format(datetime.datetime.now().isoformat())
            if log_file is None:
                log_file = os.path.abspath(caller_file).replace(".pyc", "").replace(".py",
                                                                                    "") + ".log"
            self.log_file = os.path.join(os.path.dirname(log_file), now + os.path.basename(log_file))
            self.logger.debug("Writing log to file '{:s}'.".format(self.log_file))

            # add handlers
            self.file_h = file_handler(self.log_file, level=DEBUG)
            self.console_h = stream_handler(level=DEBUG, max_level=DEBUG)
            self.mod_logger = logging.getLogger(current_module)
            self.mod_logger.addHandler(self.file_h)
            self.mod_logger.addHandler(self.console_h)

            # stop time
            self.start_time = time.time()

    def __enter__(self):
        return None

    def __exit__(self, *args, **kwargs):
        if self.active:
            # summarize
            time_used = time.time() - self.start_time
            log_id = "" if self.log_file is None else "'{:s}'".format(
                os.path.basename(self.log_file))
            self.logger.debug("Exiting Debug-Mode {:s} after {:f}s.".format(log_id, time_used))

            # revert everything
            self.logger.setLevel(self.current_level)
            self.mod_logger.removeHandler(self.file_h)
            self.mod_logger.removeHandler(self.console_h)


class TempFile(object):
    """
    Context Manager. Lets another function write into a temporary file and logs its contents.
    It won't open the file, so only the files path is returned.

    Args:
        file_path (str): Place to write the tempfile to.
        log_func (func): The function with which the content should be logged (e.g. LOG.info).
    """

    def __init__(self, file_path, log_func):
        self.path = file_path
        self.log_func = log_func

    def __enter__(self):
        return self.path

    def __exit__(self, value, traceback):
        try:
            with open(self.path, "r") as f:
                content = f.read()
            self.log_func("{:s}:\n".format(self.path) + content)
        except IOError:
            self.log_func("{:s}: -file does not exist-".format(self.path))
        else:
            os.remove(self.path)


@contextmanager
def log_pandas_settings_with_copy(log_func):
    """Logs pandas ``SettingsWithCopy`` warning to loc_func instead of printing the warning."""
    caller_line = inspect.currentframe().f_back.f_back.f_lineno  # one frame for contextmanager
    old_mode = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = 'warn'
    try:
        with warnings.catch_warnings(record=True) as warn_list:
            yield
        for w in warn_list:
            if not issubclass(w.category, pd.core.common.SettingWithCopyWarning):
                warnings.warn(w)
            else:
                message = w.message.args[0].split("\n")
                log_func("{:s} (l. {:d})".format(message[1], caller_line))
    finally:
        pd.options.mode.chained_assignment = old_mode


@contextmanager
def logging_silence():
    """Remove temporarily all loggers from root logger."""
    root_logger = getLogger("")
    handlers = list(root_logger.handlers)
    root_logger.handlers = []

    yield

    root_logger.handlers = handlers


@contextmanager
def unformatted_console_logging():
    """Log only to console and only unformatted."""
    with logging_silence():
        handler = stream_handler(level=NOTSET, fmt="%(message)s")
        rl = getLogger("")
        rl.addHandler(handler)

        yield

        rl.removeHandler(handler)


class TempStringLogger:
    """
    Temporarily log into a string that can be retrieved by ``get_log``.

    Args:
        module: module to log, defaults to the caller file.
        level: logging level, defaults to ``INFO``.
    """
    def __init__(self, module=None, level=INFO):
        if module is None:
            module = _get_caller_logger_name()

        self.stream = StringIO()
        self.level = level
        self.handler = stream_handler(stream=self.stream)
        self.log = getLogger(module)

    def __enter__(self):
        self.log.propagate = False
        self.log.setLevel(self.level)
        self.log.addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log.removeHandler(self.handler)

    def get_log(self):
        """ Get the log as string. """
        return self.stream.getvalue()


def odr_pprint(printer, odr_out):
    """Logs the odr output results. Adapted from odr_output pretty print."""
    old_opts = np.get_printoptions()
    np.set_printoptions(precision=2)
    covariance_str = str(odr_out.cov_beta).replace("\n", ",")
    odr_str = ('\nODR-Summary:\n'
               f'  Beta: {odr_out.beta}\n'
               f'  Beta Std Error: {odr_out.sd_beta}\n'
               f'  Beta Covariance: {covariance_str}\n'
               )
    if hasattr(odr_out, 'info'):
        odr_str += (f'  Residual Variance: {odr_out.res_var:.2e}\n'
                    f'  Inverse Condition #: {odr_out.inv_condnum:.2e}\n'
                    f'  Reason(s) for Halting:\n')
        for r in odr_out.stopreason:
            odr_str += f'    {r}\n'
    printer(odr_str)
    np.set_printoptions(**old_opts)


def list2str(list_: list) -> str:
    """Returns string representation of ``list_``, but without brackets."""
    return str(list_).lstrip("[").rstrip("]")

# Public Methods ###############################################################


def get_logger(name, level_root=DEBUG, level_console=None, fmt=BASIC_FORMAT, color=None):
    """
    Sets up logger if name is **__main__**. Returns logger based on module name.

    Args:
        name: only used to check if __name__ is __main__.
        level_root: main logging level, defaults to ``DEBUG``.
        level_console: console logging level, defaults to ``INFO``.
        fmt: Format of the logging. For default see ``BASIC_FORMAT``.
        color: If `None` colors are used if tty is detected.
              `False` will never use colors and `True` will always enforce them.

    Returns:
        Logger instance.
    """
    logger_name = _get_caller_logger_name()

    if name == "__main__":
        if level_console is None:
            level_console = DEBUG if sys.flags.debug else INFO

        # set up root logger
        root_logger = logging.getLogger("")
        root_logger.handlers = []  # remove handlers in case someone already created them
        root_logger.setLevel(level_root)

        logging.addLevelName(MADX, 'MADX')

        # print logs to the console
        root_logger.addHandler(
            stream_handler(
                level=max(level_console, DEBUG),
                max_level=INFO-1,
                fmt=_maybe_bring_color(fmt, DEBUG, color),
            )
        )

        root_logger.addHandler(
            stream_handler(
                level=max(level_console, INFO),
                max_level=WARNING-1,
                fmt=_maybe_bring_color(fmt, INFO, color),
            )
        )

        # print console warnings
        root_logger.addHandler(
            stream_handler(
                level=max(WARNING, level_console),
                max_level=ERROR-1,
                fmt=_maybe_bring_color(fmt, WARNING, color),
            )
        )

        # print errors to error-stream
        root_logger.addHandler(
            stream_handler(
                stream=sys.stderr,
                level=max(ERROR, level_console),
                fmt=_maybe_bring_color(fmt, ERROR, color),
            )
        )

    # logger for the current file
    return logging.getLogger(logger_name)


def file_handler(logfile, level=DEBUG, fmt=BASIC_FORMAT):
    """Convenience function so the caller does not have to import logging."""
    handler = logging.FileHandler(logfile, mode='w', )
    handler.setLevel(level)
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    return handler


def stream_handler(stream=sys.stdout, level=DEBUG, fmt=BASIC_FORMAT, max_level=None):
    """Convenience function so the caller does not have to import logging."""
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    console_formatter = logging.Formatter(fmt)
    handler.setFormatter(console_formatter)
    if max_level:
        handler.addFilter(MaxFilter(max_level))
    return handler


def add_module_handler(handler):
    """Add handler at current module level."""
    current_module = _get_current_module()
    logging.getLogger(current_module).addHandler(handler)


def add_root_handler(handler):
    """Add handler at root level."""
    logging.getLogger("").addHandler(handler)


def getLogger(name):
    """Convenience function so the caller does not have to import logging."""
    return logging.getLogger(name)


def get_my_logger_name():
    """Return the logger name for the caller."""
    return _get_caller_logger_name()


# Private Methods ##############################################################


def _get_caller():
    """Find the caller of the current log-function."""
    this_file, _ = os.path.splitext(__file__)
    caller_file = this_file
    caller_frame = inspect.currentframe()
    while this_file == caller_file:
        caller_frame = caller_frame.f_back
        (caller_file_full, _, _, _, _) = inspect.getframeinfo(caller_frame)
        caller_file, _ = os.path.splitext(caller_file_full)
    return caller_file


def _get_current_module(current_file=None):
    """Find the name of the current module."""
    if not current_file:
        current_file = _get_caller()
    path_parts = os.path.abspath(current_file).split(os.path.sep)

    repo_parts = os.path.abspath(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)
                    ).split(os.path.sep)

    current_module = '.'.join(path_parts[len(repo_parts):-1])
    return current_module


def _get_caller_logger_name():
    """Returns logger name of the caller."""
    caller_file = _get_caller()
    current_module = _get_current_module(caller_file)
    return ".".join([current_module, os.path.basename(caller_file)])


def _maybe_bring_color(format_string, colorlevel=INFO, color_flag=None):
    """Adds color to the logs (can only be used in a terminal)."""
    if color_flag is None:
        color_flag = _isatty()

    if not color_flag:
        return format_string

    level = "%(levelname)"
    message = "%(message)"
    name = "%(name)"

    if colorlevel <= WARNING:
        format_string = format_string.replace(level, COLOR_LEVEL + level)
    else:
        format_string = format_string.replace(level, COLOR_ERROR + level)

    if colorlevel <= DEBUG:
        format_string = format_string.replace(message, COLOR_MESSAGE_LOW + message)
    elif colorlevel <= INFO:
        format_string = format_string.replace(message, COLOR_MESSAGE + message)
    elif colorlevel <= WARNING:
        format_string = format_string.replace(message, COLOR_WARN + message)
    else:
        format_string = format_string.replace(message, COLOR_ERROR + message)

    format_string = format_string.replace(name, COLOR_NAME + name)
    format_string = format_string.replace(DIVIDER, COLOR_DIVIDER + DIVIDER)
    format_string = format_string + COLOR_RESET

    return format_string


def _isatty():
    """Checks if stdout is a tty, which means it should support color-codes."""
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
