"""
Entrypoint madx_wrapper
--------------------------

Runs MADX with a file or a string as an input, it processes @required macros.
If defined, writes the processed MADX script and logging output into files.
"""
from os.path import abspath, join, dirname, pardir
import os
import sys
import re
import subprocess
import optparse
import contextlib
from tempfile import mkstemp

from utils import logging_tools
LOG = logging_tools.get_logger(__name__)

LIB = abspath(join(dirname(__file__), "lib"))
_LOCAL_PATH = join(dirname(__file__), pardir, "bin")

if "darwin" in sys.platform:
    _MADX_BIN = "madx-macosx64-intel"
elif "win" in sys.platform:
    _MADX_BIN = "madx-win64-gnu.exe"
else:
    _MADX_BIN = "madx-linux64-gnu"

MADX_PATH = abspath(join(_LOCAL_PATH, _MADX_BIN))


class MadxError(Exception):
    pass


# Arguments ##################################################################


def _parse_args():
    parser = optparse.OptionParser()
    parser.add_option("-f", "--file", metavar="FILE", dest="file",
                      help="The file with the annotated MADX input to run.")
    parser.add_option("-o", "--output", metavar="OUTPUT", dest="output",
                      help="If defined, it will write the processed MADX script into the file.")
    parser.add_option("-l", "--log", metavar="LOG", dest="log",
                      help="File where to write the MADX log output.")
    parser.add_option("-m", "--madx", metavar="MADX", dest="madx_path",
                      help="Path to the MAD-X executable to use", default=MADX_PATH)
    parser.add_option("-c", "--cwd", metavar="CWD", dest="cwd",
                      help="Set current working directory")
    (options, args) = parser.parse_args()
    if len(args) > 1 or ((options.file is None) == (len(args) == 0)):
        raise IOError("No input found: either pass the file as first parameter or use --file (-f)")
    if len(args) == 1:
        return args[0], options.output, options.log, options.madx_path, options.cwd
    return options.file, options.output, options.log, options.madx_path, options.cwd


# Main Methods ###############################################################


def resolve_and_run_file(input_file, output_file=None, log_file=None,
                         madx_path=MADX_PATH, cwd=None):
    """Runs MADX in a subprocess.

    Attributes:
        input_file: MADX input file
        output_file: If given writes resolved MADX script.
        log_file: If given writes MADX logging output.
        madx_path: Path to MADX executable
    """
    input_string = _read_input_file(input_file)
    resolve_and_run_string(input_string, output_file=output_file, log_file=log_file,
                                  madx_path=madx_path, cwd=cwd)


def resolve_and_run_string(input_string, output_file=None, log_file=None,
                           madx_path=MADX_PATH, cwd=None):
    """Runs MADX in a subprocess.

    Attributes:
        input_string: MADX input string
        output_file: If given writes resolved MADX script.
        log_file: If given writes MADX logging output.
        madx_path: Path to MADX executable
    """
    _check_log_and_output_files(output_file, log_file)
    full_madx_script = _resolve(input_string)
    _run(full_madx_script, log_file, output_file, madx_path, cwd)


# Main Private Methods ######################################################


def _run(full_madx_script, log_file=None, output_file=None, madx_path=MADX_PATH, cwd=None):
    """ Starts the madx-process """
    with _madx_input_wrapper(full_madx_script, output_file) as madx_jobfile:
        process = subprocess.Popen([madx_path, madx_jobfile], shell=False,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
        with _logfile_wrapper(log_file) as log_handler, process.stdout:
            for line in process.stdout:
                log_handler(line.decode('utf-8'))
        status = process.wait()

    if status:
        _raise_madx_error(log=log_file, file=output_file)


# Macro Handling #############################################################


def _resolve(input_string):
    """Resolves the !@requires annotations of the input_string, and returns the resulting script."""
    macro_calls = "option, -echo;\n" + _resolve_required_macros(input_string) + "option, echo;\n\n"
    full_madx_script = macro_calls + input_string
    return full_madx_script


def _resolve_required_macros(file_content):
    """
    Recursively searches for "!@require lib" MADX annotations in the input script,
    adding the required macros library calls to the script header.
    """
    call_commands = ""
    for line in file_content.split("\n"):
        match = re.search("^!@require\s+([^\s]+).*$", line)
        if match is not None:
            required_macros = _add_macro_lib_ending(match.group(1))
            required_macros_file = abspath(join(LIB, required_macros))
            call_commands += _resolve_required_macros(_read_input_file(required_macros_file))
            call_commands += "call, file = \"" + required_macros_file + "\";\n"
    return call_commands


def _add_macro_lib_ending(macro_lib_name):
    macro_lib_name = macro_lib_name.strip()
    if macro_lib_name.endswith(".macros.madx"):
        return macro_lib_name
    else:
        return macro_lib_name + ".macros.madx"


# Wrapper ####################################################################


def _read_input_file(input_file):
    with open(input_file) as text_file:
        return text_file.read()


def _check_log_and_output_files(output_file, log_file):
    if output_file is not None:
        open(output_file, "a").close()
    if log_file is not None:
        open(log_file, "a").close()


@contextlib.contextmanager
def _logfile_wrapper(file_path=None):
    """ Logs into file and debug if file is given, into info otherwise """
    if file_path is None:
        def log_handler(line):
            line = line.rstrip()
            if len(line):
                LOG.info(line)
        yield log_handler
    else:
        with open(file_path, "w") as log_file:
            def log_handler(line):
                log_file.write(line)
                line = line.rstrip()
                if len(line):
                    LOG.debug(line)
            yield log_handler


@contextlib.contextmanager
def _madx_input_wrapper(content, file_path=None):
    """ Writes content into an output file and returns filepath.

    If file_path is not given, the output file is temporary and will be deleted afterwards.
    """
    if file_path is None:
        temp_file = True
        fd, file_path = mkstemp(suffix=".madx", prefix="job.", text=True)
        os.close(fd)  # close file descriptor
        if content:
            with open(file_path, "w") as f:
                f.write(content)
    else:
        temp_file = False
        with open(file_path, "w") as f:
            f.write(content)
    try:
        yield file_path
    finally:
        if temp_file:
            os.remove(file_path)


def _raise_madx_error(log=None, file=None):
    """ Rasing Error Wrapper

    Extracts extra info from log and output file if given.
    """
    message = "MADX run failed."
    if log is not None:
        try:
            with open(log, "r") as f:
                content = f.readlines()
            if content[-1].startswith("+="):
                message += " '{:s}'.".format(content[-1].replace("+=+=+=", "").strip())
        except (IOError, IndexError):
            pass

    if file is not None:
        message += " Run on File: '{:s}'.".format(file)

    raise MadxError(message)


# Script Mode ################################################################


if __name__ == "__main__":
    resolve_and_run_file(*_parse_args())
