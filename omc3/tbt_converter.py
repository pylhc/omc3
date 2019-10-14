"""
Entrypoint tbt_converter
------------------------

Created on 29/08/19

:author: Lukas Malina

Top-level script, which converts turn-by-turn files from various formats to LHC binary SDDS files.
    Optionally, it can replicate files with added noise.

"""
from collections import OrderedDict
from datetime import datetime
from os.path import join, basename
from utils import logging_tools, iotools
from definitions import formats
import tbt

from generic_parser.entrypoint import entrypoint, EntryPointParameters, save_options_to_config
from utils.contexts import timeit

LOGGER = logging_tools.get_logger(__name__)

DEFAULT_CONFIG_FILENAME = "converter_{time:s}.ini"
PLANES = ("X", "Y")


def converter_params():
    params = EntryPointParameters()
    params.add_parameter(flags="--files", name="files", required=True, nargs='+',
                         help="TbT files to analyse")
    params.add_parameter(flags="--outputdir", name="outputdir", required=True,
                         help="Output directory.")
    params.add_parameter(flags="--tbt_datatype", name="tbt_datatype", type=str, default="lhc",
                         choices=list(tbt.handler.DATA_READERS.keys()),
                         help="Choose the datatype from which to import. ")
    params.add_parameter(flags="--replication", name="replication", type=int, default=1,
                         help="Number of copies with added noise")
    params.add_parameter(flags="--noise_to_add", name="noise_to_add", type=float,
                         default=0.0,
                         help="Sigma of added Gaussian noise")
    return params


@entrypoint(converter_params(), strict=True)
def converter_entrypoint(opt):
    """
    Converts turn-by-turn files from various formats to LHC binary SDDS files.
    Optionally can replicate files files with added noise.

    Converter Kwargs:
      - **files**: TbT files to convert

        Flags: **--files**
        Required: ``True``
      - **outputdir**: Output directory.

        Flags: **--outputdir**
        Required: ``True``
      - **tbt_datatype** *(str)*: Choose datatype from which to import (e.g LHC binary SDDS).

        Flags: **--tbt_datatype**
        Default: ``lhc``
      - **replication** *(int)*: Number of copies with added noise.

        Flags: **--replication**
        Default: ``1``
      - **noise_to_add** *(float)*: Sigma of added Gaussian noise.

        Flags: **--noise_to_add**
        Default: ``0.0``

    """
    if opt.replication < 1:
        raise ValueError("Number of replications lower than 1.")
    iotools.create_dirs(opt.outputdir)
    save_options_to_config(join(opt.outputdir, DEFAULT_CONFIG_FILENAME.format(
        time=datetime.utcnow().strftime(formats.TIME))), OrderedDict(sorted(opt.items())))
    _read_and_write_files(opt)


def _read_and_write_files(opt):
    for input_file in opt.files:
        tbt_data = tbt.read_tbt(input_file, datatype=opt.tbt_datatype)
        for i in range(opt.replication):
            suffix = f"_{i}" if opt.replication > 1 else ""
            tbt.write(join(opt.outputdir, f"{_file_name(input_file)}{suffix}"), tbt_data,
                      noise=opt.noise_to_add)


def _file_name(filename: str):
    return basename(filename)[:-5] if filename.endswith(".sdds") else basename(filename)


if __name__ == "__main__":
    converter_entrypoint()
