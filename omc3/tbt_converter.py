"""
Entrypoint tbt_converter
-------------------------

Created on 29/08/19

:author: Lukas Malina

Top-level script, which converts turn-by-turn files from various formats to LHC binary SDDS files.
    Optionally, it can replicate files with added noise.

"""
from collections import OrderedDict
from datetime import datetime
from os.path import basename, join

from generic_parser.entrypoint_parser import (EntryPointParameters, entrypoint,
                                              save_options_to_config)

from omc3 import tbt
from omc3.definitions import formats
from omc3.utils import iotools, logging_tools

LOGGER = logging_tools.get_logger(__name__)

DEFAULT_CONFIG_FILENAME = "converter_{time:s}.ini"
PLANES = ("X", "Y")


def converter_params():
    params = EntryPointParameters()
    params.add_parameter(name="files", required=True, nargs='+', help="TbT files to analyse")
    params.add_parameter(name="outputdir", required=True, help="Output directory.")
    params.add_parameter(name="tbt_datatype", type=str, default="lhc",
                         choices=list(tbt.handler.DATA_READERS.keys()),
                         help="Choose the datatype from which to import. ")
    params.add_parameter(name="realizations", type=int, default=1,
                         help="Number of copies with added noise")
    params.add_parameter(name="noise_levels", nargs='+',
                         help="Sigma of added Gaussian noise")
    params.add_parameter(name="use_average", action="store_true",
                         help="If set, returned sdds only contains the average over all particle/bunches.")
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
      - **realizations** *(int)*: Number of copies with added noise.

        Flags: **--realizations**
        Default: ``1``
      - **noise_levels** *(float)*: Sigma of added Gaussian noise.

        Flags: **--noise_levels**
        Default: None
      - **use_average** *(bool)*: If set, returned sdds only contains the average over all particle/bunches.

        Flags: **--use_average**
        Default: False
    """
    if opt.realizations < 1:
        raise ValueError("Number of realizations lower than 1.")
    iotools.create_dirs(opt.outputdir)
    save_options_to_config(join(opt.outputdir, DEFAULT_CONFIG_FILENAME.format(
        time=datetime.utcnow().strftime(formats.TIME))), OrderedDict(sorted(opt.items())))
    _read_and_write_files(opt)


def _read_and_write_files(opt):
    for input_file in opt.files:
        tbt_data = tbt.read_tbt(input_file, datatype=opt.tbt_datatype)
        if opt.use_average:
            tbt_data = tbt.handler.generate_average_tbtdata(tbt_data)
        for i in range(opt.realizations):
            suffix = f"_r{i}" if opt.realizations > 1 else ""
            if opt.noise_levels is None:
                tbt.write(join(opt.outputdir, f"{_file_name(input_file)}{suffix}"), tbt_data=tbt_data)
            else:
                [tbt.write(join(opt.outputdir, f"{_file_name(input_file)}_n{n}{suffix}"), tbt_data=tbt_data, noise=n) for n in opt.noise_levels]


def _file_name(filename: str):
    return basename(filename)[:-5] if filename.endswith(".sdds") else basename(filename)


if __name__ == "__main__":
    converter_entrypoint()
