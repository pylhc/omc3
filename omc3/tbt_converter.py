"""
TbT Converter
-------------

Top-level script to convert turn-by-turn files from various formats to ``LHC`` binary SDDS files.
Optionally, it can replicate files with added noise.
"""
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Union

from generic_parser.entrypoint_parser import (
    EntryPointParameters,
    entrypoint,
    save_options_to_config,
)

import turn_by_turn as tbt
from omc3.definitions import formats
from omc3.utils import iotools, logging_tools

LOGGER = logging_tools.get_logger(__name__)

DEFAULT_CONFIG_FILENAME = "converter_{time:s}.ini"


def converter_params():
    params = EntryPointParameters()
    params.add_parameter(name="files", required=True, nargs="+", help="TbT files to analyse")
    params.add_parameter(name="outputdir", required=True, help="Output directory.")
    params.add_parameter(
        name="tbt_datatype",
        type=str,
        default="lhc",
        choices=list(tbt.io.TBT_MODULES.keys()),
        help="Choose the datatype from which to import. ",
    )
    params.add_parameter(
        name="output_datatype",
        type=str,
        default="lhc",
        choices=list(tbt.io.WRITERS),
        help="Choose the datatype to export. ",
    )
    params.add_parameter(
        name="realizations", type=int, default=1, help="Number of copies with added noise"
    )
    params.add_parameter(name="noise_levels", nargs="+", help="Sigma of added Gaussian noise")
    params.add_parameter(
        name="use_average",
        action="store_true",
        help="If set, returned sdds only contains the average over all particle/bunches.",
    )
    params.add_parameter(
        name="drop_elements",
        nargs="+",
        help="Names of elements to drop from the input file during conversion",
    )
    return params


@entrypoint(converter_params(), strict=True)
def converter_entrypoint(opt):
    """
    Converts turn-by-turn files from various formats to ``LHC`` binary SDDS files.
    Optionally can replicate files with added noise.

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
      - **output_datatype** *(str)*: Choose the datatype to export.

        Flags: **--output_datatype**
        Default: ``lhc``
      - **realizations** *(int)*: Number of copies with added noise.

        Flags: **--realizations**
        Default: ``1``
      - **noise_levels** *(float)*: Sigma of added Gaussian noise.

        Flags: **--noise_levels**
        Default: ``None``
      - **use_average** *(bool)*: If set, returned sdds only contains the average over all particle/bunches.

        Flags: **--use_average**
        Default: ``False``

      - **drop_elements**: Names of elements to drop from the input file during conversion.

        Flags: **--drop_elements**
        Default: ``None``
    """
    if opt.realizations < 1:
        raise ValueError("Number of realizations lower than 1.")
    iotools.create_dirs(opt.outputdir)
    save_options_to_config(
        str(
            Path(opt.outputdir)
            / DEFAULT_CONFIG_FILENAME.format(time=datetime.now(timezone.utc).strftime(formats.TIME))
        ),
        dict(sorted(opt.items())),
    )
    _read_and_write_files(opt)


def _read_and_write_files(opt):
    for input_file in opt.files:
        tbt_data = tbt.read_tbt(input_file, datatype=opt.tbt_datatype)
        if opt.drop_elements:
            tbt_data = _drop_elements(tbt_data, opt.drop_elements)
        if opt.use_average:
            tbt_data = tbt.utils.generate_average_tbtdata(tbt_data)
        for i in range(opt.realizations):
            suffix = f"_r{i}" if opt.realizations > 1 else ""
            if opt.noise_levels is None:
                tbt.write(
                    Path(opt.outputdir) / f"{_file_name_without_sdds(input_file)}{suffix}",
                    datatype=opt.output_datatype,
                    tbt_data=tbt_data,
                )
            else:
                for noise_level in opt.noise_levels:
                    tbt.write(
                        Path(opt.outputdir) / f"{_file_name_without_sdds(input_file)}_n{noise_level}{suffix}",
                        datatype=opt.output_datatype,
                        tbt_data=tbt_data,
                        noise=float(noise_level),
                    )


def _drop_elements(tbt_data: tbt.TbtData, elements_to_drop: Sequence[str]) -> tbt.TbtData:
    """
    Drops the provided elements from the matrices in the provided TbtData object.
    For any element not found in the matrices, a warning is logged and the element is skipped.

    Args:
        tbt_data (tbt.TbtData): a TbTData object from loading your turn-by-turn data from disk.
        elements_to_drop (Sequence[str]): list of elements to drop.

    Returns:
        A copied version of the provided TbtData object with the relevant element dropped from the matrices.
    """
    copied_data: tbt.TbtData = copy.deepcopy(tbt_data)
    LOGGER.info(f"Dropping the following unwanted elements: {', '.join(elements_to_drop)}")
    for element in elements_to_drop:
        LOGGER.debug(f"Dropping element '{element}'")
        try:
            for transverse_data in copied_data.matrices:
                transverse_data.X = transverse_data.X.drop(element)
                transverse_data.Y = transverse_data.Y.drop(element)
        except (KeyError, AttributeError):
            LOGGER.warning(f"Element '{element}' could not be found, skipped")
    return copied_data


def _file_name_without_sdds(filename: Union[str, Path]) -> str:
    """
    Returns the file name without suffix if the suffix is **.sdds**, else the filename.
    Previous elements in the file path are discarded.
    """
    filename = Path(filename)
    return filename.stem if filename.suffix == ".sdds" else filename.name


if __name__ == "__main__":
    converter_entrypoint()
