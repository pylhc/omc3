import logging
from pathlib import Path

import tfs

from omc3.model.constants import TWISS_AC_DAT, TWISS_ADT_DAT, TWISS_DAT, TWISS_ELEMENTS_DAT

LOGGER = logging.getLogger(__name__)

def model_files(model_dir: Path) -> list[Path]:
    """Return all possible twiss files in the model directory.
    
    Args:
        model_dir (Path): The directory containing the twiss files.
    
    Returns:
        list[Path]: The twiss files in the model directory.
    """
    return [
        model_dir / TWISS_ELEMENTS_DAT,
        model_dir / TWISS_DAT,
        model_dir / TWISS_AC_DAT,
        model_dir / TWISS_ADT_DAT,
    ]
    
def compress_model(model_dir: Path) -> None:
    """Compress the twiss files in the model directory.

    Args:
        model_dir (Path): The directory containing the twiss files.
    """
    files = model_files(model_dir)
    for file in files:
        read_path  = file
        write_path = model_dir / file.with_suffix(".bz2").name
        if read_path.exists():
            data = tfs.read(read_path)
            tfs.write(write_path, data)
        else:
            LOGGER.warning(f"File {read_path} does not exist. Cannot compress.")


def decompress_model(model_dir: Path) -> None:
    """Decompress the twiss files in the model directory.
    
    Args:
        model_dir (Path): The directory containing the twiss files.
    """
    files = model_files(model_dir)
    for file in files:
        read_path  = file.with_suffix(".bz2")
        write_path = model_dir / file.name
        if read_path.exists():
            data = tfs.read(read_path)
            tfs.write(write_path, data)
        else:
            LOGGER.warning(f"File {read_path} does not exist. Cannot decompress.")
