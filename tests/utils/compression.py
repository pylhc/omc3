from pathlib import Path
import logging

import tfs
from omc3.model.constants import TWISS_ELEMENTS_DAT, TWISS_DAT, TWISS_ADT_DAT, TWISS_AC_DAT

LOGGER = logging.getLogger(__name__)

def file_compression(file_path: Path, compress: bool = True) -> None:
    """Compress or decompress a file. 
    If compress is True, the file will be compressed, otherwise it will be decompressed.
    Will only compress if the file exists and emits a warning to the logger if the file does not exist.

    Args:
        file_path (Path): The file to compress or decompress.
        compress (bool, optional): Whether to compress the file. Defaults to True.
    """
    read_path  = file_path.with_suffix(".bz2") if not compress else file_path
    write_path = file_path.with_suffix(".bz2") if     compress else file_path
    if read_path.exists():
        data = tfs.read(read_path)
        tfs.write(write_path, data)
    else:
        LOGGER.warning(f"File {read_path} does not exist. Cannot {'' if compress else 'de'}compress.")

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
    

def model_compression(model_dir: Path, compress: bool = True) -> None:
    """Compress or decompress the twiss files in the model directory.
    
    Args:
        model_dir (Path): The directory containing the twiss files.
        compress (bool, optional): Whether to compress the files. Defaults to True.
    """
    files = model_files(model_dir)
    for file in files:
        file_compression(file, compress)

def compress_model(model_dir: Path) -> None:
    """Compress the twiss files in the model directory.

    Args:
        model_dir (Path): The directory containing the twiss files.
    """
    model_compression(model_dir, compress=True)

def decompress_model(model_dir: Path) -> None:
    """Decompress the twiss files in the model directory.
    
    Args:
        model_dir (Path): The directory containing the twiss files.
    """
    model_compression(model_dir, compress=False)

def delete_decompressed_files(model_dir: Path) -> None:
    """Delete the decompressed twiss files in the model directory.
    
    Args:
        model_dir (Path): The directory containing the twiss files.
    """
    files = model_files(model_dir)
    for file in files:
        if file.exists():
            file.unlink()
        else:
            LOGGER.warning(f"File {file} does not exist. Cannot delete.")