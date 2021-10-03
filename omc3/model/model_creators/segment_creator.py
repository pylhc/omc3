"""
Segment Creator
---------------

This module provides convenience functions for model creation of a ``segment``.
"""
import shutil
from pathlib import Path

from omc3.model.constants import MACROS_DIR, GENERAL_MACROS
from omc3.utils import logging_tools
from omc3.utils.iotools import create_dirs

LOGGER = logging_tools.get_logger(__name__)


class SegmentCreator(object):
    @classmethod
    def prepare_run(cls, accel):
        macros_path = accel.model_dir / MACROS_DIR
        create_dirs(macros_path)
        lib_path = Path(__file__).parent.parent.parent / "lib"
        shutil.copy(lib_path / GENERAL_MACROS, macros_path / GENERAL_MACROS)

    @classmethod
    def get_madx_script(cls, accel):
        madx_template = accel.get_file("segment.madx").read_text()
        replace_dict = {
            "MAIN_SEQ": accel.load_main_seq_madx(),  # LHC only
            "OPTICS_PATH": accel.modifiers,  # all
            "NUM_BEAM": accel.beam,  # LHC only
            "PATH": accel.model_dir,  # all
            "OUTPUT": accel.model_dir,  # Booster only
            "LABEL": accel.label,  # all
            "BETAKIND": accel.kind,  # all
            "STARTFROM": accel.start.name,  # all
            "ENDAT": accel.end.name,  # all
            "RING": accel.ring,  # Booster only
            "KINETICENERGY": accel.energy,  # PS only
            "FILES_DIR": accel.get_dir(),  # Booster and PS
            "NAT_TUNE_X": accel.nat_tunes[0],  # Booster and PS
            "NAT_TUNE_Y": accel.nat_tunes[1],  # Booster and PS
        }
        return madx_template % replace_dict
