"""
Segment Creator
---------------

This module provides convenience functions for model creation of a ``segment``.
"""
import shutil
from pathlib import Path

from omc3.model.constants import MACROS_DIR, GENERAL_MACROS, LHC_MACROS
from omc3.utils import logging_tools
from omc3.utils.iotools import create_dirs

LOGGER = logging_tools.get_logger(__name__)



class SegmentCreator(object):

    @classmethod
    def prepare_run(cls, instance, output_path):
        macros_path = Path(output_path) / MACROS_DIR
        create_dirs(macros_path)
        lib_path = Path(__file__).parent.parent/ "madx_macros"
        shutil.copy(lib_path / GENERAL_MACROS, macros_path / GENERAL_MACROS)

    @staticmethod
    def get_parameters():
        params = super(Lhc, Lhc).get_parameters()
        params.add_parameter(name="libs", type=str)
        return params

    @classmethod
    def get_madx_script(cls, instance, output_path):

        libs = f"call, file = '{output_path / MACROS_DIR / GENERAL_MACROS}';\n"
        libs = libs + f"call, file = '{output_path / MACROS_DIR / LHC_MACROS}';\n"
        madx_template = instance.get_file("segment.madx").read_text()
        print("vvv", instance.modifiers[0])
        replace_dict = {
            "MAIN_SEQ": instance.load_main_seq_madx(),  # LHC only
            "OPTICS_PATH": str(instance.modifiers[0]),  # all
            "NUM_BEAM": instance.beam,  # LHC only
            "PATH": output_path,  # all
            "OUTPUT": output_path,  # Booster only
            "LIB": libs,
            "LABEL": '',  # all
            "BETAKIND": instance.betainputfile,  # all
            "STARTFROM": instance.startbpm,  # all
            "ENDAT": instance.endbpm,  # all
            #"RING": instance.ring,  # Booster only
            #"KINETICENERGY": instance.energy,  # PS only
            #"FILES_DIR": instance.get_dir(),  # Booster and PS
            "NAT_TUNE_X": instance.nat_tunes[0],  # Booster and PS
            "NAT_TUNE_Y": instance.nat_tunes[1],  # Booster and PS
        }
        print(madx_template)
        return madx_template % replace_dict
