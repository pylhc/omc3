"""
IOTA
-------------------
"""
import os
import re
from model.accelerators.accelerator import Accelerator
from generic_parser.entrypoint import EntryPointParameters
import logging

LOGGER = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


def get_iota_modes():
    return {
        "iota_runI": IotaRunI
    }


class Iota(Accelerator):
    NAME = "iota"
    RE_DICT = {"bpm": r"IBPM", "magnet": r"Q"}

    @staticmethod
    def get_class_parameters():
        params = EntryPointParameters()
        params.add_parameter(flags=["--particle"],
                             help="Particle type.",
                             name="particle",
                             type=str,
                             choices=['p', 'e'])
        params.add_parameter(flags=["--run"],
                             help=("Specify IOTA run. Should be one of: " + str(get_iota_modes().keys())),
                             name="iota_run",
                             type=str,
                             choices=list(get_iota_modes().keys()))
        return params

    @staticmethod
    def get_iota_dir():
        return os.path.join(CURRENT_DIR, "iota")


# Specific accelerator definitions ###########################################


class IotaRunI(Iota):
    YEAR = "2019"
