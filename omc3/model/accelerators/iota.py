"""
IOTA
-------------------
"""
import os
from model.accelerators.accelerator import Accelerator, AccElementTypes
from generic_parser import EntryPointParameters
import logging

LOGGER = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


class Iota(Accelerator):
    NAME = "iota"
    RE_DICT = {AccElementTypes.BPMS: r"IBPM*",
               AccElementTypes.MAGNETS: r"Q*",
               AccElementTypes.ARC_BPMS: r"IBPM*"}
    BPM_INITIAL = 'I'

    @staticmethod
    def get_class_parameters():
        params = EntryPointParameters()
        params.add_parameter(name="particle", type=str, choices=('p', 'e'), help="Particle type.")
        return params

    @classmethod
    def verify_object(self):
        pass  # TODO
