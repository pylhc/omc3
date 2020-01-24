"""
IOTA
-------------------
"""
import logging
import os

from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import AccElementTypes, Accelerator

LOGGER = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


class Iota(Accelerator):
    NAME = "iota"
    RE_DICT = {AccElementTypes.BPMS: r"IBPM*",
               AccElementTypes.MAGNETS: r"Q*",
               AccElementTypes.ARC_BPMS: r"IBPM*"}
    BPM_INITIAL = 'I'

    @staticmethod
    def get_parameters():
        params = super(Iota, Iota).get_parameters()
        params.add_parameter(name="particle", type=str, choices=('p', 'e'), help="Particle type.")
        return params

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
        self.particle = opt.particle

    @classmethod
    def verify_object(self):
        pass  # TODO
