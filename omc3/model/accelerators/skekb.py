"""
Super KEK-B
-------------------
"""
from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import (Accelerator,
                                                 AcceleratorDefinitionError)
from omc3.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)
RINGS = ("ler", "her")


class SKekB(Accelerator):
    """
    KEK's SuperKEKB accelerator.
    """
    NAME = "skekb"
    RINGS = ("ler", "her")

    @staticmethod
    def get_parameters():
        params = super(SKekB, SKekB).get_parameters()
        params.add_parameter(name="ring", type=str, choices=RINGS, required=True,
                             help="HER or LER ring.")
        return params

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
        self.ring = opt.ring
        ring_to_beam_direction = {"ler": 1, "her": -1}
        self.beam_direction = ring_to_beam_direction[self.ring]

    @property
    def ring(self):
        if self._ring is None:
            raise AcceleratorDefinitionError("The accelerator definition is incomplete, ring "
                                             "has to be specified (--ring option missing?).")
        return self._ring

    @ring.setter
    def ring(self, value):
        if value not in RINGS:
            raise AcceleratorDefinitionError("Ring parameter has to be one of ('ler', 'her')")
        self._ring = value

    def verify_object(self):
        if self.model_dir is None:  # is the class is used to create full response?
            raise AcceleratorDefinitionError("SuperKEKB doesn't have a model creation, "
                                             "calling it this way is most probably wrong.")
