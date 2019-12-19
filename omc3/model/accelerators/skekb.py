"""
Super KEK-B
-------------------
"""
from model.accelerators.accelerator import Accelerator, AcceleratorDefinitionError
from utils import logging_tools
from generic_parser import EntryPointParameters

LOGGER = logging_tools.get_logger(__name__)


class SKekB(Accelerator):
    """
    KEK's SuperKEKB accelerator.
    """
    NAME = "skekb"

    @property
    def ring(self):
        if self._ring is None:
            raise AcceleratorDefinitionError("The accelerator definition is incomplete, ring "
                                             "has to be specified (--ring option missing?).")
        return self._ring

    @ring.setter
    def ring(self, value):
        if value not in ("ler", "her"):
            raise AcceleratorDefinitionError("Ring parameter has to be one of ('ler', 'her')")
        self._ring = value

    @staticmethod
    def get_class_parameters():
        params = EntryPointParameters()
        params.add_parameter(name="ring", type=str, choices=("ler", "her"), help="HER or LER ring.")
        return params

    @classmethod
    def _get_class(cls, opt):
        """ Actual get_class function """
        new_class = cls
        if opt.ring is not None:
            new_class.ring = opt.ring
        if new_class.ring == 'her':
            new_class.beam_direction = -1
        if new_class.ring == 'ler':
            new_class.beam_direction = 1
        return new_class

    def verify_object(self):
        if self.model_dir is None:  # is the class is used to create full response?
            raise AcceleratorDefinitionError("SuperKEKB doesn't have a model creation, "
                                             "calling it this way is most probably wrong.")
