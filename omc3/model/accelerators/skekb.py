"""
Super KEK-B
-------------------
"""
from model.accelerators.accelerator import Accelerator, AcceleratorDefinitionError
from utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


class SKekB(Accelerator):
    """
    KEK's SuperKEKB accelerator.
    Beam direction inverted for now for using with HER.
    """
    NAME = "skekb"
    MACROS_NAME = "skekb"

    def verify_object(self):  # TODO: Maybe more checks?
        if self.model_dir is None:  # is the class is used to create full response?
            raise AcceleratorDefinitionError("SuperKEKB doesn't have a model creation, calling it this "
                                             "way is most probably wrong.")

    def get_beam_direction(self):
        return -1