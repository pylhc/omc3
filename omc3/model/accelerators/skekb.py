from model.accelerators.accelerator import (Accelerator, AcceleratorDefinitionError,
                                            AccExcitationMode)
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

    def get_errordefspath(self):
        """Returns the path to the uncertainty definitions file (formerly called error definitions
        file.
        """
        if self._errordefspath is None:
            raise AttributeError("No error definitions file given in this accelerator instance.")
        return self._errordefspath
    
    @property
    def excitation(self):
        """Returns the excitation mode.
        SuperKEKB has two excitation modes:
            - feedback kicker
            - injections oscillation (horizontal), a trick can be used to get vertical but this
            seems to be problematic for machine protection.
        """
        return AccExcitationMode.FREE

    def set_errordefspath(self, path):
        self._errordefspath = path

    def get_beam_direction(self):
        return -1