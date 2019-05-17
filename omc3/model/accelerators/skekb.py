import os

from model.accelerators.accelerator import (Accelerator, AcceleratorDefinitionError,
                                            AccExcitationMode)
from utils import logging_tools
from parser.entrypoint import EntryPoint, EntryPointParameters, split_arguments

CURRENT_DIR = os.path.dirname(__file__)
LOGGER = logging_tools.get_logger(__name__)


class SKekB(Accelerator):
    """
    KEK's SuperKEKB accelerator.
    Beam direction inverted for now for using with HER.
    """
    NAME = "skekb"
    MACROS_NAME = "skekb"

    @classmethod
    def init_and_get_unknowns(cls, args=None):
        """ Initializes but also returns unknowns.

         For the desired philosophy of returning parameters all the time,
         try to avoid this function, e.g. parse outside parameters first.
         """
        opt, rest_args = split_arguments(args, cls.get_instance_parameters())
        return cls(opt), rest_args

    @classmethod
    def get_class(cls, *args, **kwargs):
        """ Returns subclass .

        """
        parser = EntryPoint(cls.get_class_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        return cls._get_class(opt)

    @classmethod
    def get_class_and_unknown(cls, *args, **kwargs):
        """ Returns subclass and unkown args .

        For the desired philosophy of returning parameters all the time,
        try to avoid this function, e.g. parse outside parameters first.
        """
        parser = EntryPoint(cls.get_class_parameters(), strict=False)
        opt, unknown_opt = parser.parse(*args, **kwargs)
        return cls._get_class(opt), unknown_opt

    @classmethod
    def _get_class(cls, opt):
        """ Actual get_class function """
        new_class = cls
        return new_class

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