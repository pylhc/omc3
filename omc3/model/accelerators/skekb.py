"""
Super KEK-B
-------------------
"""
from model.accelerators.accelerator import Accelerator, AcceleratorDefinitionError, AccExcitationMode
from utils import logging_tools
from generic_parser.entrypoint import EntryPointParameters


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

    @staticmethod
    def get_class_parameters():
        params = EntryPointParameters()
        params.add_parameter(flags=["--ring"], help="HER or LER ring.", name="ring", choices=("ler", "her"), type=str)
        return params


    @classmethod
    def _get_class(cls, opt):
        """ Actual get_class function """
        new_class = cls
        new_class = cls._get_beamed_class(new_class, opt.ring)
        return new_class


    @classmethod
    def _get_beamed_class(cls, new_class, ring):
        ringSKEKB = _Her if ring == 'her' else _Ler
        beamed_class = type(new_class.__name__ + str(ring),
                            (new_class, ringSKEKB),
                            {})
        return beamed_class


class _Ler(object):
    @classmethod
    def get_beam_direction(cls):
        return 1


class _Her(object):
    @classmethod
    def get_beam_direction(cls):
        return -1