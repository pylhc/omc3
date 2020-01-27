"""
PS BOOSTER
-------------------
"""
import logging
import os

from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import (Accelerator,
                                                 AcceleratorDefinitionError)

LOGGER = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


class Psbooster(Accelerator):
    """ Parent Class for Psbooster-Types.    """
    NAME = "psbooster"

    @staticmethod
    def get_parameters():
        params = super(Psbooster, Psbooster).get_parameters()
        params.add_parameter(name="ring", type=int, choices=(1, 2, 3, 4), help="Ring to use.")
        return params

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
        self.ring = opt.ring

    @property
    def ring(self):
        if self._ring is None:
            raise AcceleratorDefinitionError("The accelerator definition is incomplete, ring "
                                             "has to be specified (--ring option missing?).")
        return self._ring

    @ring.setter
    def ring(self, value):
        if value not in (1, 2, 3, 4):
            raise AcceleratorDefinitionError("Ring parameter has to be one of (1, 2, 3, 4)")
        self._ring = value

    def verify_object(self):
        _ = self.ring

    @classmethod
    def get_dir(cls):
        return os.path.join(CURRENT_DIR, cls.NAME)

    @classmethod
    def get_file(cls, filename):
        return os.path.join(CURRENT_DIR, cls.NAME, filename)


class _PsboosterSegmentMixin(object):

    def __init__(self):
        self._start = None
        self._end = None
