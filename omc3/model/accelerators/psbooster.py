"""
PS BOOSTER
-------------------
"""
import os
import re
from model.accelerators.accelerator import Accelerator, AcceleratorDefinitionError
from generic_parser import EntryPoint
import logging

LOGGER = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


class Psbooster(Accelerator):
    """ Parent Class for Psbooster-Types.    """
    NAME = "psbooster"

    def get_parameters(self):
        params = super().get_parameters()
        params.add_parameter(name="ring", type=int, choices=(1, 2, 3, 4), help="Ring to use.")
        return params

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
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


def _get_ring_from_seqname(seq):
    if re.match("^PSB[1-4]$", seq.upper()):
        return int(seq[3])
    LOGGER.error("Sequence name is none of the expected ones (PSB1,PSB2,PSB3,PSB4)")
    return None


class _PsboosterSegmentMixin(object):

    def __init__(self):
        self._start = None
        self._end = None



