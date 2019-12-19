"""
PS
-------------------
"""
import os
from model.accelerators.accelerator import Accelerator
import logging

LOGGER = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


class Ps(Accelerator):
    """ Parent Class for Ps-Types. """
    NAME = "ps"
    YEAR = 2018

    # Public Methods ##########################################################

    def verify_object(self):
        pass

    @classmethod
    def get_dir(cls):
        return os.path.join(CURRENT_DIR, cls.NAME, str(cls.YEAR))

    @classmethod
    def get_file(cls, filename):
        return os.path.join(CURRENT_DIR, cls.NAME, filename)


class _PsSegmentMixin(object):

    def __init__(self):
        self._start = None
        self._end = None
        self.energy = None
