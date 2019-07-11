"""
PS
-------------------
"""
import os
import datetime as dt
from model.accelerators.accelerator import Accelerator
import logging

LOGGER = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)
CURRENT_YEAR = dt.datetime.now().year
PS_DIR = os.path.join(CURRENT_DIR, "ps")


class Ps(Accelerator):
    """ Parent Class for Ps-Types. """
    NAME = "ps"
    MACROS_NAME = "ps"
    YEAR = 2018

    # Public Methods ##########################################################

    def verify_object(self):
        pass

    @classmethod
    def get_ps_dir(cls):
        return os.path.join(PS_DIR, str(cls.YEAR))

    @classmethod
    def get_segment_tmpl(cls):
        return cls.get_file("segment.madx")
    
    @classmethod
    def get_file(cls, filename):
        return os.path.join(CURRENT_DIR, "ps", filename)
    
    # Private Methods ##########################################################


class _PsSegmentMixin(object):

    def __init__(self):
        self._start = None
        self._end = None
        self.energy = None
