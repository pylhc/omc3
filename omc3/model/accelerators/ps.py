"""
PS
-------------------
"""
import logging
import os

from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import Accelerator

LOGGER = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


class Ps(Accelerator):
    """ Parent Class for Ps-Types. """
    NAME = "ps"
    YEAR = 2018

    # Public Methods ##########################################################

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)

    def verify_object(self):
        pass

    @classmethod
    def get_dir(cls):
        return os.path.join(CURRENT_DIR, cls.NAME, str(cls.YEAR))

    @classmethod
    def get_file(cls, filename):
        return os.path.join(CURRENT_DIR, cls.NAME, filename)

    def get_exciter_bpm(self, plane, bpms):
        if not self.excitation:
            return None
        plane_to_hv = dict(X="h", Y="v")
        bpms_to_find = ["PR.BPM00", "PR.BPM03"]
        found_bpms = [bpm for bpm in bpms_to_find if bpm in bpms]
        if not len(found_bpms):
            raise KeyError
        return (list(bpms).index(found_bpms[0]), found_bpms[0]), f"{plane_to_hv[plane]}acmap"


class _PsSegmentMixin(object):

    def __init__(self):
        self._start = None
        self._end = None
        self.energy = None
