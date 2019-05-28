"""
PS BOOSTER
-------------------
"""
import os
import re
from model.accelerators.accelerator import Accelerator
from parser.entrypoint import EntryPointParameters
import logging

LOGGER = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)
PSB_DIR = os.path.join(CURRENT_DIR, "psbooster")


class Psbooster(Accelerator):
    """ Parent Class for Psbooster-Types.    """
    NAME = "psbooster"

    @staticmethod
    def get_class_parameters():
        params = EntryPointParameters()
        params.add_parameter(flags=["--ring"], help="Ring to use.", name="ring", type=int, choices=[1, 2, 3, 4])
        return params

    # Entry-Point Wrappers #####################################################

    @classmethod
    def _get_class(cls, opt):
        new_class = cls
        if opt.ring is not None:
            new_class = type(
                new_class.__name__ + "Ring{}".format(opt.ring),
                (new_class,),
                {"get_ring": classmethod(lambda cls: opt.ring)}
            )
        else:
            print("No ring info in options")
        return new_class

    # Public Methods ##########################################################
    def verify_object(self):
        pass

    @classmethod
    def get_segment_tmpl(cls):
        return cls.get_file("segment.madx")

    @classmethod
    def get_corrtest_tmpl(cls):
        return cls.get_file("correction_test.madx")

    @classmethod
    def get_psb_dir(cls):
        return PSB_DIR

    @classmethod
    def get_file(cls, filename):
        return os.path.join(CURRENT_DIR, "psbooster", filename)

    def get_beam_direction(self):
        return 1


class _PsboosterSegmentMixin(object):

    def __init__(self):
        self._start = None
        self._end = None

    # Private Methods ##########################################################


def _get_file_for_ring(ring):
    return os.path.join(PSB_DIR, f"twiss_ring{ring}.dat")


def _get_ring_from_seqname(seq):
    if re.match("^PSB[1-4]$", seq.upper()):
        return int(seq[3])
    LOGGER.error("Sequence name is none of the expected ones (PSB1,PSB2,PSB3,PSB4)")
    return None
