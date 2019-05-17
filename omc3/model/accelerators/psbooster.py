import os
import re
from model.accelerators.accelerator import Accelerator, Element
from parser.entrypoint import EntryPoint, EntryPointParameters, split_arguments
import tfs
import logging

LOGGER = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)
PSB_DIR = os.path.join(CURRENT_DIR, "psbooster")


class Psbooster(Accelerator):
    """ Parent Class for Psbooster-Types.

    Keyword Args:
        Required
        ring (int): Ring number.
                            **Flags**: ['--ring']
    """
    NAME = "psbooster"

    @staticmethod
    def get_class_parameters():
        params = EntryPointParameters()
        params.add_parameter(flags=["--ring"], help="Ring to use.", name="ring", type=int, choices=[1, 2, 3, 4])
        return params

    # Entry-Point Wrappers #####################################################

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
        """ Returns Psbooster class.

        Keyword Args:
            Optional
            ring (int): Ring to use.
                        **Flags**: ['--ring']
                        **Choices**: [1, 2, 3, 4]

        Returns:
            Psbooster class.
        """
        parser = EntryPoint(cls.get_class_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        return cls._get_class(opt)

    @classmethod
    def get_class_and_unknown(cls, *args, **kwargs):
        """ Returns Psbooster subclass and unkown args .

        For the desired philosophy of returning parameters all the time,
        try to avoid this function, e.g. parse outside parameters first.
        """
        parser = EntryPoint(cls.get_class_parameters(), strict=False)
        opt, unknown_opt = parser.parse(*args, **kwargs)
        return cls._get_class(opt), unknown_opt

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
    @classmethod
    def get_segment(cls, label, first_elem, last_elem, optics_file, twiss_file):
        segment_cls = type(cls.__name__ + "Segment",
                          (_PsboosterSegmentMixin, cls),
                          {})
        LOGGER.debug('twiss_file is <%s>',twiss_file)
        tw = tfs.read(twiss_file)
        
        LOGGER.debug('twiss_file has tunes %f %f ',tw.Q1,tw.Q2)
        ring = _get_ring_from_seqname(tw.SEQUENCE)

        segment_inst = segment_cls()

        bpms_file = _get_file_for_ring(ring)
        bpms_file_data = tfs.read(bpms_file).set_index("NAME")
        first_elem_s = bpms_file_data.loc[first_elem, "S"]
        last_elem_s = bpms_file_data.loc[last_elem, "S"]
        segment_inst.label = label
        segment_inst.start = Element(first_elem, first_elem_s)
        segment_inst.end = Element(last_elem, last_elem_s)
        segment_inst.optics_file = optics_file
        segment_inst.fullresponse = None

        segment_inst.nat_tune_x = tw.Q1
        segment_inst.nat_tune_y = tw.Q2
        segment_inst.energy = tw.ENERGY
        segment_inst.sequence = tw.SEQUENCE
        segment_inst.ring = ring
        segment_inst.kind = '' # '' means beta from phase, can be 'betaamp', in the future 'betakmod'
        
        return segment_inst    

    def verify_object(self):
        pass

    @classmethod
    def get_nominal_tmpl(cls):
        return os.path.join(PSB_DIR, "nominal.madx")

    @classmethod
    def get_segment_tmpl(cls):
        return cls.get_file("segment.madx")

    @classmethod
    def get_iteration_tmpl(cls):
        return cls.get_file("template.iterate.madx")

    @classmethod
    def get_corrtest_tmpl(cls):
        return cls.get_file("correction_test.madx")

    @classmethod
    def get_psb_dir(cls):
        return PSB_DIR

    @classmethod
    def get_file(cls, filename):
        return os.path.join(CURRENT_DIR, "psbooster", filename)


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
