import os
import datetime as dt
from model.accelerators.accelerator import Accelerator, Element
import tfs
from parser.entrypoint import EntryPoint, EntryPointParameters, split_arguments
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

    @classmethod
    def get_name(cls, args=None):
        return cls.NAME
    
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
        """ Returns Ps class.

        Keyword Args:

        Returns:
            Ps class.
        """
        parser = EntryPoint(cls.get_class_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        return cls._get_class(opt)

    @classmethod
    def get_class_and_unknown(cls, *args, **kwargs):
        """ Returns Ps subclass and unkown args .

        For the desired philosophy of returning parameters all the time,
        try to avoid this function, e.g. parse outside parameters first.
        """
        parser = EntryPoint(cls.get_class_parameters(), strict=False)
        opt, unknown_opt = parser.parse(*args, **kwargs)
        return cls._get_class(opt), unknown_opt

    @classmethod
    def _get_class(cls, opt):
        new_class = cls
        return new_class

    # Public Methods ##########################################################

    @classmethod
    def get_segment(cls, label, first_elem, last_elem, optics_file, twiss_file):
        # this creates a new class called PsSegment
        segment_cls = type(cls.__name__ + "Segment",
                          (_PsSegmentMixin, cls),
                          {})
        segment_inst = segment_cls()
        
        bpms_file = os.path.join(PS_DIR, str(CURRENT_YEAR), "sequence/bpms.tfs")
        bpms_file_data = tfs.read(bpms_file).set_index("NAME")
        first_elem_s = bpms_file_data.loc[first_elem, "S"]
        last_elem_s = bpms_file_data.loc[last_elem, "S"]
        segment_inst.label = label
        segment_inst.start = Element(first_elem, first_elem_s)
        segment_inst.end = Element(last_elem, last_elem_s)
        segment_inst.optics_file = optics_file
        segment_inst.fullresponse = None
        
        LOGGER.debug('twiss_file is <%s>', twiss_file)
        tw = tfs.read(twiss_file)
        
        LOGGER.debug('twiss_file has tunes %f %f ', tw.Q1, tw.Q2)

        segment_inst.nat_tune_x = tw.Q1
        segment_inst.nat_tune_y = tw.Q2
        segment_inst.energy = tw.ENERGY
        segment_inst.kind = '' # '' means beta from phase, can be 'betaamp', in the future 'betakmod'
        
        return segment_inst    

    def verify_object(self):
        pass

    @classmethod
    def get_nominal_tmpl(cls):
        return os.path.join(PS_DIR, "nominal.madx")

    @classmethod
    def get_ps_dir(cls):
        return os.path.join(PS_DIR, str(cls.YEAR))

    @classmethod
    def get_iteration_tmpl(cls):
        return cls.get_file("template.iterate.madx")

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
