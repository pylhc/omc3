import argparse
import os

from model.accelerators.accelerator import (Accelerator, AcceleratorDefinitionError,
                                            AccExcitationMode)
import tfs
from utils import logging_tools
from utils.entrypoint import EntryPoint, EntryPointParameters, split_arguments

CURRENT_DIR = os.path.dirname(__file__)
LOGGER = logging_tools.get_logger(__name__)


class SKekB(Accelerator):
    """
    KEK's SuperKEKB accelerator.
    Beam direction inverted for now for using with HER.

    Minimal working example. Needed accelerator functions are still to be determined.
    """
    NAME = "skekb"
    MACROS_NAME = "skekb"

    @staticmethod
    def get_class_parameters():
        params = EntryPointParameters()
        return params

    @staticmethod
    def get_instance_parameters():
        params = EntryPointParameters()
        params.add_parameter(
            flags=["--model_dir", "-m"],
            help="Path to model directory (loads tunes and excitation from model!).",
            name="model_dir",
            type=str,
        )
        params.add_parameter(
            flags=["--nattunex"],
            help="Natural tune X without integer part.",
            name="nat_tune_x",
            type=float,
        )
        params.add_parameter(
            flags=["--nattuney"],
            help="Natural tune Y without integer part.",
            name="nat_tune_y",
            type=float,
        )
        params.add_parameter(
            flags=["--dpp"],
            help="Delta p/p to use.",
            name="dpp",
            default=0.0,
            type=float,
        )
        return params

    # Entry-Point Wrappers #####################################################

    def __init__(self, *args, **kwargs):
        # for reasons of import-order and class creation, decoration was not possible
        parser = EntryPoint(self.get_instance_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)

        if opt.model_dir:
            self.init_from_model_dir(opt.model_dir)
            self.energy = None
            if opt.nat_tune_x is not None:
                raise AcceleratorDefinitionError("Argument 'nat_tune_x' not allowed when loading from model directory.")
            if opt.nat_tune_y is not None:
                raise AcceleratorDefinitionError("Argument 'nat_tune_y' not allowed when loading from model directory.")
        else:
            if opt.nat_tune_x is None:
                raise AcceleratorDefinitionError("Argument 'nat_tune_x' is required.")
            if opt.nat_tune_y is None:
                raise AcceleratorDefinitionError("Argument 'nat_tune_y' is required.")

            self.nat_tune_x = opt.nat_tune_x
            self.nat_tune_y = opt.nat_tune_y

            # optional with default
            self.dpp = opt.dpp
            self.fullresponse = opt.fullresponse

            # optional no default
            self.energy = opt.get("energy", None)
            self.xing = opt.get("xing", None)
            self.optics_file = opt.get("optics", None)

            # for GetLLM
            self.model_dir = None
            self._model = None
            self._model_best_knowledge = None
            self._elements = None
            self._elements_centre = None
            self._errordefspath = None

        self.verify_object()

    def init_from_model_dir(self, model_dir):
        LOGGER.debug("Creating accelerator instance from model dir")
        self.model_dir = model_dir

        LOGGER.debug("  model path = " + os.path.join(model_dir, "twiss.dat"))
        self._model = tfs.read(os.path.join(model_dir, "twiss.dat"), index="NAME")
        self.nat_tune_x = float(self._model.headers["Q1"])
        self.nat_tune_y = float(self._model.headers["Q2"])


        # Elements #####################################
        elements_path = os.path.join(model_dir, "twiss_elements.dat")
        if os.path.isfile(elements_path):
            self._elements = tfs.read(elements_path, index="NAME")
        else:
            raise AcceleratorDefinitionError("Elements twiss not found")

        center_path = os.path.join(model_dir, "twiss_elements_centre.dat")
        if os.path.isfile(center_path):
            self._elements_centre = tfs.read(center_path, index="NAME")
        else:
            self._elements_centre = self._elements

        # Error Def #####################################
        self._errordefspath = None
        errordefspath = os.path.join(self.model_dir, "errordefs")
        if os.path.exists(errordefspath):
            self._errordefspath = errordefspath
        else:  # until we have a proper file name convention
            errordefspath = os.path.join(self.model_dir, "error_deffs.txt")
            if os.path.exists(errordefspath):
                self._errordefspath = errordefspath

    
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
 
    @classmethod
    def _get_arg_parser(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--nattunex",
            help="Natural tune X without integer part.",
            required=True,
            dest="nat_tune_x",
            type=float,
        )
        parser.add_argument(
            "--nattuney",
            help="Natural tune Y without integer part.",
            required=True,
            dest="nat_tune_y",
            type=float,
        )
       
        parser.add_argument(
            "--dpp",
            help="Delta p/p to use.",
            dest="dpp",
            default=0.0,
            type=float,
        )
        parser.add_argument(
            "--energy",
            help="Energy in Tev.",
            dest="energy",
            type=float,
        )
        parser.add_argument(
            "--optics",
            help="Path to the optics file to use (modifiers file).",
            dest="optics",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--fullresponse",
            help=("If present, fullresponse template will" +
                  "be filled and put in the output directory."),
            dest="fullresponse",
            action="store_true",
        )
        return parser

    def verify_object(self):  # TODO: Maybe more checks?
        if self.model_dir is None:  # is the class is used to create full response?
            raise AcceleratorDefinitionError("SuperKEKB doesn't have a model creation, calling it this "
                                             "way is most probably wrong.")


    def get_arc_bpms_mask(cls, list_of_elements):
        return [True] * len(list_of_elements)

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

    def get_s_first_BPM(self):
        return 0

    def get_k_first_BPM(self, list_of_bpms):
        return len(list_of_bpms)
        
    def get_model_tfs(self):
        return self._model
        
    @classmethod
    def get_element_types_mask(cls, list_of_elements, types):
        """
        Return boolean mask for elements in list_of_elements that belong
        to any of the specified types.
        Needs to handle: "bpm", "magnet", "arc_bpm"

        Args:
            list_of_elements: List of elements
            types: Kinds of elements to look for

        Returns:
            Boolean array of elements of specified kinds.

        """

        return list_of_elements == list_of_elements

    def get_elements_tfs(self):
        return self._elements

    def get_elements_centre_tfs(self):
        return self._elements_centre

    def get_amp_bpms(self, bpms):
        return bpms
