"""
Accelerator
-------------------

Contains parent accelerator class and other support classes
"""

from parser.entrypoint import EntryPoint, EntryPointParameters, split_arguments
import os
import pandas as pd
import tfs
from utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


class AccExcitationMode(object):
    # it is very important that FREE = 0
    FREE, ACD, ADT = range(3)


class Accelerator(object):
    """
    Abstract class to serve as an interface to implement the rest of the accelerators.
    """
    RE_DICT = {"bpm": r".*", "magnet": r".*", "arc_bpm": r".*"}
    MODIFIERS_MADX = "modifiers.madx"
    ELEMENTS_CENTRE_DAT = "twiss_elements_centre.dat"
    TWISS_BEST_KNOWLEDGE_DAT = "twiss_best_knowledge.dat"
    TWISS_ADT_DAT = "twiss_adt.dat"
    TWISS_AC_DAT = "twiss_ac.dat"
    TWISS_ELEMENTS_DAT = "twiss_elements.dat"
    TWISS_DAT = "twiss.dat"
    ERROR_DEFFS_TXT = "error_deffs.txt"

    @staticmethod
    def get_instance_parameters():
        params = EntryPointParameters()
        params.add_parameter(flags=["--model_dir", "-m"],
                             help="Path to model directory (loads tunes and excitation from model!).",
                             name="model_dir", type=str, )
        params.add_parameter(flags=["--nattunex"], help="Natural tune X without integer part.",
                             name="nat_tune_x", type=float, )
        params.add_parameter(flags=["--nattuney"], help="Natural tune Y without integer part.",
                             name="nat_tune_y", type=float, )
        params.add_parameter(flags=["--acd"], help="Activate excitation with ACD.", name="acd",
                             action="store_true")
        params.add_parameter(flags=["--adt"], help="Activate excitation with ADT.", name="adt",
                             action="store_true", )
        params.add_parameter(flags=["--drvtunex"], help="Driven tune X without integer part.",
                             name="drv_tune_x", type=float, )
        params.add_parameter(flags=["--drvtuney"], help="Driven tune Y without integer part.",
                             name="drv_tune_y", type=float, )
        params.add_parameter(flags=["--dpp"], help="Delta p/p to use.", name="dpp", default=0.0,
                             type=float, )
        params.add_parameter(flags=["--energy"], help="Energy in Tev.", name="energy", type=float, )
        params.add_parameter(flags=["--optics"],
                             help="Path to the optics file to use (modifiers file).", name="optics",
                             type=str, )
        params.add_parameter(flags=["--fullresponse"], help=(
            "If True, fullresponse template will be filled and put in the output directory."),
                             name="fullresponse", action="store_true", )
        params.add_parameter(flags=["--xing"],
                             help="If True, x-ing  angles will be applied to model", name="xing",
                             action="store_true", )
        params.add_parameter(flags=["--year_opt"],
                             help="Year of the optics. Default is the current year.",
                             name="year_opt", type=int, )

        return params

    def __init__(self, *args, **kwargs):
        # for reasons of import-order and class creation, decoration was not possible

        parser = EntryPoint(self.get_instance_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)

        if opt.model_dir:
            self.init_from_model_dir(opt.model_dir)
            self.energy = None
            self.dpp = 0.0
            self.xing = None
            if ((opt.nat_tune_x is not None) or (opt.nat_tune_y is not None) or
                    (opt.drv_tune_x is not None) or (opt.drv_tune_y is not None)):
                raise AcceleratorDefinitionError(
                    "None of Arguments 'nat_tune_x', 'nat_tune_y', 'drv_tune_x' and 'drv_tune_y' "
                    "are allowed when loading from model directory.")
        else:
            self.init_from_options(opt)

        self.verify_object()

    def init_from_options(self, opt):
        if opt.nat_tune_x is None:
            raise AcceleratorDefinitionError("Argument 'nat_tune_x' is required.")
        if opt.nat_tune_y is None:
            raise AcceleratorDefinitionError("Argument 'nat_tune_y' is required.")
        self.nat_tune_x = opt.nat_tune_x
        self.nat_tune_y = opt.nat_tune_y
        self.drv_tune_x = None
        self.drv_tune_y = None
        self._excitation = AccExcitationMode.FREE
        if opt.acd or opt.adt:
            if opt.acd and opt.adt:
                raise AcceleratorDefinitionError("Select only one excitation type.")

            if opt.drv_tune_x is None:
                raise AcceleratorDefinitionError("Argument 'drv_tune_x' is required.")
            if opt.drv_tune_y is None:
                raise AcceleratorDefinitionError("Argument 'drv_tune_x' is required.")
            self.drv_tune_x = opt.drv_tune_x
            self.drv_tune_y = opt.drv_tune_y

            if opt.acd:
                self._excitation = AccExcitationMode.ACD
            elif opt.adt:
                self._excitation = AccExcitationMode.ADT
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
        self._model_driven = None
        self._model_best_knowledge = None
        self._elements = None
        self._elements_centre = None
        self._errordefspath = None

    def init_from_model_dir(self, model_dir):
        LOGGER.debug("Creating accelerator instance from model dir")
        self.model_dir = model_dir

        LOGGER.debug("  model path = " + os.path.join(model_dir, self.TWISS_DAT))
        try:
            self._model = tfs.read(os.path.join(model_dir, self.TWISS_DAT), index="NAME")
        except IOError:
            self._model = tfs.read(os.path.join(model_dir, self.TWISS_ELEMENTS_DAT), index="NAME")
            bpm_index = [idx for idx in self._model.index.values if idx.startswith("B")]
            self._model = self._model.loc[bpm_index, :]
        self.nat_tune_x = float(self._model.headers["Q1"])
        self.nat_tune_y = float(self._model.headers["Q2"])

        # Excitations #####################################
        self._model_driven = None
        self.drv_tune_x = None
        self.drv_tune_y = None
        self._excitation = AccExcitationMode.FREE

        ac_filename = os.path.join(model_dir, self.TWISS_AC_DAT)
        adt_filename = os.path.join(model_dir, self.TWISS_ADT_DAT)

        if os.path.isfile(ac_filename):
            self._model_driven = tfs.read(ac_filename, index="NAME")
            self._excitation = AccExcitationMode.ACD

        if os.path.isfile(adt_filename):
            if self._excitation == AccExcitationMode.ACD:
                raise AcceleratorDefinitionError("ADT as well as ACD models provided."
                                                 "Please choose only one.")

            self._model_driven = tfs.read(adt_filename, index="NAME")
            self._excitation = AccExcitationMode.ADT

        if not self._excitation == AccExcitationMode.FREE:
            self.drv_tune_x = float(self.get_driven_tfs().headers["Q1"])
            self.drv_tune_y = float(self.get_driven_tfs().headers["Q2"])

        # Best Knowledge #####################################
        self._model_best_knowledge = None
        best_knowledge_path = os.path.join(model_dir, self.TWISS_BEST_KNOWLEDGE_DAT)
        if os.path.isfile(best_knowledge_path):
            self._model_best_knowledge = tfs.read(best_knowledge_path, index="NAME")

        # Elements #####################################
        elements_path = os.path.join(model_dir, self.TWISS_ELEMENTS_DAT)
        if os.path.isfile(elements_path):
            self._elements = tfs.read(elements_path, index="NAME")
        else:
            raise AcceleratorDefinitionError("Elements twiss not found")

        center_path = os.path.join(model_dir, self.ELEMENTS_CENTRE_DAT)
        if os.path.isfile(center_path):
            self._elements_centre = tfs.read(center_path, index="NAME")
        else:
            self._elements_centre = self._elements

        # Optics File #########################################
        self.optics_file = None
        opticsfilepath = os.path.join(self.model_dir, self.MODIFIERS_MADX)
        if os.path.exists(opticsfilepath):
            self.optics_file = opticsfilepath

        # Error Def #####################################
        self._errordefspath = None
        errordefspath = os.path.join(self.model_dir, self.ERROR_DEFFS_TXT)
        if os.path.exists(errordefspath):
            self._errordefspath = errordefspath

    # Class methods ###########################################

    @staticmethod
    def get_class_parameters():
        """
        This method should return the parameter list of arguments needed to create the class.
        """
        params = EntryPointParameters()
        return params

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
        """
        This method should return the accelerator class defined in the arguments.
        """
        parser = EntryPoint(cls.get_class_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        return cls._get_class(opt)

    @classmethod
    def get_class_and_unknown(cls, *args, **kwargs):
        """ Returns subclass and unknown args.
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
        unknown_elements = [ty for ty in types if ty not in cls.RE_DICT]
        if len(unknown_elements):
            raise TypeError("Unknown element(s): '{:s}'".format(str(unknown_elements)))
        series = pd.Series(list_of_elements)
        mask = series.str.match(cls.RE_DICT[types[0]], case=False)
        for ty in types[1:]:
            mask = mask | series.str.match(cls.RE_DICT[ty], case=False)
        return mask.values

    @classmethod
    def get_variables(cls, frm=None, to=None, classes=None):
        """
        Gets the variables with elements in the given range and the given
        classes. None means everything.
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    @classmethod
    def get_correctors_variables(cls, frm=None, to=None, classes=None):
        """
        Returns the set of corrector variables between frm and to, with classes
        in classes. None means select all.
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    # Instance methods ########################################

    def verify_object(self):
        """
        Verifies that this instance of an accelerator is properly
        instantiated.
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    # For GetLLM #############################################################
    
    def get_exciter_bpm(self, plane, distance):
        """
        Returns the BPM next to the exciter.
        The accelerator instance knows already which excitation method is used.
        distance: 1=nearest bpm 2=next to nearest bpm
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")
        
    def get_important_phase_advances(self):
        return []
    
    def get_model_tfs(self):
        return self._model

    def get_driven_tfs(self):
        if self._model_driven is None:
            raise AttributeError("No driven model given in this accelerator instance.")
        return self._model_driven

    def get_best_knowledge_model_tfs(self):
        if self._model_best_knowledge is None:
            raise AttributeError("No best knowledge model given in this accelerator instance.")
        return self._model_best_knowledge

    def get_elements_tfs(self):
        return self._elements

    def get_elements_centre_tfs(self):
        return self._elements_centre

    @property
    def excitation(self):
        return self._excitation

    @excitation.setter
    def excitation(self, excitation_mode):
        if excitation_mode not in (AccExcitationMode.FREE,
                                   AccExcitationMode.ACD,
                                   AccExcitationMode.ADT):
            raise ValueError("Wrong excitation mode.")
        self._excitation = excitation_mode

    def get_errordefspath(self):
        """Returns the path to the uncertainty definitions file (formerly called error definitions file.
        """
        if self._errordefspath is None:
            raise AttributeError("No error definitions file given in this accelerator instance.")
        return self._errordefspath

    def set_errordefspath(self, path):
        self._errordefspath = path


    # Templates ##############################################################
    @classmethod
    def get_nominal_tmpl(cls):
        """ Returns template for nominal model (Model Creator) """
        return cls.get_file("nominal.madx")

    @classmethod
    def get_file(cls, filename):
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    @classmethod
    def get_iteration_tmpl(cls):
        """
        Returns template to create fullresponse.
        TODO: only in _prepare_fullresponse in creator! Needs to be replaced by get_basic_seq
        """
        return cls.get_file("template.iterate.madx")

    # Jobs ###################################################################

    def get_update_correction_job(self, tiwss_out_path, corrections_file_path):
        """
        Returns job (string) to create an updated model from changeparameters input
        (used in iterative correction).
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    def get_basic_seq_job(self):
        """
        Returns job (string) to create the basic accelerator sequence.
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    def get_multi_dpp_job(self, dpp_list):
        """
        Returns job (string) for model with multiple dp/p values (in W-Analysis)
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    ##########################################################################


class Variable(object):
    """
    Generic corrector variable class that holds name, position (s) and
    physical elements it affectes. This variables should be logical variables
    that should have and effect in the model if modified.
    """
    def __init__(self, name, elements, classes):
        self.name = name
        self.elements = elements
        self.classes = classes


class Element(object):
    """
    Generic corrector element class that holds name and position (s)
    of the corrector. This element should represent a physical element of the
    accelerator.
    """
    def __init__(self, name, s):
        self.name = name
        self.s = s


class AcceleratorDefinitionError(Exception):
    """
    Raised when an accelerator instance is wrongly used, for
    example by calling a method that should have been overwritten.
    """
    pass


