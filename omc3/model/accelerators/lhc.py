import json
import os
from collections import OrderedDict

import pandas as pd

from model.accelerators.accelerator import Accelerator, AcceleratorDefinitionError, Element, AccExcitationMode
from utils import logging_tools
import tfs
from parser.entrypoint import EntryPoint, EntryPointParameters, split_arguments

LOGGER = logging_tools.get_logger(__name__)
CURRENT_DIR = os.path.dirname(__file__)
LHC_DIR = os.path.join(CURRENT_DIR, "lhc")


def get_lhc_modes():
    return {
        "lhc_runI": LhcRunI,
        "lhc_runII": LhcRunII2015,
        "lhc_runII_2016": LhcRunII2016,
        "lhc_runII_2016_ats": LhcRunII2016Ats,
        "lhc_runII_2017": LhcRunII2017,
        "lhc_runII_2018": LhcRunII2018,
        "hllhc10": HlLhc10,
        "hllhc12": HlLhc12,
        "hllhc13": HlLhc13,
    }


class Lhc(Accelerator):
    """ Parent Class for Lhc-Types.

    Keyword Args:
        Required
        nat_tune_x (float): Natural tune X without integer part.
                            **Flags**: ['--nattunex']
        nat_tune_y (float): Natural tune Y without integer part.
                            **Flags**: ['--nattuney']
        optics (str): Path to the optics file to use (modifiers file).
                      **Flags**: ['--optics']

        Optional
        acd (bool): Activate excitation with ACD.
                    **Flags**: ['--acd']
                    **Default**: ``False``
        adt (bool): Activate excitation with ADT.
                    **Flags**: ['--adt']
                    **Default**: ``False``
        dpp (float or list): Delta p/p to use.
                     **Flags**: ['--dpp']
                     **Default**: ``0.0``
        drv_tune_x (float): Driven tune X without integer part.
                            **Flags**: ['--drvtunex']
        drv_tune_y (float): Driven tune Y without integer part.
                            **Flags**: ['--drvtuney']
        energy (float): Energy in Tev.
                        **Flags**: ['--energy']
        fullresponse (bool): If True, fullresponse template will be filled
        and put in the output directory.
                             **Flags**: ['--fullresponse']
                             **Default**: ``False``
        xing (bool): If True, x-ing  angles will be applied to model
                     **Flags**: ['--xing']
                     **Default**: ``False``
    """
    NAME = "lhc"
    MACROS_NAME = "lhc"

    @staticmethod
    def get_class_parameters():
        params = EntryPointParameters()
        params.add_parameter(
            flags=["--lhcmode"],
            help=("LHC mode to use. Should be one of: " +
                  str(get_lhc_modes().keys())),
            name="lhc_mode",
            type=str,
            choices=list(get_lhc_modes().keys())
        )
        params.add_parameter(
            flags=["--beam"],
            help="Beam to use.",
            name="beam",
            type=int,
        )
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
            flags=["--acd"],
            help="Activate excitation with ACD.",
            name="acd",
            action="store_true"
        )
        params.add_parameter(
            flags=["--adt"],
            help="Activate excitation with ADT.",
            name="adt",
            action="store_true",
        )
        params.add_parameter(
            flags=["--drvtunex"],
            help="Driven tune X without integer part.",
            name="drv_tune_x",
            type=float,
        )
        params.add_parameter(
            flags=["--drvtuney"],
            help="Driven tune Y without integer part.",
            name="drv_tune_y",
            type=float,
        )
        params.add_parameter(
            flags=["--dpp"],
            help="Delta p/p to use.",
            name="dpp",
            default=0.0,
            type=float,
        )
        params.add_parameter(
            flags=["--energy"],
            help="Energy in Tev.",
            name="energy",
            type=float,
        )
        params.add_parameter(
            flags=["--optics"],
            help="Path to the optics file to use (modifiers file).",
            name="optics",
            type=str,
        )
        params.add_parameter(
            flags=["--fullresponse"],
            help=("If True, fullresponse template will "
                  "be filled and put in the output directory."),
            name="fullresponse",
            action="store_true",
        )
        params.add_parameter(
            flags=["--xing"],
            help="If True, x-ing  angles will be applied to model",
            name="xing",
            action="store_true",
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
            self.dpp = 0.0
            self.xing = None
            if opt.nat_tune_x is not None:
                raise AcceleratorDefinitionError(
                    "Argument 'nat_tune_x' not allowed when loading from model directory."
                )
            if opt.nat_tune_y is not None:
                raise AcceleratorDefinitionError(
                    "Argument 'nat_tune_y' not allowed when loading from model directory."
                )
            if opt.drv_tune_x is not None:
                raise AcceleratorDefinitionError(
                    "Argument 'drv_tune_x' not allowed when loading from model directory."
                )
            if opt.drv_tune_y is not None:
                raise AcceleratorDefinitionError(
                    "Argument 'drv_tune_y' not allowed when loading from model directory."
                )
        else:
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
                    raise AcceleratorDefinitionError(
                        "Select only one excitation type."
                    )

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

        self.verify_object()

    def init_from_model_dir(self, model_dir):
        LOGGER.debug("Creating accelerator instance from model dir")
        self.model_dir = model_dir

        LOGGER.debug("  model path = " + os.path.join(model_dir, "twiss.dat"))
        try:
            self._model = tfs.read(
                os.path.join(model_dir, "twiss.dat"), index="NAME")
        except IOError:
            self._model = tfs.read(
                os.path.join(model_dir, "twiss_elements.dat"), index="NAME")
            bpm_index = [idx for idx in self._model.index.values if idx.startswith("B")]
            self._model = self._model.loc[bpm_index, :]
        self.nat_tune_x = float(self._model.headers["Q1"])
        self.nat_tune_y = float(self._model.headers["Q2"])

        # Excitations #####################################
        self._model_driven = None
        self.drv_tune_x = None
        self.drv_tune_y = None
        self._excitation = AccExcitationMode.FREE

        ac_filename = os.path.join(model_dir, "twiss_ac.dat")
        adt_filename = os.path.join(model_dir, "twiss_adt.dat")

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
        best_knowledge_path = os.path.join(model_dir, "twiss_best_knowledge.dat")
        if os.path.isfile(best_knowledge_path):
            self._model_best_knowledge = tfs.read(best_knowledge_path, index="NAME")

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

        # Optics File #########################################
        self.optics_file = None
        opticsfilepath = os.path.join(self.model_dir, "modifiers.madx")
        if os.path.exists(opticsfilepath):
            self.optics_file = opticsfilepath

        # Error Def #####################################
        self._errordefspath = None
        errordefspath = os.path.join(self.model_dir, "error_deff.txt")
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
        """ Returns LHC subclass .

        Keyword Args:
            Optional
            beam (int): Beam to use.
                        **Flags**: ['--beam']
            lhc_mode (str): LHC mode to use.
                            **Flags**: ['--lhcmode']
                            **Choices**: ['lhc_runII_2016_ats', 'hllhc12', 'hllhc10', 'lhc_runI',
                            'lhc_runII', 'lhc_runII_2016', 'lhc_runII_2017']

        Returns:
            Lhc subclass.
        """
        parser = EntryPoint(cls.get_class_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        return cls._get_class(opt)

    @classmethod
    def get_class_and_unknown(cls, *args, **kwargs):
        """ Returns LHC subclass and unkown args .

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
        if opt.lhc_mode is not None:
            new_class = get_lhc_modes()[opt.lhc_mode]
        if opt.beam is not None:
            new_class = cls._get_beamed_class(new_class, opt.beam)
        return new_class

    # Public Methods ##########################################################

    @classmethod
    def get_segment(cls, label, first_elem, last_elem, optics_file, twiss_file):
        segment_cls = type(cls.__name__ + "Segment",
                           (_LhcSegmentMixin, cls),
                           {})
        segment_inst = segment_cls()
        beam = cls.get_beam()
        bpms_file_name = "beam1bpms.tfs" if beam == 1 else "beam2bpms.tfs"
        bpms_file = _get_file_for_year(cls.YEAR, bpms_file_name)
        bpms_file_data = tfs.read(bpms_file).set_index("NAME")
        first_elem_s = bpms_file_data.loc[first_elem, "S"]
        last_elem_s = bpms_file_data.loc[last_elem, "S"]
        segment_inst.label = label
        segment_inst.start = Element(first_elem, first_elem_s)
        segment_inst.end = Element(last_elem, last_elem_s)
        segment_inst.optics_file = optics_file
        segment_inst.xing = False
        segment_inst.fullresponse = False
        segment_inst.kind = '' # '' means beta from phase, can be 'betaamp', in the future 'betakmod'

        segment_inst.verify_object()
        return segment_inst

    @classmethod
    def _get_beamed_class(cls, new_class, beam):
        beam_mixin = _LhcB1Mixin if beam == 1 else _LhcB2Mixin
        beamed_class = type(new_class.__name__ + "B" + str(beam),
                            (new_class, beam_mixin),
                            {})
        return beamed_class

    def verify_object(self):  # TODO: Maybe more checks?
        """Verifies if everything is defined which should be defined
        """

        LOGGER.debug("Accelerator class verification")
        try:
            self.get_beam()
        except AttributeError:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete, beam "
                "has to be specified (--beam option missing?)."
            )

        if self.model_dir is None:  # is the class is used to create full response?
            if self.optics_file is None:
                raise AcceleratorDefinitionError(
                    "The accelerator definition is incomplete, optics "
                    "file or model directory has not been specified."
                )
            if self.xing is None:
                raise AcceleratorDefinitionError("Crossing on or off not set.")

        if self.excitation is None:
            raise AcceleratorDefinitionError("Excitation mode not set.")
        if (self.excitation == AccExcitationMode.ACD or
                self.excitation == AccExcitationMode.ADT):
            if self.drv_tune_x is None or self.drv_tune_y is None:
                raise AcceleratorDefinitionError("Driven tunes not set.")

        if self.optics_file is not None and not os.path.exists(self.optics_file):
            raise AcceleratorDefinitionError(
                "Optics file '{:s}' does not exist.".format(self.optics_file))

        # print info about the accelerator
        # TODO: write more output prints
        LOGGER.debug(
            "... verification passed. Will now print some information about the accelerator")
        LOGGER.debug("{:32s} {}".format("class name", self.__class__.__name__))
        LOGGER.debug("{:32s} {}".format("beam", self.get_beam()))
        LOGGER.debug("{:32s} {}".format("beam direction", self.get_beam_direction()))
        LOGGER.debug("")


    @classmethod
    def get_nominal_tmpl(cls):
        return cls.get_file("nominal.madx")

    @classmethod
    def get_nominal_multidpp_tmpl(cls):
        return cls.get_file("nominal_multidpp.madx")
    
    @classmethod
    def get_coupling_tmpl(cls):
        return cls.get_file("coupling_correct.madx")

    @classmethod
    def get_best_knowledge_tmpl(cls):
        return cls.get_file("best_knowledge.madx")

    @classmethod
    def get_segment_tmpl(cls):
        return cls.get_file("segment.madx")

    @classmethod
    def get_iteration_tmpl(cls):
        return cls.get_file("template.iterate.madx")

    @classmethod
    def get_basic_seq_tmpl(cls):
        return cls.get_file("template.basic_seq.madx")

    @classmethod
    def get_update_correction_tmpl(cls):
        return cls.get_file("template.update_correction.madx")

    @classmethod
    def get_file(cls, filename):
        return os.path.join(CURRENT_DIR, "lhc", filename)

    @classmethod
    def get_sequence_file(cls):
        try:
            return _get_file_for_year(cls.YEAR, "main.seq")
        except AttributeError:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete, mode " +
                "has to be specified (--lhcmode option missing?)."
            )

    @classmethod
    def get_variables(cls, frm=None, to=None, classes=None):
        correctors_dir = os.path.join(LHC_DIR, "2012", "correctors")
        all_corrs = _merge_jsons(
            os.path.join(correctors_dir, "correctors_b" + str(cls.get_beam()),
                         "beta_correctors.json"),
            os.path.join(correctors_dir, "correctors_b" + str(cls.get_beam()),
                         "coupling_correctors.json"),
            cls._get_triplet_correctors_file(),
        )
        my_classes = classes
        if my_classes is None:
            my_classes = all_corrs.keys()
        vars_by_class = set(_flatten_list(
            [all_corrs[corr_cls] for corr_cls in my_classes if corr_cls in all_corrs])
        )
        if frm is None and to is None:
            return list(vars_by_class)
        elems_matrix = tfs.read(
            cls._get_corrector_elems()
        ).sort_values("S")
        if frm is not None and to is not None:
            if frm > to:
                elems_matrix = elems_matrix[(elems_matrix.S >= frm) | (elems_matrix.S <= to)]
            else:
                elems_matrix = elems_matrix[(elems_matrix.S >= frm) & (elems_matrix.S <= to)]
        elif frm is not None:
            elems_matrix = elems_matrix[elems_matrix.S >= frm]
        elif to is not None:
            elems_matrix = elems_matrix[elems_matrix.S <= to]

        vars_by_position = _remove_dups_keep_order(_flatten_list(
            [raw_vars.split(",") for raw_vars in elems_matrix.loc[:, "VARS"]]
        ))
        return _list_intersect_keep_order(vars_by_position, vars_by_class)

    def get_update_correction_job(self, tiwss_out_path, corrections_file_path):
        """ Return string for madx job of correcting model """
        with open(self.get_update_correction_tmpl(), "r") as template:
            madx_template = template.read()
        try:
            replace_dict = {
                "LIB": self.MACROS_NAME,
                "MAIN_SEQ": self.load_main_seq_madx(),
                "OPTICS_PATH": self.optics_file,
                "CROSSING_ON": "1" if self.xing else "0",
                "NUM_BEAM": self.get_beam(),
                "DPP": self.dpp,
                "QMX": self.nat_tune_x,
                "QMY": self.nat_tune_y,
                "PATH_TWISS": tiwss_out_path,
                "CORRECTIONS": corrections_file_path,
            }
        except AttributeError:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete. " +
                "Needs to be an accelator instance. Also: --lhcmode or --beam option missing?"
            )
        return madx_template % replace_dict

    def get_basic_seq_job(self):
        """ Return string for madx job of correting model """
        with open(self.get_basic_seq_tmpl(), "r") as template:
            madx_template = template.read()
        try:
            replace_dict = {
                "LIB": self.MACROS_NAME,
                "MAIN_SEQ": self.load_main_seq_madx(),
                "OPTICS_PATH": self.optics_file,
                "CROSSING_ON": "1" if self.xing else "0",
                "NUM_BEAM": self.get_beam(),
                "DPP": self.dpp,
                "QMX": self.nat_tune_x,
                "QMY": self.nat_tune_y,
            }
        except AttributeError:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete. " +
                "Needs to be an accelator instance. Also: --lhcmode or --beam option missing?"
            )
        return madx_template % replace_dict

    def get_multi_dpp_job(self, dpp_list):
        """ Return madx job to create twisses (models) with dpps from dpp_list """
        with open(self.get_nominal_multidpp_tmpl()) as textfile:
            madx_template = textfile.read()
        try:
            output_path = self.model_dir
            use_acd = "1" if (self.excitation ==
                              AccExcitationMode.ACD) else "0"
            use_adt = "1" if (self.excitation ==
                              AccExcitationMode.ADT) else "0"
            crossing_on = "1" if self.xing else "0"
            beam = self.get_beam()

            replace_dict = {
                "LIB": self.MACROS_NAME,
                "MAIN_SEQ": self.load_main_seq_madx(),
                "OPTICS_PATH": self.optics_file,
                "NUM_BEAM": beam,
                "PATH": output_path,
                "QMX": self.nat_tune_x,
                "QMY": self.nat_tune_y,
                "USE_ACD": use_acd,
                "USE_ADT": use_adt,
                "CROSSING_ON": crossing_on,
                "QX": "",
                "QY": "",
                "QDX": "",
                "QDY": "",
                "DPP": "",
                "DPP_ELEMS": "",
                "DPP_AC": "",
                "DPP_ADT": "",
            }
            if (self.excitation in
                    (AccExcitationMode.ACD, AccExcitationMode.ADT)):
                replace_dict["QX"] = self.nat_tune_x
                replace_dict["QY"] = self.nat_tune_y
                replace_dict["QDX"] = self.drv_tune_x
                replace_dict["QDY"] = self.drv_tune_y
        except AttributeError:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete. " +
                "Needs to be an accelator instance. Also: --lhcmode or --beam option missing?"
            )

        # add different dpp twiss-command lines
        twisses_tmpl = "twiss, chrom, sequence=LHCB{beam:d}, deltap={dpp:f}, file='{twiss:s}';\n"
        for dpp in dpp_list:
            replace_dict["DPP"] += twisses_tmpl.format(
                beam=beam,
                dpp=dpp,
                twiss=os.path.join(output_path, "twiss_{:f}.dat".format(dpp))
            )
            replace_dict["DPP_ELEMS"] += twisses_tmpl.format(
                beam=beam,
                dpp=dpp,
                twiss=os.path.join(output_path, "twiss_{:f}_elements.dat".format(dpp))
            )
            replace_dict["DPP_AC"] += twisses_tmpl.format(
                beam=beam,
                dpp=dpp,
                twiss=os.path.join(output_path, "twiss_{:f}_ac.dat".format(dpp))
            )
            replace_dict["DPP_ADT"] += twisses_tmpl.format(
                beam=beam,
                dpp=dpp,
                twiss=os.path.join(output_path, "twiss_{:f}_adt.dat".format(dpp))
            )
        return madx_template % replace_dict

    LHC_IPS = ("1", "2", "5", "8")
    NORMAL_IP_BPMS = "BPMSW.1{side}{ip}.B{beam}"
    DOROS_IP_BPMS = "LHC.BPM.1{side}{ip}.B{beam}_DOROS"

    @classmethod
    def get_ips(cls):
        """ Returns an iterable with this accelerator IPs.

        Returns:
            An iterator returning tuples with:
                ("ip name", "left BPM name", "right BPM name")
        """
        beam = cls.get_beam()
        for ip in Lhc.LHC_IPS:
            yield ("IP{}".format(ip),
                   Lhc.NORMAL_IP_BPMS.format(side="L", ip=ip, beam=beam),
                   Lhc.NORMAL_IP_BPMS.format(side="R", ip=ip, beam=beam))
            yield ("IP{}_DOROS".format(ip),
                   Lhc.DOROS_IP_BPMS.format(side="L", ip=ip, beam=beam),
                   Lhc.DOROS_IP_BPMS.format(side="R", ip=ip, beam=beam))

    def log_status(self):
        LOGGER.info("  model dir = " + self.model_dir)
        LOGGER.info("{:20s} [{:10.3f}]".format("Natural Tune X", self.nat_tune_x))
        LOGGER.info("{:20s} [{:10.3f}]".format("Natural Tune Y", self.nat_tune_y))

        if self._model_best_knowledge is None:
            LOGGER.info("{:20s} [{:>10s}]".format("Best Knowledge Model", "NO"))
        else:
            LOGGER.info("{:20s} [{:>10s}]".format("Best Knowledge Model", "OK"))

        if self._excitation == AccExcitationMode.FREE:
            LOGGER.info("{:20s} [{:>10s}]".format("Excitation", "NO"))
        else:
            if self._excitation == AccExcitationMode.ACD:
                LOGGER.info("{:20s} [{:>10s}]".format("Excitation", "ACD"))
            elif self._excitation == AccExcitationMode.ADT:
                LOGGER.info("{:20s} [{:>10s}]".format("Excitation", "ADT"))
            LOGGER.info("{:20s} [{:10.3f}]".format("> Driven Tune X", self.drv_tune_x))
            LOGGER.info("{:20s} [{:10.3f}]".format("> Driven Tune Y", self.drv_tune_y))

    @classmethod
    def load_main_seq_madx(cls):
        try:
            return _get_call_main_for_year(cls.YEAR)
        except AttributeError:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete, mode " +
                "has to be specified (--lhcmode option missing?)."
            )

    # Private Methods ##########################################################

    @classmethod
    def _get_triplet_correctors_file(cls):
        correctors_dir = os.path.join(LHC_DIR, "2012", "correctors")
        return os.path.join(correctors_dir, "triplet_correctors.json")

    @classmethod
    def _get_corrector_elems(cls):
        correctors_dir = os.path.join(LHC_DIR, "2012", "correctors")
        return os.path.join(correctors_dir,
                            "corrector_elems_b" + str(cls.get_beam()) + ".tfs")

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


    def get_exciter_bpm(self, plane, commonbpms):
        beam = self.get_beam()
        adt = 'H.C' if plane == "X" else 'V.B'
        l_r = 'L' if (beam == 1 != plane == 'Y') else 'R'
        a_b = 'B' if beam == 1 else 'A'
        if self.excitation == AccExcitationMode.ACD:
            return self._is_one_of_in([f"BPMY{a_b}.6L4.B{beam}", f"BPM.7L4.B{beam}"],
                                      commonbpms), f"MKQA.6L4.B{beam}"
        if self.excitation == AccExcitationMode.ADT:
            return self._is_one_of_in([f"BPMWA.B5{l_r}4.B{beam}", f"BPMWA.A5{l_r}4.B{beam}"],
                                          commonbpms), f"ADTK{adt}5{l_r}4.B{beam}"

        return None

    @staticmethod
    def _is_one_of_in(bpms_to_find, bpms):
        found_bpms = [bpm for bpm in bpms_to_find if bpm in bpms]
        if len(found_bpms):
            return list(bpms).index(found_bpms[0]), found_bpms[0]
        raise KeyError

    def get_important_phase_advances(self):
        if self.get_beam() == 2:
            return[["MKD.O5R6.B2", "TCTPH.4R1.B2"],
                   ["MKD.O5R6.B2", "TCTPH.4R5.B2"]]
        if self.get_beam() == 1:
            return [["MKD.O5L6.B1", "TCTPH.4L1.B1"],
                    ["MKD.O5L6.B1", "TCTPH.4L5.B1"]]

    def get_exciter_name(self, plane):
        if self.get_beam() == 1:
            if self.excitation == AccExcitationMode.ACD:
                if plane == "H":
                    return 'MKQA.6L4.B1'
                elif plane == "V":
                    return 'MKQA.6L4.B1'
            elif self.excitation == AccExcitationMode.ADT:
                if plane == "H":
                    return "ADTKH.C5L4.B1"
                elif plane == "V":
                    return "ADTKV.B5R4.B1"
        elif self.get_beam() == 2:
            if self.excitation == AccExcitationMode.ACD:
                if plane == "H":
                    return 'MKQA.6L4.B2'
                elif plane == "V":
                    return 'MKQA.6L4.B2'
            elif self.excitation == AccExcitationMode.ADT:
                if plane == "H":
                    return "ADTKH.B5R4.B2"
                elif plane == "V":
                    return "ADTKV.C5L4.B2"
        return None

    def get_s_first_BPM(self):
        if self.get_beam() == 1:
            return self._model.loc["BPMSW.1L2.B1", "S"]
        elif self.get_beam() == 2:
            return self._model.loc["BPMSW.1L8.B2", "S"]
        return None

    def get_errordefspath(self):
        """Returns the path to the uncertainty definitions file (formerly called error definitions file.
        """
        if self._errordefspath is None:
            raise AttributeError("No error definitions file given in this accelerator instance.")
        return self._errordefspath

    def set_errordefspath(self, path):
        self._errordefspath = path

    def get_k_first_BPM(self, index):
        if self.get_beam() == 1:
            model_k = self._model.index.get_loc("BPMSW.1L2.B1")
            while model_k < len(self._model.index):
                kname = self._model.index[model_k]
                if kname in index:
                    return index.get_loc(kname)
                model_k = model_k + 1
        elif self.get_beam() == 2:
            model_k = self._model.index.get_loc("BPMSW.1L8.B2")
            while model_k < len(self._model.index):
                kname = self._model.index[model_k]
                if kname in index:
                    return index.get_loc(kname)
                model_k = model_k + 1
        return None

    def get_synch_BPMs(self, index):
        # expect passing index.values
        if self.get_beam() == 1:
            return [i in index for i in self.model_tfs.loc["BPMSW.33L2.B1":].index]
        elif self.get_beam() == 2:
            return [i in index for i in self.model_tfs.loc["BPMSW.33R8.B2":].index]

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

        re_dict = {
            "bpm": r"BPM",
            "magnet": r"M",
            "arc_bpm": r"BPM.*\.0*(1[5-9]|[2-9]\d|[1-9]\d{2,})[RL]",  # bpms > 14 L or R of IP
        }

        unknown_elements = [ty for ty in types if ty not in re_dict]
        if len(unknown_elements):
            raise TypeError("Unknown element(s): '{:s}'".format(str(unknown_elements)))

        series = pd.Series(list_of_elements)

        mask = series.str.match(re_dict[types[0]], case=False)
        for ty in types[1:]:
            mask = mask | series.str.match(re_dict[ty], case=False)
        return mask.values


class _LhcSegmentMixin(object):

    def __init__(self):
        self._start = None
        self._end = None

    def get_segment_vars(self, classes=None):
        return self.get_variables(frm=self.start.s,
                                  to=self.end.s,
                                  classes=classes)

    def verify_object(self):
        try:
            self.get_beam()
        except AttributeError:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete, beam "
                "has to be specified (--beam option missing?)."
            )
        if self.optics_file is None:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete, optics "
                "file has not been specified."
            )
        if self.xing is None:
            raise AcceleratorDefinitionError("Crossing on or off not set.")
        if self.label is None:
            raise AcceleratorDefinitionError("Segment label not set.")
        if self.start is None:
            raise AcceleratorDefinitionError("Segment start not set.")
        if self.end is None:
            raise AcceleratorDefinitionError("Segment end not set.")


class _LhcB1Mixin(object):
    @classmethod
    def get_beam(cls):
        return 1

    @classmethod
    def get_beam_direction(cls):
        return 1


class _LhcB2Mixin(object):
    @classmethod
    def get_beam(cls):
        return 2

    @classmethod
    def get_beam_direction(cls):
        return -1


class LhcAts(Lhc):
    MACROS_NAME = "lhc_runII_ats"


# Specific accelerator definitions ###########################################


class LhcRunI(Lhc):
    YEAR = "2012"

    @classmethod
    def load_main_seq_madx(cls):
        load_main_seq = _get_call_main_for_year("2012")
        load_main_seq += _get_madx_call_command(
            os.path.join(LHC_DIR, "2012", "install_additional_elements.madx")
        )
        return load_main_seq


class LhcRunII2015(Lhc):
    YEAR = "2015"


class LhcRunII2016(Lhc):
    YEAR = "2016"


class LhcRunII2016Ats(LhcAts, LhcRunII2016):
    pass


class LhcRunII2017(LhcAts):
    YEAR = "2017"


class LhcRunII2018(LhcAts):
    YEAR = "2018"


class HlLhc10(LhcAts):
    MACROS_NAME = "hllhc"
    YEAR = "hllhc1.0"

    @classmethod
    def load_main_seq_madx(cls):
        load_main_seq = _get_call_main_for_year("2015")
        load_main_seq += _get_call_main_for_year("hllhc1.0")
        return load_main_seq


class HlLhc12(LhcAts):
    MACROS_NAME = "hllhc"
    YEAR = "hllhc1.2"

    @classmethod
    def load_main_seq_madx(cls):
        load_main_seq = _get_call_main_for_year("2015")
        load_main_seq += _get_call_main_for_year("hllhc1.2")
        return load_main_seq

    @classmethod
    def _get_triplet_correctors_file(cls):
        correctors_dir = os.path.join(LHC_DIR, "hllhc1.2", "correctors")
        return os.path.join(correctors_dir, "triplet_correctors.json")

    @classmethod
    def _get_corrector_elems(cls):
        correctors_dir = os.path.join(LHC_DIR, "hllhc1.2", "correctors")
        return os.path.join(correctors_dir,
                            "corrector_elems_b" + str(cls.get_beam()) + ".tfs")


class HlLhc12NewCircuit(LhcAts):
    MACROS_NAME = "hllhc"
    YEAR = "hllhc12"


class HlLhc12NoQ2Trim(HlLhc12):
    MACROS_NAME = "hllhc"
    YEAR = "hllhc12"


class HlLhc13(LhcAts):
    MACROS_NAME = "hllhc"
    YEAR = "hllhc1.3"

    @classmethod
    def load_main_seq_madx(cls):
        load_main_seq = _get_madx_call_command(
            os.path.join(LHC_DIR, "hllhc1.3", "lhcrunIII.seq")
        )
        load_main_seq += _get_call_main_for_year("hllhc1.3")
        return load_main_seq

    @classmethod
    def _get_triplet_correctors_file(cls):
        correctors_dir = os.path.join(LHC_DIR, "hllhc1.3", "correctors")
        return os.path.join(correctors_dir, "triplet_correctors.json")

    @classmethod
    def _get_corrector_elems(cls):
        correctors_dir = os.path.join(LHC_DIR, "hllhc1.3", "correctors")
        return os.path.join(correctors_dir,
                            "corrector_elems_b" + str(cls.get_beam()) + ".tfs")


# General functions ##########################################################


def _get_call_main_for_year(year):
    call_main = _get_madx_call_command(
        _get_file_for_year(year, "main.seq")
    )
    return call_main


def _get_madx_call_command(path_to_call):
    command = "call, file = \""
    command += path_to_call
    command += "\";\n"
    return command


def _get_file_for_year(year, filename):
    return os.path.join(LHC_DIR, year, filename)


def _merge_jsons(*files):
    full_dict = {}
    for json_file in files:
        with open(json_file, "r") as json_data:
            json_dict = json.load(json_data)
            for key in json_dict.keys():
                full_dict[key] = json_dict[key]
    return full_dict


def _flatten_list(my_list):
    return [item for sublist in my_list for item in sublist]


def _remove_dups_keep_order(my_list):
    return list(OrderedDict.fromkeys(my_list))


def _list_intersect_keep_order(primary_list, secondary_list):
    return [elem for elem in primary_list if elem in secondary_list]
