"""
LHC
-------------------
"""
import json
import os
from collections import OrderedDict
from model.accelerators.accelerator import Accelerator, AcceleratorDefinitionError, AccExcitationMode
from utils import logging_tools
import tfs
from parser.entrypoint import EntryPointParameters

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
    """
    NAME = "lhc"
    MACROS_NAME = "lhc"
    RE_DICT = {"bpm": r"BPM", "magnet": r"M",
               "arc_bpm": r"BPM.*\.0*(1[5-9]|[2-9]\d|[1-9]\d{2,})[RL]"}  # bpms > 14 L or R of IP

    @staticmethod
    def get_class_parameters():
        params = EntryPointParameters()
        params.add_parameter(flags=["--lhcmode"], help=("LHC mode to use. Should be one of: " + str(get_lhc_modes().keys())), name="lhc_mode", type=str, choices=list(get_lhc_modes().keys()))
        params.add_parameter(flags=["--beam"], help="Beam to use.", name="beam", type=int,)
        return params

    # Entry-Point Wrappers #####################################################

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
    def get_lhc_error_dir(cls):
        return os.path.join(LHC_DIR, "systematic_errors")

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

    def get_synch_BPMs(self, index):
        # expect passing index.values
        if self.get_beam() == 1:
            return [i in index for i in self.model_tfs.loc["BPMSW.33L2.B1":].index]
        elif self.get_beam() == 2:
            return [i in index for i in self.model_tfs.loc["BPMSW.33R8.B2":].index]


class _LhcSegmentMixin(object):

    def __init__(self):
        self._start = None
        self._end = None

    def get_segment_vars(self, classes=None):
        return self.get_variables(frm=self.start.s, to=self.end.s, classes=classes)

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
