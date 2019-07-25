import logging
import os
import sys
import shutil
from model.model_creators import model_creator
from model.accelerators.accelerator import AccExcitationMode

LOGGER = logging.getLogger(__name__)


class LhcModelCreator(model_creator.ModelCreator):

    @classmethod
    def get_madx_script(cls, lhc_instance, output_path):
        use_acd = "1" if (lhc_instance.excitation ==
                          AccExcitationMode.ACD) else "0"
        use_adt = "1" if (lhc_instance.excitation ==
                          AccExcitationMode.ADT) else "0"
        crossing_on = "1" if lhc_instance.xing else "0"
        beam = lhc_instance.get_beam()

        replace_dict = {
            "LIB": lhc_instance.MACROS_NAME,
            "MAIN_SEQ": lhc_instance.load_main_seq_madx(),
            "OPTICS_PATH": lhc_instance.modifiers_file,
            "NUM_BEAM": beam,
            "PATH": output_path,
            "QMX": lhc_instance.nat_tune_x,
            "QMY": lhc_instance.nat_tune_y,
            "USE_ACD": use_acd,
            "USE_ADT": use_adt,
            "DPP": lhc_instance.dpp,
            "CROSSING_ON": crossing_on,
            "QX": "", "QY": "", "QDX": "", "QDY": "",
        }
        if (lhc_instance.excitation in
                (AccExcitationMode.ACD, AccExcitationMode.ADT)):
            replace_dict["QX"] = lhc_instance.nat_tune_x
            replace_dict["QY"] = lhc_instance.nat_tune_y
            replace_dict["QDX"] = lhc_instance.drv_tune_x
            replace_dict["QDY"] = lhc_instance.drv_tune_y

        with open(lhc_instance.get_nominal_tmpl()) as textfile:
            madx_template = textfile.read()

        madx_script = madx_template % replace_dict
        return madx_script

    @classmethod
    def prepare_run(cls, lhc_instance, output_path):
        if lhc_instance.fullresponse:
            cls._prepare_fullresponse(lhc_instance, output_path)
        if lhc_instance.energy is not None:
            file_name = f"{int(lhc_instance.energy*1000):04d}GeV.tfs"
            file_path = lhc_instance.get_lhc_error_dir()
            src_path = os.path.join(file_path, file_name)
            dest_path = os.path.join(output_path, "error_deffs.txt")
            shutil.copy(src_path, dest_path)


    @classmethod
    def _prepare_fullresponse(cls, lhc_instance, output_path):
        with open(lhc_instance.get_iteration_tmpl()) as textfile:
            iterate_template = textfile.read()

        crossing_on = "1" if lhc_instance.xing else "0"
        replace_dict = {
            "LIB": lhc_instance.MACROS_NAME,
            "MAIN_SEQ": lhc_instance.load_main_seq_madx(),
            "OPTICS_PATH": lhc_instance.modifiers_file,
            "NUM_BEAM": lhc_instance.get_beam(),
            "PATH": output_path,
            "QMX": lhc_instance.nat_tune_x,
            "QMY": lhc_instance.nat_tune_y,
            "CROSSING_ON": crossing_on,
        }

        with open(os.path.join(output_path,
                               "job.iterate.madx"), "w") as textfile:
            textfile.write(iterate_template % replace_dict)


class LhcBestKnowledgeCreator(LhcModelCreator):

    @classmethod
    def get_madx_script(cls, lhc_instance, output_path):
        if lhc_instance.excitation is not AccExcitationMode.FREE:
            raise model_creator.ModelCreationError(
                "Don't set ACD or ADT for best knowledge model."
            )
        if lhc_instance.energy is None:
            raise model_creator.ModelCreationError(
                "Best knowledge model requires energy."
            )
        with open(lhc_instance.get_best_knowledge_tmpl()) as textfile:
            madx_template = textfile.read()
        crossing_on = "1" if lhc_instance.xing else "0"
        replace_dict = {
            "LIB": lhc_instance.MACROS_NAME,
            "MAIN_SEQ": lhc_instance.load_main_seq_madx(),
            "OPTICS_PATH": lhc_instance.modifiers_file,
            "NUM_BEAM": lhc_instance.get_beam(),
            "PATH": output_path,
            "DPP": lhc_instance.dpp,
            "QMX": lhc_instance.nat_tune_x,
            "QMY": lhc_instance.nat_tune_y,
            "ENERGY": lhc_instance.energy,
            "CROSSING_ON": crossing_on,
        }
        madx_script = madx_template % replace_dict
        return madx_script


class LhcSegmentCreator(model_creator.ModelCreator):
    @classmethod
    def get_madx_script(cls, lhc_instance, output_path):
        with open(lhc_instance.get_segment_tmpl()) as textfile:
            madx_template = textfile.read()
        replace_dict = {
            "LIB": lhc_instance.MACROS_NAME,
            "MAIN_SEQ": lhc_instance.load_main_seq_madx(),
            "OPTICS_PATH": lhc_instance.modifiers_file,
            "NUM_BEAM": lhc_instance.get_beam(),
            "PATH": output_path,
            "LABEL": lhc_instance.label,
            "BETAKIND": lhc_instance.kind,
            "STARTFROM": lhc_instance.start.name,
            "ENDAT": lhc_instance.end.name,
        }
        madx_script = madx_template % replace_dict
        return madx_script


class LhcCouplingCreator(model_creator.ModelCreator):
    @classmethod
    def get_madx_script(cls, lhc_instance, output_path):
        with open(lhc_instance.get_coupling_tmpl()) as textfile:
            madx_template = textfile.read()
            print(madx_template)
        crossing_on = "1" if lhc_instance.xing else "0"
        replace_dict = {
            "LIB": lhc_instance.MACROS_NAME,
            "MAIN_SEQ": lhc_instance.load_main_seq_madx(),
            "OPTICS_PATH": lhc_instance.modifiers_file,
            "NUM_BEAM": lhc_instance.get_beam(),
            "PATH": output_path,
            "QMX": lhc_instance.nat_tune_x,
            "QMY": lhc_instance.nat_tune_y,
            "CROSSING_ON": crossing_on,

        }
        madx_script = madx_template % replace_dict
        return madx_script
