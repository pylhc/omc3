from model.model_creators import model_creator
from model.accelerators.accelerator import AccExcitationMode
import os
import logging
import shutil

LOGGER = logging.getLogger("__name__")


class PsModelCreator(model_creator.ModelCreator):

    @classmethod
    def get_madx_script(cls, instance, output_path):
        use_acd = "1" if (instance.excitation ==
                          AccExcitationMode.ACD) else "0"
        replace_dict = {
            "FILES_DIR": instance.get_ps_dir(),
            "USE_ACD": use_acd,
            "NAT_TUNE_X": instance.nat_tune_x,
            "NAT_TUNE_Y": instance.nat_tune_y,
            "KINETICENERGY": instance.energy,
            "DPP": instance.dpp,
            "OUTPUT": output_path,
            "DRV_TUNE_X": "", 
            "DRV_TUNE_Y": "",
            "OPTICS_PATH": instance.modifiers_file,
        }
        LOGGER.info(f"instance name {instance.NAME}")
        if use_acd:
            replace_dict["DRV_TUNE_X"] = instance.drv_tune_x
            replace_dict["DRV_TUNE_Y"] = instance.drv_tune_y
            LOGGER.debug(f"ACD is ON. Driven tunes {replace_dict['DRV_TUNE_X']}, {replace_dict['DRV_TUNE_Y']}")
        else:
            LOGGER.debug("ACD is OFF")

        with open(instance.get_nominal_tmpl()) as textfile:
            madx_template = textfile.read()
        out = madx_template % replace_dict
        return out
    
    @classmethod
    def _prepare_fullresponse(cls, instance, output_path):
        with open(instance.get_iteration_tmpl()) as textfile:
            iterate_template = textfile.read()

        replace_dict = {
            "FILES_DIR": instance.get_ps_dir(),
            "LIB": instance.MACROS_NAME,
            "OPTICS_PATH": instance.modifiers_file,
            "PATH": output_path,
            "KINETICENERGY": instance.energy,
            "NAT_TUNE_X": instance.nat_tune_x,
            "NAT_TUNE_Y": instance.nat_tune_y,
            "DRV_TUNE_X": "", 
            "DRV_TUNE_Y": "",
        }

        with open(os.path.join(output_path,
                               "job.iterate.madx"), "w") as textfile:
            textfile.write(iterate_template % replace_dict)

    @classmethod
    def prepare_run(cls, instance, output_path):
        if instance.fullresponse:
            cls._prepare_fullresponse(instance, output_path)
        
        # get path of file from PS model directory (without year at the end)
        src_path = instance.get_file("error_deff.txt")
        dest_path = os.path.join(output_path, "error_deffs.txt")
        shutil.copy(src_path, dest_path)


class PsSegmentCreator(model_creator.ModelCreator):
    @classmethod
    def get_madx_script(cls, instance, output_path):
        """ instance is Ps class"""
        LOGGER.info(f"instance.energy {instance.energy}")
        
        with open(instance.get_segment_tmpl()) as textfile:
            madx_template = textfile.read()
        replace_dict = {
            "KINETICENERGY": instance.energy,
            "NAT_TUNE_X": instance.nat_tune_x,
            "NAT_TUNE_Y": instance.nat_tune_y,
            "FILES_DIR": instance.get_ps_dir(),
            "OPTICS_PATH": instance.modifiers_file,
            "PATH": output_path,
            "LABEL": instance.label,
            "BETAKIND": instance.kind,
            "STARTFROM": instance.start.name,
            "ENDAT": instance.end.name,
        }
        madx_script = madx_template % replace_dict
        return madx_script

