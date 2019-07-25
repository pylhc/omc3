from model.model_creators import model_creator
from model.accelerators.accelerator import AccExcitationMode
import shutil
import os


class PsboosterModelCreator(model_creator.ModelCreator):

    @classmethod
    def get_madx_script(cls, instance, output_path):
        use_acd = "1" if (instance.excitation ==
                          AccExcitationMode.ACD) else "0"
        replace_dict = {
            "FILES_DIR": instance.get_psb_dir(),
            "RING": instance.get_ring(),
            "USE_ACD": use_acd,
            "NAT_TUNE_X": instance.nat_tune_x,
            "NAT_TUNE_Y": instance.nat_tune_y,
            "KINETICENERGY": instance.energy,
            "DPP": instance.dpp,
            "OUTPUT": output_path,
            "DRV_TUNE_X": "",
            "DRV_TUNE_Y": "",
        }
        if use_acd:
            replace_dict["DRV_TUNE_X"] = instance.drv_tune_x
            replace_dict["DRV_TUNE_Y"] = instance.drv_tune_y

        with open(instance.get_nominal_tmpl()) as textfile:
            madx_template = textfile.read()

        return madx_template % replace_dict

    @classmethod
    def _prepare_fullresponse(cls, instance, output_path):
        with open(instance.get_iteration_tmpl()) as textfile:
            iterate_template = textfile.read()

        replace_dict = {
            "FILES_DIR": instance.get_psb_dir(),
            "RING": instance.get_ring(),
            "LIB": instance.NAME,  # "psbooster"
            "OPTICS_PATH": instance.modifiers_file,
            "PATH": output_path,
            "KINETICENERGY": instance.energy,
            "NAT_TUNE_X": instance.nat_tune_x,
            "NAT_TUNE_Y": instance.nat_tune_y,
            "DRV_TUNE_X": "",
            "DRV_TUNE_Y": "",
            "DPP": instance.dpp,
            "OUTPUT": output_path,
        }

        with open(os.path.join(output_path,
                               "job.iterate.madx"), "w") as textfile:
            textfile.write(iterate_template % replace_dict)

    @classmethod
    def _prepare_corrtest(cls, instance, output_path):
        """ Partially fills mask file for tests of corrections
            Reads correction_test.madx (defined in psbooster.get_corrtest_tmpl()) 
            and produces correction_test.mask2.madx.
            Java GUI fills the remaining fields 
           """
        with open(instance.get_corrtest_tmpl()) as textfile:
            template = textfile.read()

        replace_dict = {
            "KINETICENERGY": instance.energy,
            "FILES_DIR": instance.get_psb_dir(),
            "RING": instance.get_ring(),
            "NAT_TUNE_X": instance.nat_tune_x,
            "NAT_TUNE_Y": instance.nat_tune_y,
            "DPP": instance.dpp,
            "PATH": "%TESTPATH",  # field filled later by Java GUI
            "COR": "%COR"  # field filled later by Java GUI
        }

        with open(os.path.join(output_path,
                               "correction_test.mask2.madx"), "w") as textfile:
            textfile.write(template % replace_dict)

    @classmethod
    def prepare_run(cls, instance, output_path):
        if instance.fullresponse:
            cls._prepare_fullresponse(instance, output_path)
            cls._prepare_corrtest(instance, output_path)

        file_name = "error_deff_ring" + str(instance.get_ring()) + ".txt"
        file_path = instance.get_psb_dir()
        src_path = os.path.join(file_path, file_name)
        dest_path = os.path.join(output_path, "error_deffs.txt")

        shutil.copy(src_path, dest_path)

        # os.link(src, dst) (file_path, link_path)


class PsboosterSegmentCreator(model_creator.ModelCreator):
    @classmethod
    def get_madx_script(cls, instance, output_path):
        with open(instance.get_segment_tmpl()) as textfile:
            madx_template = textfile.read()
        replace_dict = {
            "FILES_DIR": instance.get_psb_dir(),
            "RING": instance.ring,
            "NAT_TUNE_X": instance.nat_tune_x,
            "NAT_TUNE_Y": instance.nat_tune_y,
            "LIB": instance.NAME,  # "psbooster"
            "OPTICS_PATH": instance.modifiers_file,
            "PATH": output_path,
            "OUTPUT": output_path,
            "LABEL": instance.label,
            "BETAKIND": instance.kind,
            "STARTFROM": instance.start.name,
            "ENDAT": instance.end.name,
        }

        madx_script = madx_template % replace_dict
        return madx_script
