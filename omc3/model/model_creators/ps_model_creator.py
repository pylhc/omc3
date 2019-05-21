from model.model_creators import model_creator
import os
import logging
import shutil

LOGGER = logging.getLogger("__name__")

class PsModelCreator(model_creator.ModelCreator):

    @classmethod
    def get_madx_script(cls, instance, output_path):
        replace_dict = {
            "FILES_DIR": instance.get_ps_dir(),
            "USE_ACD": 1 if instance.acd else 0,
            "NAT_TUNE_X": instance.nat_tune_x,
            "NAT_TUNE_Y": instance.nat_tune_y,
            "KINETICENERGY": instance.energy,
            "DPP": instance.dpp,
            "OUTPUT": output_path,
            "DRV_TUNE_X": "", 
            "DRV_TUNE_Y": "",
            "OPTICS_PATH": instance.optics_file,
        }
        LOGGER.info("instance name <%s>", instance.NAME)

        
        if instance.acd:
            replace_dict["DRV_TUNE_X"] = instance.drv_tune_x
            replace_dict["DRV_TUNE_Y"] = instance.drv_tune_y
            LOGGER.debug("ACD is ON. Driven tunes %f %f",replace_dict["DRV_TUNE_X"], replace_dict["DRV_TUNE_Y"])
        else:
            LOGGER.debug("ACD is OFF")

        with open(instance.get_nominal_tmpl()) as textfile:
            madx_template = textfile.read()
        
        #print(replace_dict)
        #print(madx_template)
        
        out = madx_template % replace_dict
        
        #print(out)
        
        return out
    
    @classmethod
    def _prepare_fullresponse(cls, instance, output_path):
        with open(instance.get_iteration_tmpl()) as textfile:
            iterate_template = textfile.read()

       
        replace_dict = {
            "FILES_DIR": instance.get_ps_dir(),
            "LIB": instance.MACROS_NAME,
            "OPTICS_PATH": instance.optics_file,
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
        #print("error file: src=%s dst=%s"%(src_path,dest_path))
        shutil.copy(src_path, dest_path)

class PsSegmentCreator(model_creator.ModelCreator):
    @classmethod
    def get_madx_script(cls, instance, output_path):
        """ instance is Ps class
        """
        
        LOGGER.info('instance.energy %f',instance.energy)
        
        with open(instance.get_segment_tmpl()) as textfile:
            madx_template = textfile.read()
        replace_dict = {
            "KINETICENERGY": instance.energy,
            "NAT_TUNE_X": instance.nat_tune_x,
            "NAT_TUNE_Y": instance.nat_tune_y,
            "FILES_DIR": instance.get_ps_dir(),
            "OPTICS_PATH": instance.optics_file,
            "PATH": output_path,
            "LABEL": instance.label,
            "BETAKIND": instance.kind,
            "STARTFROM": instance.start.name,
            "ENDAT": instance.end.name,
        }
        madx_script = madx_template % replace_dict
        return madx_script
            
