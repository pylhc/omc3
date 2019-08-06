import logging
import madx_wrapper

LOGGER = logging.getLogger(__name__)


class ModelCreator(object):

    @classmethod
    def create_model(creator, instance, output_path, **kwargs):
        LOGGER.info("instance name <%s>", instance.NAME)
        
        instance.verify_object()
        madx_script = creator.get_madx_script(
            instance,
            output_path
        )
        creator.prepare_run(instance, output_path)
        writeto = kwargs.get("writeto", None)
        logfile = kwargs.get("logfile", None)
        creator.run_madx(madx_script, logfile, writeto)

    @classmethod
    def prepare_run(cls, acc_instance, output_path):
        if acc_instance.fullresponse:
            cls._prepare_fullresponse(acc_instance, output_path)
            
    @staticmethod
    def run_madx(madx_script, logfile=None, writeto=None):
        madx_wrapper.resolve_and_run_string(
            madx_script,
            output_file=writeto,
            log_file=logfile
        )


class ModelCreationError(Exception):
    """
    Raised when an error happens during model creation.
    """
    pass
