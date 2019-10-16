import sys
import logging
from model import manager
from utils.iotools import create_dirs
from generic_parser import EntryPointParameters, entrypoint
from model.model_creators.lhc_model_creator import (  # noqa
    LhcModelCreator,
    LhcBestKnowledgeCreator,
    LhcSegmentCreator,
    LhcCouplingCreator,
)
from model.model_creators.psbooster_model_creator import PsboosterModelCreator, PsboosterSegmentCreator
from model.model_creators.ps_model_creator import PsModelCreator, PsSegmentCreator

LOGGER = logging.getLogger(__name__)

CREATORS = {
    "lhc": {"nominal": LhcModelCreator,
            "best_knowledge": LhcBestKnowledgeCreator,
            "segment": LhcSegmentCreator,
            "coupling_correction": LhcCouplingCreator},
    "psbooster": {"nominal": PsboosterModelCreator,
                  "segment": PsboosterSegmentCreator},
    "ps": {"nominal": PsModelCreator,
           "segment": PsSegmentCreator},
}


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(name="type", choices=("nominal", "best_knowledge", "coupling_correction"),
                         help="Type of model to create, either nominal or best_knowledge")
    params.add_parameter(name="output", required=True, type=str,
                         help="Output path for model, twiss files will be writen here.")
    params.add_parameter(name="writeto", type=str,
                         help="Path to the file where to write the resulting MAD-X script.")
    params.add_parameter(name="logfile", type=str,
                         help=("Path to the file where to write the MAD-X script output."
                               "If not provided it will be written to sys.stdout."))
    return params


# Main functions ###############################################################


@entrypoint(_get_params())
def create_instance_and_model(opt, accel_opt):
    if sys.flags.debug:
        numeric_level = getattr(logging, "DEBUG", None)
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(' %(asctime)s %(levelname)s | %(name)s : %(message)s')
        ch.setFormatter(formatter)
        logging.getLogger().addHandler(ch)
        logging.getLogger().setLevel(numeric_level)
        
    else:
        numeric_level = getattr(logging, "WARNING", None)
        logging.basicConfig(level=numeric_level) # warning level to stderr

    create_dirs(opt.output)
    accel_inst = manager.get_accel_instance(accel_opt)
    create_model(accel_inst, opt.type, opt.output, writeto=opt.writeto, logfile=opt.logfile)


def create_model(accel_inst, model_type, output_path, **kwargs):
    LOGGER.info(f"Accelerator Instance {accel_inst.NAME}, model type {model_type}")
    CREATORS[accel_inst.NAME][model_type].create_model(accel_inst, output_path, **kwargs)


if __name__ == "__main__":
    create_instance_and_model()
