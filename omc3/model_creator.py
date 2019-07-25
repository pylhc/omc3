import sys
import logging
from model import manager
from utils.iotools import create_dirs
from generic_parser.entrypoint import EntryPointParameters, entrypoint
from model.model_creators.lhc_model_creator import (  # noqa
    LhcModelCreator,
    LhcBestKnowledgeCreator,
    LhcSegmentCreator,
    LhcCouplingCreator,
)
from model.model_creators.psbooster_model_creator import PsboosterModelCreator, PsboosterSegmentCreator
from model.model_creators.ps_model_creator import PsModelCreator, PsSegmentCreator

LOGGER = logging.getLogger("__name__")

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
    params.add_parameter(
        flags=["--type"],
        name="type",
        help="Type of model to create, either nominal or best_knowledge",
        choices=("nominal", "best_knowledge", "coupling_correction"),
    )
    params.add_parameter(
        flags=["--output"],
        help="Output path for model, twiss files will be writen here.",
        name="output",
        required=True,
        type=str,
    )
    params.add_parameter(
        flags=["--writeto"],
        help="Path to the file where to write the resulting MAD-X script. ",
        name="writeto",
        type=str,
    )
    params.add_parameter(
        flags=["--logfile"],
        help=("Path to the file where to write the MAD-X script output."
              "If not provided it will be written to sys.stdout."),
        name="logfile",
        type=str,
    )
    return params


# Main functions ###############################################################


@entrypoint(_get_params())
def create_instance_and_model(opt, accel_opt):
    numeric_level = 0
    if (sys.flags.debug):
        numeric_level = getattr(logging, "DEBUG", None)
        #print "DEBUG Level %d" % numeric_level
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(' %(asctime)s %(levelname)s | %(name)s : %(message)s')
        ch.setFormatter(formatter)
        logging.getLogger().addHandler(ch)
        logging.getLogger().setLevel(numeric_level)
        
    else:
        numeric_level = getattr(logging, "WARNING", None)
        #print "WARNING Level %d" % numeric_level
        logging.basicConfig(level=numeric_level) # warning level to stderr
    
    
    create_dirs(opt.output)
    accel_inst = manager.get_accel_instance(accel_opt)
    create_model(
        accel_inst,
        opt.type,
        opt.output,
        writeto=opt.writeto,
        logfile=opt.logfile,
    )


def create_model(accel_inst, model_type, output_path, **kwargs):
    LOGGER.info("Accelerator Instance <%s>, model type <%s>", accel_inst.NAME, model_type )
    CREATORS[accel_inst.NAME][model_type].create_model(
        accel_inst,
        output_path,
        **kwargs
    )


if __name__ == "__main__":
    create_instance_and_model()
