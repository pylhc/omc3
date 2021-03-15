"""
Model Creator
-------------

Entrypoint to run the model creator for LHC, PSBooster and PS models.
"""
import logging
import sys

from generic_parser import EntryPointParameters, entrypoint

from omc3.madx_wrapper import run_string
from omc3.model import manager
from omc3.model.model_creators.lhc_model_creator import (  # noqa
    LhcBestKnowledgeCreator,
    LhcCouplingCreator,
    LhcModelCreator,
)
from omc3.model.model_creators.ps_model_creator import PsModelCreator
from omc3.model.model_creators.psbooster_model_creator import PsboosterModelCreator
from omc3.model.model_creators.segment_creator import SegmentCreator
from omc3.utils.iotools import create_dirs

LOGGER = logging.getLogger(__name__)

CREATORS = {
    "lhc": {"nominal": LhcModelCreator,
            "best_knowledge": LhcBestKnowledgeCreator,
            "segment": SegmentCreator,
            "coupling_correction": LhcCouplingCreator},
    "psbooster": {"nominal": PsboosterModelCreator,
                  "segment": SegmentCreator},
    "ps": {"nominal": PsModelCreator,
           "segment": SegmentCreator},
}


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(name="type", choices=("nominal", "best_knowledge", "coupling_correction"),
                         help="Type of model to create, either nominal or best_knowledge")
    params.add_parameter(name="outputdir", required=True, type=str,
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
    """
    Manager Keyword Args:
        *--Required--*

        - **accel**:

            Choose the accelerator to use.Can be the class already.

            choices: ``['lhc', 'ps', 'esrf', 'psbooster', 'skekb', 'JPARC', 'petra', 'iota']``


    Creator Keyword Args:
        *--Required--*

        - **outputdir** *(str)*:

            Output path for model, twiss files will be writen here.


        *--Optional--*

        - **logfile** *(str)*:

            Path to the file where to write the MAD-X script output.If not
            provided it will be written to sys.stdout.


        - **type**:

            Type of model to create, either nominal or best_knowledge

            choices: ``('nominal', 'best_knowledge', 'coupling_correction')``


        - **writeto** *(str)*:

            Path to the file where to write the resulting MAD-X script.


    Accelerator Keyword Args:
        lhc: :mod:`omc3.model.accelerators.lhc`

        ps: :mod:`omc3.model.accelerators.ps`

        esrf: :mod:`omc3.model.accelerators.esrf`

        psbooster: :mod:`omc3.model.accelerators.psbooster`

        skekb: :mod:`omc3.model.accelerators.skekb`

        iota: :mod:`omc3.model.accelerators.iota`

        petra: :mod:`omc3.model.accelerators.petra` (not implemented)

        JPARC: Not implemented
    """
    if sys.flags.debug:
        numeric_level = getattr(logging, "DEBUG", None)
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(' %(asctime)s %(levelname)s | %(name)s : %(message)s')
        ch.setFormatter(formatter)
        logging.getLogger().addHandler(ch)
        logging.getLogger().setLevel(numeric_level)
        
    else:
        numeric_level = getattr(logging, "WARNING", None)
        logging.basicConfig(level=numeric_level)  # warning level to stderr

    create_dirs(opt.outputdir)

    accel_inst = manager.get_accelerator(accel_opt)
    LOGGER.info(f"Accelerator Instance {accel_inst.NAME}, model type {opt.type}")
    accel_inst.verify_object()
    creator = CREATORS[accel_inst.NAME][opt.type]
    creator.prepare_run(accel_inst, opt.outputdir)
    madx_script = creator.get_madx_script(accel_inst, opt.outputdir)
    run_string(madx_script, output_file=opt.writeto, log_file=opt.logfile)


if __name__ == "__main__":
    create_instance_and_model()
