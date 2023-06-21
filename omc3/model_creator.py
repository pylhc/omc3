"""
Model Creator
-------------

Entrypoint to run the model creator for LHC, PSBooster and PS models.
"""
from pathlib import Path

from generic_parser import EntryPointParameters, entrypoint

from omc3.model import manager
from omc3.model.accelerators.accelerator import Accelerator
from omc3.model.model_creators.lhc_model_creator import (  # noqa
    LhcBestKnowledgeCreator,
    LhcCorrectionModelCreator,
    LhcModelCreator,
)
from omc3.model.model_creators.ps_model_creator import PsModelCreator
from omc3.model.model_creators.psbooster_model_creator import PsboosterModelCreator
from omc3.utils.iotools import create_dirs
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


CREATORS = {
    "lhc": {"nominal": LhcModelCreator,
            "best_knowledge": LhcBestKnowledgeCreator,
            "correction": LhcCorrectionModelCreator},
    "psbooster": {"nominal": PsboosterModelCreator},
    "ps": {"nominal": PsModelCreator},
}


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="type",
        choices=("nominal", "best_knowledge", "coupling_correction", "segment"),
        help="Type of model to create.",
        default='nominal',
    )
    params.add_parameter(
        name="outputdir",
        required=True,
        type=Path,
        help="Output path for model, twiss files will be writen here."
    )
    params.add_parameter(
        name="logfile",
        type=Path,
        help=("Path to the file where to write the MAD-X script output."
              "If not provided it will be written to sys.stdout.")
    )
    return params


# Main functions ###############################################################


@entrypoint(_get_params())
def create_instance_and_model(opt, accel_opt) -> Accelerator:
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

            Type of model to create.

            choices: ``('nominal', 'best_knowledge', 'correction', 'segment')``


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
    # Prepare paths
    create_dirs(opt.outputdir)

    accel_inst = manager.get_accelerator(accel_opt)
    accel_inst.model_dir = opt.outputdir

    LOG.info(f"Accelerator Instance {accel_inst.NAME}, model type {opt.type}")
    creator = CREATORS[accel_inst.NAME][opt.type](accel_inst, logfile=opt.logfile)
    creator.full_run()
    return accel_inst


if __name__ == "__main__":
    create_instance_and_model()
