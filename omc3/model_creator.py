"""
Model Creator
-------------

Entrypoint to run the model creator for LHC, PSBooster and PS models.
"""
from pathlib import Path

from generic_parser import EntryPointParameters, entrypoint, DotDict

from omc3.model import manager
from omc3.model.accelerators.accelerator import Accelerator
from omc3.model.model_creators.lhc_model_creator import (  # noqa
    LhcBestKnowledgeCreator,
    LhcModelCreator,
)
from omc3.model.model_creators.ps_model_creator import PsModelCreator
from omc3.model.model_creators.psbooster_model_creator import PsboosterModelCreator
from omc3.utils.iotools import create_dirs
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


CREATORS = {
    "lhc": {"nominal": LhcModelCreator,
            "best_knowledge": LhcBestKnowledgeCreator},
    "psbooster": {"nominal": PsboosterModelCreator},
    "ps": {"nominal": PsModelCreator},
}


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="type",
        choices=("nominal", "best_knowledge"),
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

    # Initialize from this model dir, so that elements are loaded
    # This should probably be done by the model-creator themselves instead
    new_accel_opt = _get_required_accelerator_parameters(accel_inst)
    new_accel_opt.accel = accel_inst.NAME
    new_accel_opt.model_dir = opt.outputdir
    accel_inst = manager.get_accelerator(new_accel_opt)

    return accel_inst




def _get_required_accelerator_parameters(accel_inst: Accelerator) -> DotDict:
    """Return the required parameters with the values from  the accelerator instance."""
    parameters_required = DotDict()
    parameters_accel = accel_inst.__class__.get_parameters()
    for name, param in parameters_accel.items():
        if param.get("required", False):
            parameters_required[name] = getattr(accel_inst, name)
    return parameters_required



if __name__ == "__main__":
    create_instance_and_model()
