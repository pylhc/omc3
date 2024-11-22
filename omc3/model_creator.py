"""
Model Creator
-------------

Entrypoint to run the model creator for LHC, PSBooster and PS models.
"""
from pathlib import Path

from generic_parser import EntryPointParameters, entrypoint

from omc3.model import manager
from omc3.model.accelerators.accelerator import Accelerator
from omc3.model.constants import AFSFETCHER, PATHFETCHER
from omc3.model.model_creators import abstract_model_creator
from omc3.model.model_creators.lhc_model_creator import (  # noqa
    LhcBestKnowledgeCreator,
    LhcModelCreator,
)
from omc3.model.model_creators.ps_model_creator import PsModelCreator
from omc3.model.model_creators.psbooster_model_creator import BoosterModelCreator
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config
from omc3.utils.parsertools import print_help, require_param

LOGGER = logging_tools.get_logger(__name__)

NOMINAL: str = "nominal"
BEST_KNOWLEDGE: str = "best_knowledge"

CREATORS = {
    "lhc": {
        NOMINAL: LhcModelCreator,
        BEST_KNOWLEDGE: LhcBestKnowledgeCreator,
    },
    "psbooster": {
        NOMINAL: BoosterModelCreator
    },
    "ps": {
        NOMINAL: PsModelCreator
    },
}


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="type",
        choices=(NOMINAL, BEST_KNOWLEDGE),
        help="Type of model to create. [Required]",
    )
    params.add_parameter(
        name="outputdir",
        type=Path,
        help="Output path for model, twiss files will be writen here. [Required]",
    )
    params.add_parameter(
        name="logfile",
        type=Path,
        help=("Path to the file where to write the MAD-X script output."
              "If not provided it will be written to sys.stdout.")
    )
    params.add_parameter(
        name="fetch",
        type=str,
        help=("Select the fetcher which sets up the lattice definition (madx, seq, strength files)."
              "Note: not all fetchers might be available for the chosen Model Creator"),
        choices=[PATHFETCHER, AFSFETCHER]  # [PATHFETCHER, AFSFETCHER, GITFETCHER, LSAFETCHER]
    )
    params.add_parameter(
        name="path",
        type=PathOrStr,
        help="If path fetcher is selected, this option sets the path",
    )
    params.add_parameter(
        name="list_choices",
        action="store_true",
        help="if selected, a list of valid optics files is printed",
    )
    params.add_parameter(name="show_help", action="store_true", help="instructs the subsequent modules to print a help message")
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
    # first, if help is requested, gather all help info and print it
    if opt.show_help:
        try:
            #with silence():
            accel_class = manager.get_accelerator_class(accel_opt)
            print(f"---- Accelerator {accel_class.__name__}  | Usage ----\n")
            print_help(accel_class.get_parameters())
        except Exception as e:
            LOGGER.debug(f"An error occurred: {e}")
            pass

        print("---- Model Creator | Usage ----\n")
        print_help(manager._get_params())
        print_help(_get_params())
        return 

    # proceed to the creator
    accel_inst = manager.get_accelerator(accel_opt)
    require_param("type", _get_params(), opt)
    LOGGER.debug(f"Accelerator Instance {accel_inst.NAME}, model type {opt.type}")
    
    # model_dir is used as the output directory
    require_param("outputdir", _get_params(), opt)
    accel_inst.model_dir = Path(opt.outputdir).absolute()

    creator: abstract_model_creator.ModelCreator = CREATORS[accel_inst.NAME][opt.type](accel_inst, logfile=opt.logfile)

    # Check if the options (i.e. the values of the arguments given) are valid choices.
    # This needs to be done by the creator itself, as it should know what valid options are
    # and can then also print them. If this fails, we have to abort.
    if not creator.check_options(opt):
        return None
    
    # Save config only now, to not being written out for each time the choices are listed
    save_config(Path(opt.outputdir), opt=opt, unknown_opt=accel_opt, script=__file__)

    # Run the actual model creation
    creator.full_run()
    
    return accel_inst


if __name__ == "__main__":
    create_instance_and_model()
