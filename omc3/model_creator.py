"""
Model Creator
-------------

Entrypoint to run the model creator for LHC, PSBooster and PS models.
"""
from __future__ import annotations

from pathlib import Path

from generic_parser import EntryPointParameters, entrypoint

from omc3.model import manager as model_manager
from omc3.model.accelerators.accelerator import Accelerator, AcceleratorDefinitionError
from omc3.model.constants import Fetcher
from omc3.model.model_creators.abstract_model_creator import ModelCreator 
from omc3.model.model_creators.manager import CreatorType, get_model_creator_class
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config
from omc3.utils.parsertools import print_help, require_param

LOGGER = logging_tools.get_logger(__name__)


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="outputdir",
        type=Path,
        help="Output path for model, twiss files will be writen here. [Required]",
    )
    params.add_parameter(
        name="type",
        choices=(CreatorType.NOMINAL.value, CreatorType.BEST_KNOWLEDGE.value),  # this script manages only these two
        default=CreatorType.NOMINAL.value,
        help="Type of model to create.",
    )
    params.add_parameter(
        name="logfile",
        type=Path,
        help=("Path to the file where to write the MAD-X script output."
              "If not provided it will be written to sys.stdout.")
    )
    params.add_parameter(
        name="show_help", 
        action="store_true", 
        help="Instructs the subsequent modules to print a help message"
    )
    params.update(get_fetcher_params())
    return params


def get_fetcher_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="fetch",
        type=str,
        help=("Select the fetcher which sets up the lattice definition (madx, seq, strength files)."
              "Note: not all fetchers might be available for the chosen Model Creator"),
        choices=[Fetcher.PATH.value, Fetcher.AFS.value]  # tuple(fetcher.value for fetcher in Fetcher) when GIT and LSA are implemented
    )
    params.add_parameter(
        name="path",
        type=PathOrStr,
        help="If path fetcher is selected, this option sets the path",
    )
    params.add_parameter(
        name="list_choices",
        action="store_true",
        help="If selected, a list of valid optics files is printed",
    )
    return params


# Main functions ###############################################################


@entrypoint(_get_params())
def create_instance_and_model(opt, accel_opt) -> Accelerator | None:
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

            choices: ``('nominal', 'best_knowledge')``


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
            accel_class = model_manager.get_accelerator_class(accel_opt)
            print(f"---- Accelerator {accel_class.__name__}  | Usage ----\n")
            print_help(accel_class.get_parameters())
        except Exception as e:
            LOGGER.debug(f"An error occurred: {e}")
            pass

        print("---- Model Creator | Usage ----\n")
        print_help(model_manager._get_params())
        print_help(_get_params())
        return 

    # proceed to the creator
    accel_inst: Accelerator = model_manager.get_accelerator(accel_opt)
    require_param("type", _get_params(), opt)
    LOGGER.debug(f"Accelerator Instance {accel_inst.NAME}, model type {opt.type}")
    
    # model_dir is used as the output directory
    require_param("outputdir", _get_params(), opt)
    outputdir = Path(opt.outputdir)
    accel_inst.model_dir = outputdir.absolute()

    creator_class = get_model_creator_class(accel_inst, opt.type)
    creator: ModelCreator = creator_class(accel_inst, logfile=opt.logfile)

    # Check if the options (i.e. the values of the arguments given) are valid choices.
    # This needs to be done by the creator itself, as it should know what valid options are
    # and can then also print them. If this fails, we have to abort.
    #
    # !! NOTE: If succesfull, THIS CAN MODIFY THE ACCELERATOR INSTANCE on creator.accel !!
    try:
        creator.prepare_options(opt)
    except AcceleratorDefinitionError:
        if not opt.list_choices:
            raise
        return
    
    # Save config only now, to not being written out for each time the choices are listed
    save_config(outputdir, opt=opt, unknown_opt=accel_opt, script=__file__)

    # Run the actual model creation
    creator.full_run()
    
    return accel_inst


if __name__ == "__main__":
    create_instance_and_model()
