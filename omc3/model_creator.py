"""
Model Creator
-------------

Entrypoint to run the model creator for LHC, PSBooster and PS models.
"""
from pathlib import Path

from generic_parser import EntryPointParameters, entrypoint

from omc3.madx_wrapper import run_string
from omc3.model import manager
from omc3.model.accelerators.accelerator import Accelerator
from omc3.model.constants import JOB_MODEL_MADX_MASK, PATHFETCHER, AFSFETCHER, OPTICS_SUBDIR
from omc3.model.model_creators.lhc_model_creator import (  # noqa
    LhcBestKnowledgeCreator,
    LhcCouplingCreator,
    LhcModelCreator,
)
from omc3.model.model_creators.ps_model_creator import PsModelCreator
from omc3.model.model_creators.psbooster_model_creator import BoosterModelCreator
from omc3.model.model_creators.segment_creator import SegmentCreator
from omc3.utils.iotools import create_dirs, PathOrStr, save_config
from omc3.utils import logging_tools
from omc3.utils.parsertools import print_help, require_param
from omc3.model.model_creators import abstract_model_creator

LOGGER = logging_tools.get_logger(__name__)


CREATORS = {
    "lhc": {"nominal": LhcModelCreator,
            "best_knowledge": LhcBestKnowledgeCreator,
            "segment": SegmentCreator,
            "coupling_correction": LhcCouplingCreator},
    "psbooster": {"nominal": BoosterModelCreator,
                  "segment": SegmentCreator},
    "ps": {"nominal": PsModelCreator,
           "segment": SegmentCreator},
}


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="type",
        choices=("nominal", "best_knowledge", "coupling_correction"),
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

            choices: ``('nominal', 'best_knowledge', 'coupling_correction')``


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
        return None

    
    # proceed to the creator
    accel_inst = manager.get_accelerator(accel_opt)
    require_param("type", _get_params(), opt)

    LOGGER.debug(f"Accelerator Instance {accel_inst.NAME}, model type {opt.type}")

    creator: abstract_model_creator.ModelCreator = CREATORS[accel_inst.NAME][opt.type]

    # now that the creator is initialised, we can ask for modifiers that are actually present
    # using the fetcher we chose
    if not creator.check_options(accel_inst, opt):
        return None

    accel_inst.verify_object()
    require_param("outputdir", _get_params(), opt)

    # Prepare model-dir output directory
    accel_inst.model_dir = Path(opt.outputdir).absolute()

    # adjust modifier paths, to allow giving only filenames in default directories (e.g. optics)
    if accel_inst.modifiers is not None:
        accel_inst.modifiers = [_find_modifier(m, accel_inst) for m in accel_inst.modifiers]

    # Prepare paths
    create_dirs(opt.outputdir)
    creator.prepare_run(accel_inst)
    
    madx_script = creator.get_madx_script(accel_inst)
    # Run madx to create model
    run_string(madx_script,
               output_file=opt.outputdir / JOB_MODEL_MADX_MASK.format(opt.type),
               log_file=opt.logfile,
               cwd=opt.outputdir)
    
    # Save config at the end, to not being written out for each time the choices are listed
    save_config(Path(opt.outputdir), opt=opt, unknown_opt=accel_opt, script=__file__)
    
    # Return accelerator instance
    accel_inst.model_dir = opt.outputdir
    return accel_inst


def _find_modifier(modifier: Path, accel_inst: Accelerator):
    # first case: if modifier exists as is, take it
    if modifier.exists():
        return modifier

    # second case: try if it is already in the output dir
    model_dir_path: Path = accel_inst.model_dir / modifier
    if model_dir_path.exists():
        return model_dir_path.absolute()

    # and last case, try to find it in the acc-models rep
    if accel_inst.acc_model_path is not None:
        optics_path: Path = accel_inst.acc_model_path / OPTICS_SUBDIR / modifier
        if optics_path.exists():
            return optics_path.absolute()

    raise FileNotFoundError(f"couldn't find modifier {modifier}. Tried in {accel_inst.model_dir} and {accel_inst.acc_model_path}/{OPTICS_SUBDIR}")


if __name__ == "__main__":
    create_instance_and_model()
