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
from omc3.model.constants import JOB_MODEL_MADX, PATHFETCHER, AFSFETCHER, GITFETCHER, LSAFETCHER
from omc3.model.model_creators.lhc_model_creator import (  # noqa
    LhcBestKnowledgeCreator,
    LhcCouplingCreator,
    LhcModelCreator,
)
from omc3.model.model_creators.ps_model_creator import PsModelCreator
from omc3.model.model_creators.psbooster_model_creator import PsboosterModelCreator
from omc3.model.model_creators.segment_creator import SegmentCreator
from omc3.utils.iotools import create_dirs
from omc3.utils import logging_tools
from omc3.utils.parsertools import print_help

LOG = logging_tools.get_logger(__name__)

DRY_RUN = "*** ==> dry-run, no model created ***"

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
    params.add_parameter(
        name="type",
        choices=("nominal", "best_knowledge", "coupling_correction"),
        help="Type of model to create. [Required]",
    )
    params.add_parameter(
        name="outputdir",
        type=Path,
        help="Output path for model, twiss files will be writen here. [Required]"
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
        type=str,
        help=("If path fetcher is selected, this option sets the path"),
    )
    params.add_parameter(
        name="list_modifiers",
        action="store_true",
        help="if selected, a list of valid modifier files is printed",
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
    # Prepare paths
    print("model creator")
    create_dirs(opt.outputdir)

    # because the different modules (model_creator, accelerator_class, accelerator_modelcreator)
    # eagerly evaluate their state, we cannot predetermine any help to print.
    # the solution (workaround?) is to construct until failure, all the modules encountered will do
    # as much work as possible and print help / errors if appropriate
    created_model = manager.get_accelerator(accel_opt)

    # maybe a bit clunky
    # if `accel_inst` is None, the manager couldn't create it, but did handle the input
    # (either it crashed with a meaningful error message or help was requested and it printed that)
    if created_model.accelerator is None:
        print(DRY_RUN)
        return None
    # if `accel_inst` is not None AND help was requested, we print the help of **the modelcreator**
    if created_model.help_requested:
        print("Model Creators help requested")
        print_help(_get_params())
        print(DRY_RUN)
        return None

    accel_inst = created_model.accelerator

    # if none of the above are true, the instance was successfully created
    # proceed to the creator
    print(f"Accelerator Instance {accel_inst.NAME}, model type {opt.type}")
    creator = CREATORS[accel_inst.NAME][opt.type]

    if not created_model.help_requested and creator.get_opt(accel_inst, opt):
        accel_inst.verify_object()
        # Prepare model-dir output directory
        accel_inst.model_dir = opt.outputdir
        creator.prepare_run(accel_inst)

        # get madx-script with relative output-paths
        # as `cwd` changes run to correct directory.
        # The resulting model-dir is then more self-contained. (jdilly)
        accel_inst.model_dir = Path()
        madx_script = creator.get_madx_script(accel_inst)

        # Run madx to create model
        run_string(madx_script,
                   output_file=opt.outputdir / JOB_MODEL_MADX,
                   log_file=opt.logfile,
                   cwd=opt.outputdir)
        # Return accelerator instance
        accel_inst.model_dir = opt.outputdir
        return accel_inst

    print(DRY_RUN)


if __name__ == "__main__":
    create_instance_and_model()
