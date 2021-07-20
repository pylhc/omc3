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
from omc3.model.constants import JOB_MODEL_MADX
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

DRY_RUN = "*** dry-run, no model created ***"

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
        help="Type of model to create.",
        required=True,
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
    params.add_parameter(
        name="fetch",
        type=str,
        help="select the fetcher which sets up the lattice definition (madx, seq, strength files)",
        choices=["path", "afs"]  # ["path", "afs", "git", "lsa"]
    )
    params.add_parameter(
        name="list-modifiers",
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

    create_dirs(opt.outputdir)

    accel_inst, help_requested = manager.get_accelerator(accel_opt)

    if accel_inst is None:
        print(DRY_RUN)
        return
    if help_requested:
        print_help(_get_params())
        print(DRY_RUN)
        return

    print(f"Accelerator Instance {accel_inst.NAME}, model type {opt.type}")
    creator = CREATORS[accel_inst.NAME][opt.type]

    if not help_requested and creator.get_opt(opt):
        accel_inst.verify_object()
        # Prepare model-dir output directory
        accel_inst.model_dir = opt.outputdir
        creator.prepare_run(accel_inst, opt.outputdir)

        # get madx-script with relative output-paths
        # as `cwd` changes run to correct directory.
        # The resulting model-dir is then more self-contained. (jdilly)
        accel_inst.model_dir = Path()
        madx_script = creator.get_madx_script(accel_inst)
        run_string(madx_script,
                output_file=opt.outputdir / JOB_MODEL_MADX,
                log_file=opt.logfile)

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
