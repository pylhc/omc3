from abc import ABC
from pathlib import Path

from omc3.model.accelerators.accelerator import AcceleratorDefinitionError
from omc3.model.accelerators.psbase import PsBase
from omc3.model.constants import AFS_ACCELERATOR_MODEL_REPOSITORY, Fetcher
from omc3.model.model_creators.abstract_model_creator import ModelCreator, check_folder_choices
from omc3.utils import logging_tools
from omc3.utils.iotools import get_check_suffix_func

LOGGER = logging_tools.get_logger(__name__)


class PsBaseModelCreator(ModelCreator, ABC):
    acc_model_name = None

    def prepare_options(self, opt) -> bool:
        """ Use the fetcher to list choices if requested. """
        accel: PsBase = self.accel
        
        if opt.fetch == Fetcher.PATH:
            if opt.path is None:
                raise AcceleratorDefinitionError(
                    "Path fetcher chosen, but no path proivided."
                )
            acc_model_path = Path(opt.path)

        elif opt.fetch == Fetcher.AFS:
            # list 'year' choices ---
            acc_model_path = check_folder_choices(
                AFS_ACCELERATOR_MODEL_REPOSITORY / self.acc_model_name,
                msg="No optics tag (flag --year) given",
                selection=accel.year,
                list_choices=opt.list_choices,
                predicate=Path.is_dir
            )  # raises AcceleratorDefintionError if not valid choice
        else:
            LOGGER.warning(
                f"{accel.NAME} model creation is expected to run via a fetcher these days. "
                "If you are creating an older model, this might all be correct "
                "and you can ignore this warning. Otherwise you will soon run into "
                "a MAD-X error. In this case, please provide a fetcher for the model via --fetch flag. "
            )
            return

        scenario_path = check_folder_choices(
            acc_model_path / "scenarios",
            msg="No/Unknown scenario (flag --scenario) selected",
            selection=accel.scenario,
            list_choices=opt.list_choices
        )

        cycle_point_path = check_folder_choices(
            scenario_path,
            msg="No/Unknown cycle_point (flag --cycle_point) selected",
            selection=accel.cycle_point,
            list_choices=opt.list_choices
        )

        str_file = check_folder_choices(
            cycle_point_path,
            msg="No/Unknown strength file (flag --str_file) selected",
            selection=accel.str_file,
            list_choices=opt.list_choices,
            predicate=get_check_suffix_func(".str")
        )

        possible_beam_files = list(cycle_point_path.glob("*.beam"))

        # if no `.beam` file is found, try any madx job file, maybe we get lucky there
        if not len(possible_beam_files):
            possible_beam_files = list(cycle_point_path.glob("*.*job"))
        
            if not len(possible_beam_files):
                raise AcceleratorDefinitionError(f"No beam file found in {cycle_point_path}")

        if len(possible_beam_files) > 1:
            LOGGER.error(f"More than one beam file found in {cycle_point_path}. "
                         f"Taking first one: {possible_beam_files[0]}")

        beam_file = possible_beam_files[0]

        if opt.list_choices:
            with open(beam_file) as beamf:
                print(beamf.read())
            raise AcceleratorDefinitionError()  # not really an error, just indicates to stop
        
        # Set the found paths ---
        accel.acc_model_path = acc_model_path
        accel.str_file = str_file
        accel.beam_file = beam_file

