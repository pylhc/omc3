from abc import ABC
from pathlib import Path

from omc3.model.accelerators.accelerator import AcceleratorDefinitionError
from omc3.model.accelerators.psbooster import Psbooster
from omc3.model.constants import (AFS_ACCELERATOR_MODEL_REPOSITORY, AFSFETCHER,
                                  PATHFETCHER)
from omc3.model.model_creators.abstract_model_creator import ModelCreator, check_folder_choices
from omc3.utils import logging_tools
from omc3.utils.iotools import get_check_suffix_func

LOGGER = logging_tools.get_logger(__name__)


class PsBaseModelCreator(ModelCreator, ABC):
    acc_model_name = None

    @classmethod
    def get_correction_check_script(cls, accel: Psbooster, corr_file: str, chrom: bool) -> str:
        raise NotImplementedError(
            "Correction check is not implemented for the PsBooster model creator yet. "
        )

    def check_options(self, opt) -> bool:
        """ Use the fetcher to list choices if requested. """
        accel = self.accel

        if opt.fetch == PATHFETCHER:
            accel.acc_model_path = Path(opt.path)

        elif opt.fetch == AFSFETCHER:
            accel.acc_model_path = check_folder_choices(
                AFS_ACCELERATOR_MODEL_REPOSITORY / self.acc_model_name,
                msg="No optics tag (flag --year) given",
                selection=accel.year,
                list_choices=opt.list_choices
            )
        else:
            raise AttributeError(
                f"{accel.NAME} model creation requires one of the following fetchers: "
                f"[{PATHFETCHER}, {AFSFETCHER}]. "
                "Please provide one with the flag `--fetch afs` "
                "or `--fetch path --path PATH`."
            )

        if accel.acc_model_path is None:
            return False

        scenario_path = check_folder_choices(
            accel.acc_model_path / "scenarios",
            msg="No scenario (flag --scenario) selected",
            selection=accel.scenario,
            list_choices=opt.list_choices
        )
        if scenario_path is None:
            return False

        cycle_point_path = check_folder_choices(
            scenario_path,
            msg="No cycle_point (flag --cycle_point) selected",
            selection=accel.cycle_point,
            list_choices=opt.list_choices
        )
        if cycle_point_path is None:
            return False

        str_file = check_folder_choices(
            cycle_point_path,
            msg="No strength file (flag --str_file) selected",
            selection=accel.str_file,
            list_choices=opt.list_choices,
            predicate=get_check_suffix_func(".str")
        )
        if str_file is None:
            return False

        accel.str_file = str_file
        possible_beam_files = list(cycle_point_path.glob("*.beam"))

        # if now `.beam` file is found, try any madx job file, maybe we get lucky there
        if not len(possible_beam_files):
            possible_beam_files = list(cycle_point_path.glob("*.*job"))
        
            if not len(possible_beam_files):
                raise AcceleratorDefinitionError(f"no beam file found in {cycle_point_path}")

        if len(possible_beam_files) > 1:
            LOGGER.error(f"More than one beam file found in {cycle_point_path}. "
                         f"Taking first one: {possible_beam_files[0]}")

        accel.beam_file = possible_beam_files[0]

        if opt.list_choices:
            with open(accel.beam_file) as beamf:
                print(beamf.read())
            return False

        return True
