#!/usr/bin/env python3
#
import os
from pathlib import Path

from omc3.model.accelerators.accelerator import AcceleratorDefinitionError
from omc3.model.accelerators.psbooster import Psbooster
from omc3.model.constants import (ACCELERATOR_MODEL_REPOSITORY, AFSFETCHER,
                                  PATHFETCHER)
from omc3.model.model_creators.abstract_model_creator import ModelCreator
from omc3.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)

def always_true(_: Path) -> bool:
    return True


def is_str_file(path: Path) -> bool:
    return path.suffix == ".str"


def _check_folder_choices(parent: Path, msg: str,
                          selection: str,
                          list_choices: bool = False,
                          predicate=always_true) -> Path:
    choices = [d.name for d in parent.iterdir() if predicate(d)]
    if selection is None or selection not in choices:
        if list_choices:
            for choice in choices:
                print(choice)
            return None
        raise AttributeError(f"{msg}.\nSelected: '{selection}'.\nChoices: [{', '.join(choices)}]")
    return parent / selection



class PsBaseModelCreator(ModelCreator):
    acc_model_name = None


    @classmethod
    def get_correction_check_script(cls, accel: Psbooster, corr_file: str, chrom: bool) -> str:
        raise NotImplementedError(
            "Correction check is not implemented for the PsBooster model creator yet. "
        )

    @classmethod
    def get_options(cls, accel_inst, opt) -> bool:
        if opt.fetch == PATHFETCHER:
            accel_inst.acc_model_path = Path(opt.path)
        elif opt.fetch == AFSFETCHER:
            accel_inst.acc_model_path = _check_folder_choices(ACCELERATOR_MODEL_REPOSITORY / cls.acc_model_name,
                                                              "No optics tag (flag --year) given",
                                                              accel_inst.year,
                                                              opt.list_opticsfiles)
        else:
            raise AttributeError(f"{accel_inst.NAME} model creation requires one of the following fetchers: "
                                 f"[{PATHFETCHER}, {AFSFETCHER}]. "
                                 "Please provide one with the flag `--fetch afs` "
                                 "or `--fetch path --path PATH`.")
        if accel_inst.acc_model_path is None:
            return False

        scenario_path = _check_folder_choices(accel_inst.acc_model_path / "scenarios",
                                              "No scenario (flag --scenario) selected",
                                              accel_inst.scenario,
                                              opt.list_opticsfiles)
        if scenario_path is None:
            return False

        cycle_point_path = _check_folder_choices(scenario_path,
                                                 "No cycle_point (flag --cycle_point) selected",
                                                 accel_inst.cycle_point,
                                                 opt.list_opticsfiles)
        if cycle_point_path is None:
            return False

        str_file = _check_folder_choices(cycle_point_path,
                                         "No strength file (flag --str_file) selected",
                                         accel_inst.str_file,
                                         opt.list_opticsfiles,
                                         is_str_file
                                         )
        if str_file is None:
            return False


        accel_inst.str_file = str_file
        possible_beam_files = list(cycle_point_path.glob("*.beam"))
        # if now `.beam` file is found, try any madx job file, maybe we get lucky there
        if len(possible_beam_files) == 0:
            possible_beam_files = list(cycle_point_path.glob("*.*job"))

        if len(possible_beam_files) > 1:
            LOGGER.error("more than one beam file found in %s. Taking first one: %s",
                         cycle_point_path,
                         possible_beam_files[0])
        if len(possible_beam_files) == 0:
            raise AcceleratorDefinitionError(f"no beam file found in {cycle_point_path}")
        accel_inst.beam_file = possible_beam_files[0]
        if accel_inst.beam_file and opt.list_opticsfiles:
            with open(accel_inst.beam_file) as beamf:
                print(beamf.read())
            return False

        return True
