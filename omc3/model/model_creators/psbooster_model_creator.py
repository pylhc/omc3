"""
PS Booster Model Creator
------------------------

This module provides convenience functions for model creation of the ``PSB``.
"""
import os
import shutil
from pathlib import Path

from omc3.model.accelerators.accelerator import AccExcitationMode, AcceleratorDefinitionError
from omc3.model.accelerators.psbooster import Psbooster
from omc3.model.constants import (ACCELERATOR_MODEL_REPOSITORY, AFSFETCHER,
                                  ERROR_DEFFS_TXT, PATHFETCHER)
from omc3.model.model_creators.abstract_model_creator import ModelCreator
from omc3.utils.parsertools import require_param
from omc3.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def _check_folder_choices(parent: Path, msg: str, selection: str, list_choices: bool=False) -> Path:
    choices = [d.name for d in parent.iterdir() if d.is_dir()]
    if selection is None or selection not in choices:
        if list_choices:
            for choice in choices:
                print(choice)
            return None
        raise AttributeError(f"{msg}.\nSelected: '{selection}'.\nChoices: [{', '.join(choices)}]")
    return parent / selection


class PsboosterModelCreator(ModelCreator):
    @classmethod
    def get_madx_script(cls, accel: Psbooster) -> str:
        madx_script = accel.get_base_madx_script()
        replace_dict = {
            "USE_ACD": str(int(accel.excitation == AccExcitationMode.ACD)),
            "RING": accel.ring,
            "DPP": accel.dpp,
            "OUTPUT": str(accel.model_dir),
        }
        madx_template = accel.get_file("twiss.mask").read_text()
        madx_script += madx_template % replace_dict
        return madx_script

    @classmethod
    def get_correction_check_script(cls, accel: Psbooster, corr_file: str, chrom: bool) -> str:
        raise NotImplementedError(
            "Correction check is not implemented for the PsBooster model creator yet. "
        )


    def get_options(self, accel_inst: Psbooster, opt) -> bool:
        if opt.fetch == PATHFETCHER:
            accel_inst.acc_model_path = Path(opt.path)
        elif opt.fetch == AFSFETCHER:
            accel_inst.acc_model_path = _check_folder_choices(ACCELERATOR_MODEL_REPOSITORY / "psb",
                                                              "No optics tag (flag --year) given",
                                                              accel_inst.year,
                                                              opt.list_opticsfiles)
        else:
            raise AttributeError("PSB model creation requires one of the following fetchers: "
                                 f"{PATHFETCHER}, {AFSFETCHER}]. "
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

        possible_beam_files = list(cycle_point_path.glob("*.beam"))
        if len(possible_beam_files) > 1:
            LOGGER.error("more than one beam file found in %s. Taking first one: %s",
                         cycle_point_path,
                         possible_beam_files[0])
        if len(possible_beam_files) == 0:
            raise AcceleratorDefinitionError(f"no beam file found in {cycle_point_path}")
        accel_inst.beam_file = possible_beam_files[0]

        possible_str_files = list(cycle_point_path.glob("*.str"))
        if len(possible_str_files) > 1:
            LOGGER.error("more than one str file found in %s. Taking first one: %s",
                         cycle_point_path,
                         possible_str_files[0])
        if len(possible_str_files) == 0:
            raise AcceleratorDefinitionError(f"no str file found in {cycle_point_path}")
        accel_inst.str_file = possible_str_files[0]

        return True


    @classmethod
    def prepare_run(cls, accel: Psbooster):
        shutil.copy(
            accel.get_file(f"error_deff_ring{accel.ring}.txt"), accel.model_dir / ERROR_DEFFS_TXT
        )
