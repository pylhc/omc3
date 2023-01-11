"""
PS Model Creator
----------------

This module provides convenience functions for model creation of the ``PS``.
"""
import logging
import shutil

from omc3.model.accelerators.accelerator import AccExcitationMode
from omc3.model.accelerators.ps import Ps
from omc3.model.constants import ERROR_DEFFS_TXT
from omc3.model.model_creators.abstract_model_creator import ModelCreator
from omc3.model.accelerators.psbooster import PsBase
from omc3.model.model_creators.psbooster_model_creator import PsOrBoosterModelCreator

LOGGER = logging.getLogger(__name__)


class PsModelCreator(PsOrBoosterModelCreator):
    acc_model_name = "ps"

    @classmethod
    def get_madx_script(cls, accel: PsBase) -> str:
        if (accel.energy is None):
            raise RuntimeError("PS model creation currently relies on manual Energy management. Please provide the --energy ENERGY flag")
        madx_script = accel.get_base_madx_script()
        replace_dict = {
            "USE_ACD": str(int(accel.excitation == AccExcitationMode.ACD)),
            "DPP": accel.dpp,
            "OUTPUT": str(accel.model_dir),
        }
        madx_template = accel.get_file("twiss.mask").read_text()
        print(madx_template)
        madx_script += madx_template % replace_dict
        return madx_script
