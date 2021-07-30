"""
PS Model Creator
----------------

This module provides convenience functions for model creation of the ``PS``.
"""
import logging
import shutil

from omc3.model.accelerators.accelerator import AccExcitationMode
from omc3.model.constants import ERROR_DEFFS_TXT
from omc3.model.model_creators.abstract_model_creator import ModelCreator

LOGGER = logging.getLogger(__name__)


class PsModelCreator(ModelCreator):

    def get_madx_script(self) -> str:
        madx_script = self.accel.get_base_madx_script()
        replace_dict = {
            "USE_ACD": str(int(self.accel.excitation == AccExcitationMode.ACD)),
            "DPP": self.accel.dpp,
            "OUTPUT": str(self.accel.model_dir),
        }
        madx_template = self.accel.get_file("twiss.mask").read_text()
        madx_script += madx_template % replace_dict
        return madx_script

    def prepare_run(self) -> None:
        # get path of file from PS model directory (without year at the end)
        shutil.copy(self.accel.get_file("error_deff.txt"), self.accel.model_dir / ERROR_DEFFS_TXT)
