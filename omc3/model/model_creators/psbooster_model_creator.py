"""
PS Booster Model Creator
------------------------

This module provides convenience functions for model creation of the ``PSB``.
"""
import shutil

from omc3.model.accelerators.accelerator import AccExcitationMode
from omc3.model.model_creators.ps_base_model_creator import PsBaseModelCreator
from omc3.model.constants import ERROR_DEFFS_TXT

class BoosterModelCreator(PsBaseModelCreator):
    acc_model_name = "psb"

    def get_madx_script(self) -> str:
        madx_script = self.accel.get_base_madx_script()
        replace_dict = {
            "USE_ACD": str(int(self.accel.excitation == AccExcitationMode.ACD)),
            "RING": self.accel.ring,
            "DPP": self.accel.dpp,
            "OUTPUT": str(self.accel.model_dir),
        }
        madx_template = self.accel.get_file("twiss.mask").read_text()
        madx_script += madx_template % replace_dict
        return madx_script

    def prepare_run(self):
        shutil.copy(
            self.accel.get_file(f"error_deff_ring{self.accel.ring}.txt"), self.accel.model_dir / ERROR_DEFFS_TXT
        )
