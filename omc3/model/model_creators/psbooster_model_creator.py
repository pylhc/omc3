"""
PS Booster Model Creator
------------------------

This module provides convenience functions for model creation of the ``PSB``.
"""
import shutil

from omc3.model.accelerators.accelerator import AccExcitationMode
from omc3.model.accelerators.psbooster import PsBase
from omc3.model.model_creators.ps_base_model_creator import PsBaseModelCreator
from omc3.model.constants import ERROR_DEFFS_TXT

class BoosterModelCreator(PsBaseModelCreator):
    acc_model_name = "psb"

    @classmethod
    def get_madx_script(cls, accel: PsBase) -> str:
        madx_script = accel.get_base_madx_script()
        replace_dict = {
            "USE_ACD": str(int(accel.excitation == AccExcitationMode.ACD)),
            "DPP": accel.dpp,
            "OUTPUT": str(accel.model_dir),
            "RING": accel.ring,
        }
        madx_template = accel.get_file("twiss.mask").read_text()
        madx_script += madx_template % replace_dict
        return madx_script

    @classmethod
    def prepare_run(cls, accel: PsBase):
        shutil.copy(
            accel.get_file(f"error_deff_ring{accel.ring}.txt"), accel.model_dir / ERROR_DEFFS_TXT
        )
