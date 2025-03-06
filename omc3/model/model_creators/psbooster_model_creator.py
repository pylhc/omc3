"""
PS Booster Model Creator
------------------------

This module provides convenience functions for model creation of the ``PSB``.
"""
import shutil

from omc3.model.accelerators.accelerator import AccExcitationMode
from omc3.model.accelerators.psbooster import Psbooster
from omc3.model.model_creators.ps_base_model_creator import PsBaseModelCreator
from omc3.model.constants import ERROR_DEFFS_TXT

class PsboosterModelCreator(PsBaseModelCreator):
    acc_model_name = "psb"

    def get_madx_script(self) -> str:
        accel: Psbooster = self.accel
        madx_script = self.get_base_madx_script()
        replace_dict = {
            "USE_ACD": str(int(accel.excitation == AccExcitationMode.ACD)),
            "RING": accel.ring,
            "DPP": accel.dpp,
            "OUTPUT": str(accel.model_dir),
        }
        madx_template = accel.get_file("twiss.mask").read_text()
        madx_script += madx_template % replace_dict
        return madx_script

    def get_base_madx_script(self):
        accel: Psbooster = self.accel
        use_acd = accel.excitation == AccExcitationMode.ACD
        replace_dict = {
            "FILES_DIR": str(accel.get_dir()),
            "USE_ACD": str(int(use_acd)),
            "RING": str(accel.ring),
            "NAT_TUNE_X": accel.nat_tunes[0],
            "NAT_TUNE_Y": accel.nat_tunes[1],
            "KINETICENERGY": 0 if accel.energy is None else accel.energy,
            "USE_CUSTOM_PC": "0" if accel.energy is None else "1",
            "ACC_MODELS_DIR": accel.acc_model_path,
            "BEAM_FILE": accel.beam_file,
            "STR_FILE": accel.str_file,
            "DRV_TUNE_X": "",
            "DRV_TUNE_Y": "",
        }
        if use_acd:
            replace_dict["DRV_TUNE_X"] = accel.drv_tunes[0]
            replace_dict["DRV_TUNE_Y"] = accel.drv_tunes[1]
        mask = accel.get_file('base.mask').read_text()
        return mask % replace_dict

    def prepare_run(self):
        super().prepare_run()
        shutil.copy(
            self.accel.get_file(f"error_deff_ring{self.accel.ring}.txt"), self.accel.model_dir / ERROR_DEFFS_TXT
        )
