"""
PS Model Creator
----------------

This module provides convenience functions for model creation of the ``PS``.
"""
import logging
import shutil

from omc3.model.accelerators.accelerator import AccExcitationMode
from omc3.model.accelerators.ps import Ps
from omc3.model.constants import ERROR_DEFFS_TXT, MODIFIER_TAG
from omc3.model.model_creators.ps_base_model_creator import PsBaseModelCreator

LOGGER = logging.getLogger(__name__)


class PsModelCreator(PsBaseModelCreator):
    acc_model_name = "ps"

    def get_madx_script(self) -> str:
        accel: Ps = self.accel
        if (accel.energy is None):
            raise RuntimeError("PS model creation currently relies on manual Energy management. Please provide the --energy ENERGY flag")
        madx_script = self.get_base_madx_script()
        replace_dict = {
            "USE_ACD": str(int(accel.excitation == AccExcitationMode.ACD)),
            "DPP": accel.dpp,
            "OUTPUT": str(accel.model_dir),
        }
        madx_template = accel.get_file("twiss.mask").read_text()
        madx_script += madx_template % replace_dict
        return madx_script
    
    def get_base_madx_script(self):
        accel: Ps = self.accel
        use_acd = accel.excitation == AccExcitationMode.ACD
        replace_dict = {
            "FILES_DIR": str(accel.get_dir()),
            "USE_ACD": str(int(use_acd)),
            "NAT_TUNE_X": accel.nat_tunes[0],
            "NAT_TUNE_Y": accel.nat_tunes[1],
            "KINETICENERGY": 0 if accel.energy is None else accel.energy,
            "USE_CUSTOM_PC": "0" if accel.energy is None else "1",
            "ACC_MODELS_DIR": accel.acc_model_path,
            "BEAM_FILE": accel.beam_file,
            "STR_FILE": accel.str_file,
            "DRV_TUNE_X": "0",
            "DRV_TUNE_Y": "0",
            "MODIFIERS": "",
            "USE_MACROS": "0" if accel.year == "2018" else "1",  # 2018 doesn't provide a macros file
            "PS_TUNE_METHOD": accel.tune_method,
        }
        if accel.modifiers:
            replace_dict["MODIFIERS"] = '\n'.join([f" call, file = '{m}'; {MODIFIER_TAG}" for m in accel.modifiers])
        if use_acd:
            replace_dict["DRV_TUNE_X"] = accel.drv_tunes[0]
            replace_dict["DRV_TUNE_Y"] = accel.drv_tunes[1]
        mask = accel.get_file('base.madx').read_text()
        return mask % replace_dict

    def prepare_run(self) -> None:
        super().prepare_run()
        # get path of file from PS model directory (without year at the end)
        shutil.copy(self.accel.get_file("error_deff.txt"), self.accel.model_dir / ERROR_DEFFS_TXT)
