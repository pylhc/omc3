"""
PS Booster Model Creator
------------------------

This module provides convenience functions for model creation of the ``PSB``.
"""
import shutil

from omc3.model.accelerators.accelerator import AccExcitationMode
from omc3.model.accelerators.psbooster import Psbooster
from omc3.model.constants import ERROR_DEFFS_TXT
from omc3.model.model_creators.abstract_model_creator import ModelCreator


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
        raise NotImplemented(
            "Correction check is not implemented for the PsBooster model creator yet. "
        )

    @classmethod
    def prepare_run(cls, accel: Psbooster):
        shutil.copy(
            accel.get_file(f"error_deff_ring{accel.ring}.txt"), accel.model_dir / ERROR_DEFFS_TXT
        )
