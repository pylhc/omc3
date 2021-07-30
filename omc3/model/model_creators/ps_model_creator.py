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

LOGGER = logging.getLogger(__name__)


class PsModelCreator(ModelCreator):
    @classmethod
    def get_madx_script(cls, accel: Ps) -> str:
        madx_script = accel.get_base_madx_script()
        replace_dict = {
            "USE_ACD": str(int(accel.excitation == AccExcitationMode.ACD)),
            "DPP": accel.dpp,
            "OUTPUT": str(accel.model_dir),
        }
        madx_template = accel.get_file("twiss.mask").read_text()
        madx_script += madx_template % replace_dict
        return madx_script

    @classmethod
    def get_correction_check_script(cls, accel: Ps, corr_file: str, chrom: bool) -> str:
        raise NotImplemented("Correction check is not implemented for the Ps model creator yet. ")

    @classmethod
    def prepare_run(cls, accel: Ps) -> None:
        # get path of file from PS model directory (without year at the end)
        shutil.copy(accel.get_file("error_deff.txt"), accel.model_dir / ERROR_DEFFS_TXT)
