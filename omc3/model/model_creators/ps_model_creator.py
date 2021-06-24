"""
PS Model Creator
----------------

This module provides convenience functions for model creation of the ``PS``.
"""
import logging
import os
import shutil
from pathlib import Path

from omc3.model.accelerators.accelerator import AccExcitationMode
from omc3.model.constants import ERROR_DEFFS_TXT

LOGGER = logging.getLogger(__name__)


class PsModelCreator(object):

    @classmethod
    def get_madx_script(cls, accel, output_path: Path):
        madx_script = accel.get_base_madx_script(output_path)
        replace_dict = {
            "USE_ACD": str(int(accel.excitation == AccExcitationMode.ACD)),
            "DPP": accel.dpp,
            "OUTPUT": str(output_path),
        }
        madx_template = accel.get_file("twiss.mask").read_text()
        madx_script += madx_template % replace_dict
        return madx_script

    # TODO: Remove when Response Creation implemented (just here for reference) jdilly, 2021
    # @classmethod
    # def _prepare_fullresponse(cls, instance, output_path):
    #     iterate_template = instance.get_file("template.iterate.madx").read_text()
    #     replace_dict = {
    #         "FILES_DIR": str(instance.get_dir()),
    #         "OPTICS_PATH": instance.modifiers,
    #         "PATH": output_path,
    #         "KINETICENERGY": instance.energy,
    #         "NAT_TUNE_X": instance.nat_tunes[0],
    #         "NAT_TUNE_Y": instance.nat_tunes[1],
    #         "DRV_TUNE_X": "",
    #         "DRV_TUNE_Y": "",
    #     }
    #     output_file = output_path / JOB_ITERATE_MADX
    #     output_file.write_text(iterate_template % replace_dict)

    @classmethod
    def prepare_run(cls, instance, output_path):
        # get path of file from PS model directory (without year at the end)
        shutil.copy(
            instance.get_file("error_deff.txt"),
            output_path / ERROR_DEFFS_TXT
        )



