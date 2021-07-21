"""
PS Booster Model Creator
------------------------

This module provides convenience functions for model creation of the ``PSB``.
"""
import shutil

from omc3.model.accelerators.accelerator import AccExcitationMode
from omc3.model.constants import ERROR_DEFFS_TXT
from pathlib import Path


class PsboosterModelCreator(object):

    @classmethod
    def get_madx_script(cls, accel, opt):
        output_path = opt.outputdir
        madx_script = accel.get_base_madx_script(output_path)
        replace_dict = {
            "USE_ACD": str(int(accel.excitation == AccExcitationMode.ACD)),
            "RING": accel.ring,
            "DPP": accel.dpp,
            "OUTPUT": str(output_path),
        }
        madx_template = accel.get_file("twiss.mask").read_text()
        madx_script += madx_template % replace_dict
        return madx_script

    # TODO: Remove when Response Creation implemented (just here for reference) jdilly, 2021
    # @classmethod
    # def _prepare_fullresponse(cls, instance, output_path):
    #     with open(instance.get_file("template.iterate.madx")) as textfile:
    #         iterate_template = textfile.read()
    #
    #     replace_dict = {
    #         "FILES_DIR": instance.get_dir(),
    #         "RING": instance.ring,
    #         "OPTICS_PATH": instance.modifiers,
    #         "PATH": output_path,
    #         "KINETICENERGY": instance.energy,
    #         "NAT_TUNE_X": instance.nat_tunes[0],
    #         "NAT_TUNE_Y": instance.nat_tunes[1],
    #         "DRV_TUNE_X": "",
    #         "DRV_TUNE_Y": "",
    #         "DPP": instance.dpp,
    #         "OUTPUT": output_path,
    #     }
    #
    #     with open(os.path.join(output_path, JOB_ITERATE_MADX), "w") as textfile:
    #         textfile.write(iterate_template % replace_dict)
    #
    # @classmethod
    # def _prepare_corrtest(cls, instance, output_path):
    #     """ Partially fills mask file for tests of corrections
    #         Reads correction_test.madx (defined in psbooster.get_corrtest_tmpl())
    #         and produces correction_test.mask2.madx.
    #         Java GUI fills the remaining fields
    #        """
    #     with open(instance.get_file("correction_test.madx")) as textfile:
    #         template = textfile.read()
    #
    #     replace_dict = {
    #         "KINETICENERGY": instance.energy,
    #         "FILES_DIR": instance.get_dir(),
    #         "RING": instance.ring,
    #         "NAT_TUNE_X": instance.nat_tunes[0],
    #         "NAT_TUNE_Y": instance.nat_tunes[1],
    #         "DPP": instance.dpp,
    #         "PATH": "%TESTPATH",  # field filled later by Java GUI
    #         "COR": "%COR"  # field filled later by Java GUI
    #     }
    #
    #     with open(os.path.join(output_path, "correction_test.mask2.madx"), "w") as textfile:
    #         textfile.write(template % replace_dict)

    @classmethod
    def prepare_run(cls, instance, output_path):
        shutil.copy(
            instance.get_file(f"error_deff_ring{instance.ring}.txt"),
            output_path / ERROR_DEFFS_TXT
        )
