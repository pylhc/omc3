"""
PETRA Model Creator
----------------

This module provides convenience functions for model creation of the ``PETRA``.
"""
import logging
import shutil

from omc3.model.accelerators.accelerator import AccExcitationMode
from omc3.model.accelerators.petra import Petra
from omc3.model.constants import ERROR_DEFFS_TXT, TWISS_AC_DAT, TWISS_DAT, TWISS_ELEMENTS_DAT, MACROS_DIR
from omc3.model.model_creators.abstract_model_creator import ModelCreator

from omc3.model.accelerators.petra import BETA_TO_SEQUENCE, MACROS_MADX
from omc3.utils import iotools

LOGGER = logging.getLogger(__name__)


class PetraModelCreator(ModelCreator):
    @classmethod
    def get_madx_script(cls, accel: Petra) -> str:
        madx_script = accel.get_base_madx_script()
        use_acd = "1" if (accel.excitation == AccExcitationMode.ACD) else "0"
        madx_script += (
            f"exec, do_twiss('{accel.model_dir / TWISS_DAT}');\n"
            f"exec, do_twiss_elements('{accel.model_dir / TWISS_ELEMENTS_DAT}');\n"
        )

        if accel.excitation != AccExcitationMode.FREE or accel.drv_tunes is not None:
            # allow user to modify script and enable excitation, if driven tunes are given
            madx_script += (
                f"use_acd={use_acd};\n"
                f"if(use_acd == 1){{\n"
                f"!  driven tunes\n"
                f"dQx = {accel.drv_tunes[0]};\n"
                f"dQy = {accel.drv_tunes[1]};\n"
                f"exec, install_adt_ac_dipole(Qx, Qy, dQx, dQy);\n"
                f"exec, do_twiss('{accel.model_dir / TWISS_AC_DAT}');\n"
                f"}}"
            )

        return madx_script

    @classmethod
    def get_correction_check_script(cls, accel: Petra, corr_file: str = "changeparameters_couple.madx",
                                    chrom: bool = False) -> str:
        madx_script = accel.get_base_madx_script()
        madx_script += (
            f"exec, do_twiss('{accel.model_dir / 'twiss_no.dat'!s}');\n"
            f"call, file = '{corr_file}';\n"
            f"exec, do_twiss('{accel.model_dir / 'twiss_cor.dat'!s}');\n"
        )
        if chrom:
            raise NotImplemented("Correction check is not implemented for the Petra model creator yet. ")
        return madx_script

    @classmethod
    def prepare_run(cls, accel: Petra) -> None:
        shutil.copy(accel.get_file(ERROR_DEFFS_TXT), accel.model_dir / ERROR_DEFFS_TXT)
        shutil.copy(accel.get_file(MACROS_MADX), accel.model_dir / MACROS_MADX)
        shutil.copy(accel.get_file(BETA_TO_SEQUENCE[accel.beta]), accel.model_dir / BETA_TO_SEQUENCE[accel.beta])