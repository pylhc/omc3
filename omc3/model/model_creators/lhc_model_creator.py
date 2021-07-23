"""
LHC Model Creator
-----------------

This module provides convenience functions for model creation of the ``LHC``.
"""
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tfs

from omc3.model.accelerators.accelerator import AccExcitationMode, AcceleratorDefinitionError, Accelerator
from omc3.model.accelerators.lhc import Lhc
from omc3.model.constants import (B2_ERRORS_TFS, B2_SETTINGS_MADX,
                                  ERROR_DEFFS_TXT, GENERAL_MACROS,
                                  LHC_MACROS, MACROS_DIR,
                                  TWISS_AC_DAT, TWISS_ADT_DAT,
                                  TWISS_BEST_KNOWLEDGE_DAT, TWISS_DAT,
                                  TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT,
                                  TWISS_ELEMENTS_DAT)
from omc3.model.model_creators.abstract_model_creator import ModelCreator
from omc3.utils import iotools

LOGGER = logging.getLogger(__name__)


def _b2_columns():
    cols_outer = [f"{KP}{num}{S}L" for KP in ("K", "P") for num in range(21) for S in ("", "S")]
    cols_middle = ["DX", "DY", "DS", "DPHI", "DTHETA", "DPSI", "MREX", "MREY", "MREDX", "MREDY",
                   "AREX", "AREY", "MSCALX", "MSCALY", "RFM_FREQ", "RFM_HARMON", "RFM_LAG"]
    return cols_outer[:42] + cols_middle + cols_outer[42:]


class LhcModelCreator(ModelCreator):

    @classmethod
    def get_madx_script(cls, accel: Lhc) -> str:  # nominal
        use_acd = "1" if (accel.excitation == AccExcitationMode.ACD) else "0"
        use_adt = "1" if (accel.excitation == AccExcitationMode.ADT) else "0"
        madx_script = accel.get_base_madx_script()
        madx_script +=(
            f"exec, do_twiss_monitors(LHCB{accel.beam}, '{accel.model_dir / TWISS_DAT}', {accel.dpp});\n"
            f"exec, do_twiss_elements(LHCB{accel.beam}, '{accel.model_dir / TWISS_ELEMENTS_DAT}', {accel.dpp});\n"
        )
        if accel.excitation != AccExcitationMode.FREE or accel.drv_tunes is not None:
            # allow user to modify script and enable excitation, if driven tunes are given
            madx_script +=(
                f"use_acd={use_acd};\nuse_adt={use_adt};\n"
                f"if(use_acd == 1){{\n"
                f"exec, twiss_ac_dipole({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.drv_tunes[0]}, {accel.drv_tunes[1]}, {accel.beam}, '{accel.model_dir / TWISS_AC_DAT}', {accel.dpp});\n"
                f"}}else if(use_adt == 1){{\n"
                f"exec, twiss_adt({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.drv_tunes[0]}, {accel.drv_tunes[1]}, {accel.beam}, '{accel.model_dir / TWISS_ADT_DAT}', {accel.dpp});\n"
                f"}}\n"
            )
        return madx_script

    @classmethod
    def get_correction_check_script(cls, accel: Lhc, corr_file: str = "changeparameters_couple.madx", chrom: bool = False) -> str:
        madx_script = accel.get_base_madx_script()
        madx_script += (
            f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{accel.model_dir / 'twiss_no.dat'!s}', 0.0);\n"
            f"call, file = '{corr_file}';\n"
            f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{accel.model_dir / 'twiss_cor.dat'!s}', 0.0);\n")
        if chrom:
            madx_script +=(
                f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{accel.model_dir / 'twiss_cor_dpm.dat'}', %DELTAPM);\n"
                f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{accel.model_dir / 'twiss_cor_dpp.dat'}', %DELTAPP);\n"
            )
        return madx_script

    @classmethod
    def prepare_run(cls, accel: Lhc):
        cls.check_accelerator_instance(accel)
        macros_path = accel.model_dir / MACROS_DIR
        iotools.create_dirs(macros_path)
        lib_path = Path(__file__).parent.parent / "madx_macros"
        shutil.copy(lib_path / GENERAL_MACROS, macros_path / GENERAL_MACROS)
        shutil.copy(lib_path / LHC_MACROS, macros_path / LHC_MACROS)
        if accel.energy is not None:
            core = f"{int(accel.energy * 1000):04d}"
            error_dir_path = accel.get_lhc_error_dir()
            shutil.copy(error_dir_path / f"{core}GeV.tfs", accel.model_dir / ERROR_DEFFS_TXT)
            shutil.copy(error_dir_path / "b2_errors_settings" / f"beam{accel.beam}_{core}GeV.madx",
                        accel.model_dir / B2_SETTINGS_MADX)
            b2_table = tfs.read(error_dir_path / f"b2_errors_beam{accel.beam}.tfs", index="NAME")
            gen_df = pd.DataFrame(data=np.zeros((b2_table.index.size, len(_b2_columns()))),
                                  index=b2_table.index, columns=_b2_columns())
            gen_df["K1L"] = b2_table.loc[:, f"K1L_{core}"].to_numpy()
            tfs.write(accel.model_dir / B2_ERRORS_TFS, gen_df,
                      headers_dict={"NAME": "EFIELD", "TYPE": "EFIELD"}, save_index="NAME")

    @staticmethod
    def check_accelerator_instance(accel: Lhc):
        accel.verify_object()  # should have been done anyway, but cannot hurt (jdilly)

        # Creator specific checks
        if accel.model_dir is None:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete: "
                "Model directory (the output directory) is not given."
            )

        if accel.modifiers is None or not len(accel.modifiers):
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete: "
                "No modifiers could be found."
            )

        # hint, if modifiers are given as absolute paths: `path / abs_path` returns `abs_path`  (jdilly)
        inexistent_modifiers = [m for m in accel.modifiers if not (accel.model_dir / m).exists()]
        if len(inexistent_modifiers):
            raise AcceleratorDefinitionError(
                "The following modifier files do not exist: "
                f"{', '.join([str(accel.model_dir / modifier) for modifier in inexistent_modifiers])}"
            )


class LhcBestKnowledgeCreator(LhcModelCreator):
    EXTRACTED_MQTS_FILENAME = 'extracted_mqts.str'
    CORRECTIONS_FILENAME = 'corrections.madx'

    @classmethod
    def get_madx_script(cls, accel: Lhc):
        if accel.excitation is not AccExcitationMode.FREE:
            raise AcceleratorDefinitionError("Don't set ACD or ADT for best knowledge model.")
        if accel.energy is None:
            raise AcceleratorDefinitionError("Best knowledge model requires energy.")

        corrections_file = accel.model_dir / cls.CORRECTIONS_FILENAME  # existence is tested in madx
        mqts_file = accel.model_dir / cls.EXTRACTED_MQTS_FILENAME  # existence is tested in madx

        madx_script = accel.get_base_madx_script(best_knowledge=True)
        madx_script += (
            f"call, file = '{corrections_file}';\n"
            f"call, file = '{mqts_file}';\n"
            f"exec, do_twiss_monitors(LHCB{accel.beam}, '{accel.model_dir / TWISS_BEST_KNOWLEDGE_DAT}', {accel.dpp});\n"
            f"exec, do_twiss_elements(LHCB{accel.beam}, '{accel.model_dir / TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT}', {accel.dpp});\n"
        )
        return madx_script

    @classmethod
    def check_run_output(cls, accel: Lhc):
        to_check = [TWISS_BEST_KNOWLEDGE_DAT, TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT]
        cls._check_files_exist(accel.model_dir, to_check)


class LhcCouplingCreator(LhcModelCreator):

    @classmethod
    def get_madx_script(cls, accel: Lhc):
        return cls.get_correction_check_script(accel)
