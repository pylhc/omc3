"""
LHC Model Creator
-----------------

This module provides convenience functions for model creation of the ``LHC``.
"""
import logging
import os
import shutil

import numpy as np
import pandas as pd
import tfs

from omc3.model.accelerators.accelerator import AccExcitationMode, AcceleratorDefinitionError
from omc3.model.constants import (B2_ERRORS_TFS, B2_SETTINGS_MADX,
                                  ERROR_DEFFS_TXT, GENERAL_MACROS,
                                  LHC_MACROS, MACROS_DIR,
                                  TWISS_AC_DAT, TWISS_ADT_DAT,
                                  TWISS_BEST_KNOWLEDGE_DAT, TWISS_DAT,
                                  TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT,
                                  TWISS_ELEMENTS_DAT)
from omc3.utils import iotools
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _b2_columns():
    cols_outer = [f"{KP}{num}{S}L" for KP in ("K", "P") for num in range(21) for S in ("", "S")]
    cols_middle = ["DX", "DY", "DS", "DPHI", "DTHETA", "DPSI", "MREX", "MREY", "MREDX", "MREDY",
                   "AREX", "AREY", "MSCALX", "MSCALY", "RFM_FREQ", "RFM_HARMON", "RFM_LAG"]
    return cols_outer[:42] + cols_middle + cols_outer[42:]


class LhcModelCreator(object):

    @classmethod
    def get_correction_check_script(cls, accel, outdir, corr_file="changeparameters_couple.madx", chrom=False):
        madx_script = accel.get_base_madx_script(accel.model_dir)
        madx_script += (
            f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{outdir / 'twiss_no.dat'!s}', 0.0);\n"
            f"call, file = '{corr_file}';\n"
            f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{outdir / 'twiss_cor.dat'!s}', 0.0);\n")
        if chrom:
            madx_script +=(
                f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{outdir / 'twiss_cor_dpm.dat'}', %DELTAPM);\n"
                f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{outdir / 'twiss_cor_dpp.dat'}', %DELTAPP);\n"
            )
        return madx_script

    @classmethod
    def get_madx_script(cls, accel, outdir):  # nominal
        use_acd = "1" if (accel.excitation == AccExcitationMode.ACD) else "0"
        use_adt = "1" if (accel.excitation == AccExcitationMode.ADT) else "0"
        madx_script = accel.get_base_madx_script(outdir)
        madx_script +=(
                f"use_acd={use_acd};\nuse_adt={use_adt};\n"
                f"exec, do_twiss_monitors(LHCB{accel.beam}, '{outdir /TWISS_DAT}', {accel.dpp});\n"
                f"exec, do_twiss_elements(LHCB{accel.beam}, '{outdir / TWISS_ELEMENTS_DAT}', {accel.dpp});\n"
                f"if(use_acd == 1){{\n"
                f"exec, twiss_ac_dipole({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.drv_tunes[0]}, {accel.drv_tunes[1]}, {accel.beam}, '{outdir / TWISS_AC_DAT}', {accel.dpp});\n"
                f"}}else if(use_adt == 1){{\n"
                f"exec, twiss_adt({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.drv_tunes[0]}, {accel.drv_tunes[1]}, {accel.beam}, '{outdir / TWISS_ADT_DAT}', {accel.dpp});\n"
                f"}}\n")
        return madx_script

    @classmethod
    def prepare_run(cls, lhc_instance, output_path: Path):
        macros_path = output_path / MACROS_DIR
        iotools.create_dirs(macros_path)
        lib_path = Path(__file__).parent.parent / "madx_macros"
        shutil.copy(lib_path / GENERAL_MACROS, macros_path / GENERAL_MACROS)
        shutil.copy(lib_path / LHC_MACROS, macros_path / LHC_MACROS)
        if lhc_instance.energy is not None:
            core = f"{int(lhc_instance.energy*1000):04d}"
            error_dir_path = lhc_instance.get_lhc_error_dir()
            shutil.copy(error_dir_path / f"{core}GeV.tfs", output_path / ERROR_DEFFS_TXT)
            shutil.copy(error_dir_path / "b2_errors_settings" / f"beam{lhc_instance.beam}_{core}GeV.madx",
                        output_path / B2_SETTINGS_MADX)
            b2_table = tfs.read(error_dir_path / f"b2_errors_beam{lhc_instance.beam}.tfs", index="NAME")
            gen_df = pd.DataFrame(data=np.zeros((b2_table.index.size, len(_b2_columns()))),
                                  index=b2_table.index, columns=_b2_columns())
            gen_df["K1L"] = b2_table.loc[:, f"K1L_{core}"].to_numpy()
            tfs.write(output_path / B2_ERRORS_TFS, gen_df,
                      headers_dict={"NAME": "EFIELD", "TYPE": "EFIELD"}, save_index="NAME")



class LhcBestKnowledgeCreator(LhcModelCreator):

    @classmethod
    def get_madx_script(cls, accel, outdir):
        if accel.excitation is not AccExcitationMode.FREE:
            raise AcceleratorDefinitionError("Don't set ACD or ADT for best knowledge model.")
        if accel.energy is None:
            raise AcceleratorDefinitionError("Best knowledge model requires energy.")
        madx_script = accel.get_base_madx_script(outdir, best_knowledge=True)
        madx_script += (
            f"call, file = '{outdir / 'corrections.madx'}';\n"
            f"call, file = '{outdir / 'extracted_mqts.str'}';\n"
            f"exec, do_twiss_monitors(LHCB{accel.beam}, '{outdir / TWISS_BEST_KNOWLEDGE_DAT}', {accel.dpp});\n"
            f"exec, do_twiss_elements(LHCB{accel.beam}, '{outdir / TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT}', {accel.dpp});\n"
        )
        return madx_script



class LhcCouplingCreator(LhcModelCreator):
    @classmethod
    def get_madx_script(cls, lhc_instance, output_path):
        return cls.get_correction_check_script(lhc_instance, output_path)


