import logging
import os
import shutil
from os.path import join
from utils import iotools
import pandas as pd
import numpy as np
from model.accelerators.accelerator import AccExcitationMode, AcceleratorDefinitionError
from model.constants import (MACROS_DIR, GENERAL_MACROS, LHC_MACROS, ERROR_DEFFS_TXT,
                             JOB_ITERATE_MADX, MODIFIERS_MADX, TWISS_BEST_KNOWLEDGE_DAT,
                             TWISS_ADT_DAT, TWISS_AC_DAT, TWISS_ELEMENTS_DAT, TWISS_DAT,
                             TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT, B2_ERRORS_TFS, B2_SETTINGS_MADX )
LOGGER = logging.getLogger(__name__)
import tfs


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
            f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{join(outdir, 'twiss_no.dat')}', 0.0);\n"
            f"call, file = '{corr_file}';\n"
            f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{join(outdir, 'twiss_cor.dat')}', 0.0);\n")
        if chrom:
            madx_script +=(
                f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{join(outdir, 'twiss_cor_dpm.dat')}', %DELTAPM);\n"
                f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{join(outdir, 'twiss_cor_dpp.dat')}', %DELTAPP);\n"
            )
        return madx_script

    @classmethod
    def get_madx_script(cls, accel, outdir):  # nominal
        use_acd = "1" if (accel.excitation == AccExcitationMode.ACD) else "0"
        use_adt = "1" if (accel.excitation == AccExcitationMode.ADT) else "0"
        madx_script = accel.get_base_madx_script(outdir)
        madx_script +=(
                f"use_acd={use_acd};\nuse_adt={use_adt};\n"
                f"exec, do_twiss_monitors(LHCB{accel.beam}, '{join(outdir, TWISS_DAT)}', {accel.dpp});\n"
                f"exec, do_twiss_elements(LHCB{accel.beam}, '{join(outdir, TWISS_ELEMENTS_DAT)}', {accel.dpp});\n"
                f"if(use_acd == 1){{\n"
                f"exec, twiss_ac_dipole({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.drv_tunes[0]}, {accel.drv_tunes[1]}, {accel.beam}, '{join(outdir, TWISS_AC_DAT)}', {accel.dpp});\n"
                f"}}else if(use_adt == 1){{\n"
                f"exec, twiss_adt({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.drv_tunes[0]}, {accel.drv_tunes[1]}, {accel.beam}, '{join(outdir, TWISS_ADT_DAT)}', {accel.dpp});\n"
                f"}}\n")
        return madx_script

    @classmethod
    def prepare_run(cls, lhc_instance, output_path):
        if lhc_instance.fullresponse:
            cls._prepare_fullresponse(lhc_instance, output_path)
        macros_path = join(output_path, MACROS_DIR)
        iotools.create_dirs(macros_path)
        lib_path = join(os.path.dirname(__file__), os.pardir, os.pardir, "lib")
        shutil.copy(join(lib_path, GENERAL_MACROS), join(macros_path, GENERAL_MACROS))
        shutil.copy(join(lib_path, LHC_MACROS), join(macros_path, LHC_MACROS))
        if lhc_instance.energy is not None:
            core = f"{int(lhc_instance.energy*1000):04d}"
            file_path = lhc_instance.get_lhc_error_dir()
            shutil.copy(join(file_path, f"{core}GeV.tfs"), join(output_path, ERROR_DEFFS_TXT))
            shutil.copy(join(file_path, "b2_errors_settings", f"beam{lhc_instance.beam}_{core}GeV.madx"),
                        join(output_path, B2_SETTINGS_MADX))
            b2_table = tfs.read(join(lhc_instance.get_lhc_error_dir(), f"b2_errors_beam{lhc_instance.beam}.tfs"), index="NAME")
            gen_df = pd.DataFrame(data=np.zeros((b2_table.index.size, len(_b2_columns()))),
                                  index=b2_table.index, columns=_b2_columns())
            gen_df["K1L"] = b2_table.loc[:, f"K1L_{core}"].to_numpy()
            tfs.write(join(output_path, B2_ERRORS_TFS), gen_df,
                      headers_dict={"NAME": "EFIELD", "TYPE": "EFIELD"}, save_index="NAME")

    @classmethod
    def _prepare_fullresponse(cls, lhc_instance, output_path):
        madx_script = lhc_instance.get_base_madx_script(output_path)
        madx_script += f"exec, select_monitors();\ncall, file = '{join(output_path, 'iter.madx')}';\n"
        with open(os.path.join(output_path, JOB_ITERATE_MADX), "w") as textfile:
            textfile.write(madx_script)


class LhcBestKnowledgeCreator(LhcModelCreator):

    @classmethod
    def get_madx_script(cls, accel, outdir):
        if accel.excitation is not AccExcitationMode.FREE:
            raise AcceleratorDefinitionError("Don't set ACD or ADT for best knowledge model.")
        if accel.energy is None:
            raise AcceleratorDefinitionError("Best knowledge model requires energy.")
        madx_script = accel.get_base_madx_script(outdir, best_knowledge=True)
        madx_script += (
            f"call, file = '{join(outdir, 'corrections.madx')}';\n"
            f"call, file = '{join(outdir, 'extracted_mqts.str')}';\n"
            f"exec, do_twiss_monitors(LHCB{accel.beam}, '{join(outdir, TWISS_BEST_KNOWLEDGE_DAT)}', {accel.dpp});\n"
            f"exec, do_twiss_elements(LHCB{accel.beam}, '{join(outdir, TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT)}', {accel.dpp});\n"
        )
        return madx_script


class LhcCouplingCreator(LhcModelCreator):
    @classmethod
    def get_madx_script(cls, lhc_instance, output_path):
        return cls.get_correction_check_script(lhc_instance, output_path)



