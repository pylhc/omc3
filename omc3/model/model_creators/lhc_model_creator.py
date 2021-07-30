"""
LHC Model Creator
-----------------

This module provides convenience functions for model creation of the ``LHC``.
"""
import logging
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tfs
from omc3.model.accelerators.accelerator import AcceleratorDefinitionError, AccExcitationMode
from omc3.model.accelerators.lhc import Lhc
from omc3.model.constants import (
    B2_ERRORS_TFS,
    B2_SETTINGS_MADX,
    ERROR_DEFFS_TXT,
    GENERAL_MACROS,
    LHC_MACROS,
    MACROS_DIR,
    TWISS_AC_DAT,
    TWISS_ADT_DAT,
    TWISS_BEST_KNOWLEDGE_DAT,
    TWISS_DAT,
    TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT,
    TWISS_ELEMENTS_DAT,
)
from omc3.model.model_creators.abstract_model_creator import ModelCreator
from omc3.segment_by_segment.phase_writer import create_phase_segment
from omc3.utils import iotools

LOGGER = logging.getLogger(__name__)


def _b2_columns() -> List[str]:
    cols_outer = [f"{KP}{num}{S}L" for KP in ("K", "P") for num in range(21) for S in ("", "S")]
    cols_middle = ["DX", "DY", "DS", "DPHI", "DTHETA", "DPSI", "MREX", "MREY", "MREDX", "MREDY",
                   "AREX", "AREY", "MSCALX", "MSCALY", "RFM_FREQ", "RFM_HARMON", "RFM_LAG"]
    return cols_outer[:42] + cols_middle + cols_outer[42:]


class LhcModelCreator(ModelCreator):
    def __init__(self, accel: Lhc, *args, **kwargs):
        super().__init__(accel, *args, **kwargs)

    def get_madx_script(self) -> str:  # nominal
        accel = self.accel
        use_acd = "1" if (accel.excitation == AccExcitationMode.ACD) else "0"
        use_adt = "1" if (accel.excitation == AccExcitationMode.ADT) else "0"
        madx_script = accel.get_base_madx_script()
        madx_script += (
            f"exec, do_twiss_monitors(LHCB{accel.beam}, '{accel.model_dir / TWISS_DAT}', {accel.dpp});\n"
            f"exec, do_twiss_elements(LHCB{accel.beam}, '{accel.model_dir / TWISS_ELEMENTS_DAT}', {accel.dpp});\n"
        )
        if accel.excitation != AccExcitationMode.FREE or accel.drv_tunes is not None:
            # allow user to modify script and enable excitation, if driven tunes are given
            madx_script += (
                f"use_acd={use_acd};\nuse_adt={use_adt};\n"
                f"if(use_acd == 1){{\n"
                f"exec, twiss_ac_dipole({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.drv_tunes[0]}, {accel.drv_tunes[1]}, {accel.beam}, '{accel.model_dir / TWISS_AC_DAT}', {accel.dpp});\n"
                f"}}else if(use_adt == 1){{\n"
                f"exec, twiss_adt({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.drv_tunes[0]}, {accel.drv_tunes[1]}, {accel.beam}, '{accel.model_dir / TWISS_ADT_DAT}', {accel.dpp});\n"
                f"}}\n"
            )
        return madx_script

    def prepare_run(self) -> None:
        accel = self.accel
        self.check_accelerator_instance(accel)
        LOGGER.debug("Preparing model creation structure")
        macros_path = accel.model_dir / MACROS_DIR
        iotools.create_dirs(macros_path)

        LOGGER.debug("Copying macros to model directory")
        lib_path = Path(__file__).parent.parent / "madx_macros"
        shutil.copy(lib_path / GENERAL_MACROS, macros_path / GENERAL_MACROS)
        shutil.copy(lib_path / LHC_MACROS, macros_path / LHC_MACROS)

        if accel.energy is not None:
            LOGGER.debug("Copying B2 error files for given energy in model directory")
            core = f"{int(accel.energy * 1000):04d}"
            error_dir_path = accel.get_lhc_error_dir()
            shutil.copy(error_dir_path / f"{core}GeV.tfs", accel.model_dir / ERROR_DEFFS_TXT)
            shutil.copy(
                error_dir_path / "b2_errors_settings" / f"beam{accel.beam}_{core}GeV.madx",
                accel.model_dir / B2_SETTINGS_MADX,
            )
            b2_table = tfs.read(error_dir_path / f"b2_errors_beam{accel.beam}.tfs", index="NAME")
            gen_df = pd.DataFrame(
                data=np.zeros((b2_table.index.size, len(_b2_columns()))),
                index=b2_table.index,
                columns=_b2_columns(),
            )
            gen_df["K1L"] = b2_table.loc[:, f"K1L_{core}"].to_numpy()
            tfs.write(
                accel.model_dir / B2_ERRORS_TFS,
                gen_df,
                headers_dict={"NAME": "EFIELD", "TYPE": "EFIELD"},
                save_index="NAME",
            )

    @staticmethod
    def check_accelerator_instance(accel: Lhc) -> None:
        accel.verify_object()  # should have been done anyway, but cannot hurt (jdilly)

        # Creator specific checks
        if accel.model_dir is None:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete: model directory (outputdir option) was not given."
            )

        if accel.modifiers is None or not len(accel.modifiers):
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete: no modifiers could be found."
            )

        # hint: if modifiers are given as absolute paths: `path / abs_path` returns `abs_path`  (jdilly)
        inexistent_modifiers = [m for m in accel.modifiers if not (accel.model_dir / m).exists()]
        if len(inexistent_modifiers):
            raise AcceleratorDefinitionError(
                "The following modifier files do not exist: "
                f"{', '.join([str(accel.model_dir / modifier) for modifier in inexistent_modifiers])}"
            )


class LhcBestKnowledgeCreator(LhcModelCreator):
    EXTRACTED_MQTS_FILENAME: str = "extracted_mqts.str"
    CORRECTIONS_FILENAME: str = "corrections.madx"

    def get_madx_script(self) -> str:
        accel = self.accel
        if accel.excitation is not AccExcitationMode.FREE:
            raise AcceleratorDefinitionError("Don't set ACD or ADT for best knowledge model.")
        if accel.energy is None:
            raise AcceleratorDefinitionError("Best knowledge model requires energy.")

        madx_script = accel.get_base_madx_script(best_knowledge=True)
        madx_script += (
            f"call, file = '{accel.model_dir / self.CORRECTIONS_FILENAME}';\n"
            f"call, file = '{accel.model_dir / self.EXTRACTED_MQTS_FILENAME}';\n"
            f"exec, do_twiss_monitors(LHCB{accel.beam}, '{accel.model_dir / TWISS_BEST_KNOWLEDGE_DAT}', {accel.dpp});\n"
            f"exec, do_twiss_elements(LHCB{accel.beam}, '{accel.model_dir / TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT}', {accel.dpp});\n"
        )
        return madx_script

    def post_run(self) -> None:
        files_to_check = [TWISS_BEST_KNOWLEDGE_DAT, TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT]
        self._check_files_exist(self.accel.model_dir, files_to_check)


class LhcSegmentCreator(LhcModelCreator):
    """ Creates Segment of a model. """
    def __init__(self, accel: Lhc, measurement_dir: Path, start: str, end: str, label: str, *args, **kwargs):
        super().__init__(accel, *args, **kwargs)
        self.start = start
        self.end = end
        self.label = label
        self.measurement_dir = measurement_dir

    def prepare_run(self):
        super().prepare_run()
        self._create_correction_file()
        self.create_measurement_file()

    def get_madx_script(self):
        accel = self.accel
        madx_script = accel.get_base_madx_script()

        madx_template = accel.get_file("segment.madx").read_text()
        replace_dict = {
            "NUM_BEAM": accel.beam,  # LHC only
            "LABEL": self.label,  # all
            "STARTFROM": self.start,  # all
            "ENDAT": self.end,  # all
            "OUTPUT": str(self.accel.model_dir),  # all
        }
        madx_script += madx_template % replace_dict
        return madx_script

    def create_measurement_file(self):
        betain_name = f"measurement_{self.label}.madx"
        df_betx = tfs.read(self.measurement_dir / 'beta_phase_x.tfs', index="NAME")
        df_bety = tfs.read(self.measurement_dir / 'beta_phase_y.tfs', index="NAME")

        betx_start = df_betx.loc[self.start, 'BETX']
        betx_end = df_betx.loc[self.end, 'BETX']

        alfx_start = df_betx.loc[self.start, 'ALFX']
        alfx_end = -df_betx.loc[self.end, 'ALFX']

        bety_start = df_bety.loc[self.start, 'BETY']
        bety_end = df_bety.loc[self.end, 'BETY']

        alfy_start = df_bety.loc[self.start, 'ALFY']
        alfy_end = -df_bety.loc[self.end, 'ALFY']

        # # For Tests
        # from optics_functions.coupling import rmatrix_from_coupling
        # f_ini=pd.DataFrame()
        # f_ini["BETX"] = betx_start
        # f_ini["BETY"] = bety_start
        # f_ini["ALFX"] = alfx_start
        # f_ini["ALFy"] = alfy_start
        # f_ini['F1001'] = 0.001 + 0.002j
        # f_ini["F1010"] = 0.0001 + 0.0002
        #
        # f_end=pd.DataFrame()
        # f_end["BETX"] = betx_end
        # f_end["BETY"] = bety_end
        # f_end["ALFX"] = alfx_end
        # f_end["ALFy"] = alfy_end
        # f_end["F1001"] = 0.0032 + 0.0012j
        # f_end["F1010"] = 0.00013 + 0.0002j
        #
        # ini_r = rmatrix_from_coupling(f_ini)
        # end_r = rmatrix_from_coupling(f_end)

        measurement_dict = dict(
            betx_ini=betx_start,
            bety_ini=bety_start,
            alfx_ini=alfx_start,
            alfy_ini=alfy_start,
            dx_ini=0,
            dy_ini=0,
            dpx_ini=0,
            dpy_ini=0,
            wx_ini=0,
            phix_ini=0,
            wy_ini=0,
            phiy_ini=0,
            wx_end=0,
            phix_end=0,
            wy_end=0,
            phiy_end=0,
            ini_r11=0,
            ini_r12=0,
            ini_r21=0,
            ini_r22=0,
            end_r11=0,
            end_r12=0,
            end_r21=0,
            end_r22=0,
            betx_end=betx_end,
            bety_end=bety_end,
            alfx_end=alfx_end,
            alfy_end=alfy_end,
            dx_end=0,
            dy_end=0,
            dpx_end=0,
            dpy_end=0,
        )
        betainputfile = self.accel.model_dir / betain_name
        betainputfile.write_text(
            "\n".join(f"{name} = {value};" for name, value in measurement_dict.items())
        )

    def _create_correction_file(self):
        corr_file = Path("corrections_" + self.label + ".madx")
        corr_file = self.accel.model_dir / corr_file
        if not corr_file.is_file():
            corr_file.write_text("! Enter the corrections below:")

    def post_run(self):
        create_phase_segment(self.measurement_dir, self.accel.model_dir, self.label)


class LhcCorrectionCreator(LhcModelCreator):
    TWISS_UNCORRECTED_DAT = 'twiss_no.dat'
    TWISS_CORRECTED_DAT = 'twiss_corr.dat'
    TWISS_CORRECTED_DELTAP_MINUS_DAT = 'twiss_corr_dpm.dat'
    TWISS_CORRECTED_DELTAP_PLUS_DAT = 'twiss_corr_dpp.dat'

    def __init__(self, accel: Lhc, chrom: bool = False, corrections: str = "changeparameters_couple.madx", *args, **kwargs):
        super().__init__(accel, *args, **kwargs)
        self.corrections = corrections
        self.chrom = chrom

    def get_madx_script(self) -> str:
        """
        Returns the ``MAD-X`` script used to verify global corrections. This script should create twiss
        files for before (``twiss_no.dat``) and after (``twiss_corr.dat``) correction.

        Args:
            corr_file (str): File containing the corrections (madx-readable).

        Returns:
            The string of the ``MAD-X`` script used to verify global corrections.
        """
        accel = self.accel
        madx_script = accel.get_base_madx_script()
        madx_script += (
            f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{accel.model_dir / self.TWISS_UNCORRECTED_DAT}', 0.0);\n"
            f"call, file = '{self.corrections}';\n"
            f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{accel.model_dir / self.TWISS_CORRECTED_DAT}', 0.0);\n"
        )
        if self.chrom:
            madx_script += (
                f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{accel.model_dir / self.TWISS_CORRECTED_DELTAP_MINUS_DAT}', %DELTAPM);\n"
                f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{accel.model_dir / self.TWISS_CORRECTED_DELTAP_PLUS_DAT}', %DELTAPP);\n"
            )
        return madx_script
