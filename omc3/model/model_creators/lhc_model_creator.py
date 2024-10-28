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
    LHC_MACROS_RUN3,
    MACROS_DIR,
    TWISS_AC_DAT,
    TWISS_ADT_DAT,
    TWISS_BEST_KNOWLEDGE_DAT,
    TWISS_DAT,
    TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT,
    TWISS_ELEMENTS_DAT,
    PATHFETCHER, AFSFETCHER,  # GITFETCHER, LSAFETCHER,
    AFS_ACCELERATOR_MODEL_REPOSITORY,
    OPTICS_SUBDIR,
    AFS_B2_ERRORS_ROOT,
)
from omc3.model.model_creators.abstract_model_creator import ModelCreator, check_folder_choices
from omc3.utils.iotools import get_check_suffix_func, create_dirs

LOGGER = logging.getLogger(__name__)


def _b2_columns() -> List[str]:
    cols_outer = [f"{KP}{num}{S}L" for KP in (
        "K", "P") for num in range(21) for S in ("", "S")]
    cols_middle = ["DX", "DY", "DS", "DPHI", "DTHETA", "DPSI", "MREX", "MREY", "MREDX", "MREDY",
                   "AREX", "AREY", "MSCALX", "MSCALY", "RFM_FREQ", "RFM_HARMON", "RFM_LAG"]
    return cols_outer[:42] + cols_middle + cols_outer[42:]


class LhcModelCreator(ModelCreator):
    acc_model_name = "lhc"

    @classmethod
    def check_options(cls, accel: Lhc, opt) -> bool:
        """ Use the fetcher to list choices if requested. """
        
        # Set the fetcher paths ---
        if opt.fetch == PATHFETCHER:
            accel.acc_model_path = Path(opt.path)

        elif opt.fetch == AFSFETCHER:
            # list 'year' choices ---
            accel.acc_model_path = check_folder_choices(
                AFS_ACCELERATOR_MODEL_REPOSITORY / cls.acc_model_name,
                msg="No optics tag (flag --year) given",
                selection=accel.year,
                list_choices=opt.list_choices,
                predicate=Path.is_dir
            )
        else:
            raise AttributeError(
                f"{accel.NAME} model creation requires one of the following fetchers: "
                f"[{PATHFETCHER}, {AFSFETCHER}]. "
                "Please provide one with the flag `--fetch afs` "
                "or `--fetch path --path PATH`."
            )

        if accel.acc_model_path is None:
            return False

        # list optics choices ---
        if opt.list_choices:
            check_folder_choices(
                accel.acc_model_path / OPTICS_SUBDIR,
                msg="No modifier given",
                selection=None,  # TODO: could check if user made valid choice
                list_choices=opt.list_choices,
                predicate=get_check_suffix_func(".madx")
            )
            return False

        return True

    @classmethod
    def get_madx_script(cls, accel: Lhc) -> str:  # nominal
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

    @classmethod
    def get_correction_check_script(
        cls, accel: Lhc, corr_file: str = "changeparameters_couple.madx", chrom: bool = False
    ) -> str:
        madx_script = accel.get_base_madx_script()
        madx_script += (
            f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{accel.model_dir / 'twiss_no.dat'!s}', 0.0);\n"
            f"call, file = '{corr_file}';\n"
            f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{accel.model_dir / 'twiss_cor.dat'!s}', 0.0);\n"
        )
        if chrom:
            madx_script += (
                f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{accel.model_dir / 'twiss_cor_dpm.dat'}', %DELTAPM);\n"
                f"exec, do_twiss_monitors_and_ips(LHCB{accel.beam}, '{accel.model_dir / 'twiss_cor_dpp.dat'}', %DELTAPP);\n"
            )
        return madx_script

    @classmethod
    def prepare_run(cls, accel: Lhc) -> None:
        LOGGER.info("preparing run ...")
        cls.prepare_symlink(accel)
        cls.check_accelerator_instance(accel)
        
        LOGGER.debug("Preparing model creation structure")
        macros_path = accel.model_dir / MACROS_DIR
        LOGGER.info("creating macros dirs")
        create_dirs(macros_path)

        LOGGER.debug("Copying macros to model directory")
        lib_path = Path(__file__).parent.parent / "madx_macros"
        shutil.copy(lib_path / GENERAL_MACROS, macros_path / GENERAL_MACROS)
        shutil.copy(lib_path / LHC_MACROS, macros_path / LHC_MACROS)
        shutil.copy(lib_path / LHC_MACROS_RUN3, macros_path / LHC_MACROS_RUN3)

        # reconstruct path to b2_errors
        b2_error_path = None
        if accel.b2_errors is not None:
            LOGGER.debug("copying B2 error tables")

            b2_error_path = AFS_B2_ERRORS_ROOT / f"Beam{accel.beam}" / f"{accel.b2_errors}.errors"
            b2_madx_path = AFS_B2_ERRORS_ROOT / f"Beam{accel.beam}" / f"{accel.b2_errors}.madx"
            shutil.copy(
                b2_madx_path,
                accel.model_dir / B2_SETTINGS_MADX,
            )
            b2_table = tfs.read(b2_error_path, index="NAME")
            gen_df = pd.DataFrame(
                data=np.zeros((b2_table.index.size, len(_b2_columns()))),
                index=b2_table.index,
                columns=_b2_columns(),
            )
            gen_df["K1L"] = b2_table.loc[:, "K1L"].to_numpy()
            tfs.write(
                accel.model_dir / B2_ERRORS_TFS,
                gen_df,
                headers_dict={"NAME": "EFIELD", "TYPE": "EFIELD"},
                save_index="NAME",
            )

        if accel.energy is not None:
            core = f"{int(accel.energy):04d}"

            LOGGER.debug("Copying error defs for analytical N-BPM method")
            error_dir_path = accel.get_lhc_error_dir()
            shutil.copy(error_dir_path /
                        f"{core}GeV.tfs", accel.model_dir / ERROR_DEFFS_TXT)


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
        inexistent_modifiers = [
            m for m in accel.modifiers if not (accel.model_dir / m).exists()]
        if len(inexistent_modifiers):
            raise AcceleratorDefinitionError(
                "The following modifier files do not exist: "
                f"{', '.join([str(accel.model_dir / modifier) for modifier in inexistent_modifiers])}"
            )



class LhcBestKnowledgeCreator(LhcModelCreator):
    EXTRACTED_MQTS_FILENAME: str = "extracted_mqts.str"

    @classmethod
    def check_options(cls, accel_inst, opt) -> bool:

        if accel_inst.list_b2_errors:
            errors_dir = AFS_B2_ERRORS_ROOT / f"Beam{accel_inst.beam}"
            for d in errors_dir.iterdir():
                if d.suffix==".errors" and d.name.startswith("MB2022"):
                    print(d.stem)
            return False

        return super().check_options(accel_inst, opt)

    @classmethod
    def get_madx_script(cls, accel: Lhc) -> str:
        if accel.excitation is not AccExcitationMode.FREE:
            raise AcceleratorDefinitionError(
                "Don't set ACD or ADT for best knowledge model.")

        madx_script = accel.get_base_madx_script(best_knowledge=True)

        mqts_file = accel.model_dir / cls.EXTRACTED_MQTS_FILENAME
        if mqts_file.exists():
            madx_script += f"call, file = '{mqts_file}';\n"

        madx_script += (
            f"exec, do_twiss_monitors(LHCB{accel.beam}, '{accel.model_dir / TWISS_BEST_KNOWLEDGE_DAT}', {accel.dpp});\n"
            f"exec, do_twiss_elements(LHCB{accel.beam}, '{accel.model_dir / TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT}', {accel.dpp});\n"
        )
        return madx_script

    @classmethod
    def check_run_output(cls, accel: Lhc) -> None:
        files_to_check = [TWISS_BEST_KNOWLEDGE_DAT,
                          TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT]
        cls._check_files_exist(accel.model_dir, files_to_check)


class LhcCouplingCreator(LhcModelCreator):
    @classmethod
    def get_madx_script(cls, accel: Lhc) -> str:
        return cls.get_correction_check_script(accel)
