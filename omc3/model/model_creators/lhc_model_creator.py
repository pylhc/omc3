"""
LHC Model Creator
-----------------

This module provides convenience functions for model creation of the ``LHC``.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

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
    JOB_MODEL_MADX_BEST_KNOWLEDGE,
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
from omc3.model.model_creators.abstract_model_creator import ModelCreator, check_folder_choices, SegmentCreator
from omc3.optics_measurements.constants import NAME
from omc3.utils.iotools import get_check_suffix_func, create_dirs

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)


def _b2_columns() -> list[str]:
    cols_outer = [f"{KP}{num}{S}L" for KP in (
        "K", "P") for num in range(21) for S in ("", "S")]
    cols_middle = ["DX", "DY", "DS", "DPHI", "DTHETA", "DPSI", "MREX", "MREY", "MREDX", "MREDY",
                   "AREX", "AREY", "MSCALX", "MSCALY", "RFM_FREQ", "RFM_HARMON", "RFM_LAG"]
    return cols_outer[:42] + cols_middle + cols_outer[42:]


class LhcModelCreator(ModelCreator):
    acc_model_name = "lhc"

    def __init__(self, accel: Lhc, *args, **kwargs):
        super(LhcModelCreator, self).__init__(accel, *args, **kwargs)

    def check_options(self, opt) -> bool:
        """ Use the fetcher to list choices if requested. """
        accel = self.accel
        
        # Set the fetcher paths ---
        if opt.fetch == PATHFETCHER:
            accel.acc_model_path = Path(opt.path)

        elif opt.fetch == AFSFETCHER:
            # list 'year' choices ---
            accel.acc_model_path = check_folder_choices(
                AFS_ACCELERATOR_MODEL_REPOSITORY / self.acc_model_name,
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


    def prepare_run(self) -> None:
        LOGGER.info("preparing run ...")
        accel = self.accel
        self.prepare_symlink()
        self.check_accelerator_instance()
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

    def check_accelerator_instance(self) -> None:
        accel = self.accel

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
    CORRECTIONS_FILENAME: str = "corrections.madx"
    jobfile: str = JOB_MODEL_MADX_BEST_KNOWLEDGE

    def check_options(self, opt) -> bool:
        accel = self.accel
        if accel.list_b2_errors:
            errors_dir = AFS_B2_ERRORS_ROOT / f"Beam{accel.beam}"
            for d in errors_dir.iterdir():
                if d.suffix==".errors" and d.name.startswith("MB2022"):
                    print(d.stem)
            return False

        return super().check_options(opt)

    def get_madx_script(self) -> str:
        accel = self.accel

        if accel.excitation is not AccExcitationMode.FREE:
            raise AcceleratorDefinitionError(
                "Don't set ACD or ADT for best knowledge model."
            )
        if accel.energy is None:
            raise AcceleratorDefinitionError(
                "Best knowledge model requires energy."
            )

        madx_script = accel.get_base_madx_script(best_knowledge=True)

        corrections_file = accel.model_dir / self.CORRECTIONS_FILENAME  # existence is tested in madx
        mqts_file = accel.model_dir / self.EXTRACTED_MQTS_FILENAME  # existence is tested in madx

        madx_script += (
            f"call, file = '{corrections_file}';\n"
            f"call, file = '{mqts_file}';\n"
            f"exec, do_twiss_monitors(LHCB{accel.beam}, '{accel.model_dir / TWISS_BEST_KNOWLEDGE_DAT}', {accel.dpp});\n"
            f"exec, do_twiss_elements(LHCB{accel.beam}, '{accel.model_dir / TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT}', {accel.dpp});\n"
        )
        return madx_script

    def post_run(self) -> None:
        files_to_check = [TWISS_BEST_KNOWLEDGE_DAT, TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT]
        self._check_files_exist(self.accel.model_dir, files_to_check)
        
        self.accel.model_best_knowledge = tfs.read(self.accel.model_dir / TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT, index=NAME)


class LhcCorrectionModelCreator(LhcModelCreator):
    """
    Creates an updated model from multiple changeparameters inputs 
    (used in iterative correction).
    """
    jobfile = None  # set in init

    def __init__(self, accel: Lhc, twiss_out: Path | str, change_params: Sequence[Path], *args, **kwargs):
        """Model creator for the corrected/matched model of the LHC.

        Args:
            accel (Lhc): Accelerator Class Instance
            twiss_out (Union[Path, str]): Path to the twiss(-elements) file to write
            change_params (Sequence[Path]): Sequence of correction/matching files
        """
        super().__init__(accel, *args, **kwargs)
        self.twiss_out = Path(twiss_out)

        # use absolute paths to force files into twiss_out directory instead of model-dir
        self.jobfile = self.twiss_out.parent.absolute() / f"job.create_{self.twiss_out.stem}.madx"
        self.logfile= self.twiss_out.parent.absolute() / f"job.create_{self.twiss_out.stem}.log"
        self.change_params = change_params

    def get_madx_script(self) -> str:
        accel = self.accel
        madx_script = f"! Based on model '{self.accel.model_dir}'\n{self.accel.get_base_madx_script()}" 
        for corr_file in self.change_params:
            madx_script += f"call, file = '{str(corr_file)}';\n"
        madx_script += f"exec, do_twiss_elements(LHCB{accel.beam}, '{str(self.twiss_out)}', {accel.dpp});\n"
        return madx_script

    def prepare_run(self) -> None:
        # As the matched/corrected model is created in the same directory as the original model,
        # we do not need to prepare as much.
        self.check_accelerator_instance()
        LOGGER.debug("Preparing model creation structure")
        macros_path = self.accel.model_dir / MACROS_DIR
        if not macros_path.exists():
            raise AcceleratorDefinitionError(f"Folder for the macros does not exist at {macros_path:s}.")
    
    def post_run(self) -> None:
        files_to_check = [self.twiss_out, self.jobfile, self.logfile]
        self._check_files_exist(self.accel.model_dir, files_to_check)


class LhcSegmentCreator(SegmentCreator, LhcModelCreator):

    def get_madx_script(self):
        accel = self.accel
        madx_script = accel.get_base_madx_script()

        madx_script += "\n".join([
            "",
            f"! ----- Segment-by-Segment propagation for {self.segment.name} -----",
            f"",
            f"use, period = LHCB{accel.beam};",
            f"option, echo;",
            f"",
            f"twiss;",
            f"exec, save_initial_and_final_values(",
            f"    LHCB{accel.beam},",
            f"    {self.segment.start},",
            f"    {self.segment.end}, ",
            f"    \"{accel.model_dir / self.measurement_madx!s}\",",
            f"    biniLHCB{accel.beam},",
            f"    bendLHCB{accel.beam}",
            f");",
            f"",
            f"exec, extract_segment_sequence(",
            f"    LHCB{accel.beam},",
            f"    forward_LHCB{accel.beam},",
            f"    backward_LHCB{accel.beam},",
            f"    {self.segment.start},",
            f"    {self.segment.end},",
            f");",
            f"",
            f"exec, beam_LHCB{accel.beam}(forward_LHCB{accel.beam});",  # TODO: use engery in macro
            f"exec, beam_LHCB{accel.beam}(backward_LHCB{accel.beam});",  # TODO: use engery in macro
            f"exec, twiss_segment(forward_LHCB{accel.beam}, \"{self.twiss_forward!s}\", biniLHCB{accel.beam});",
            f"exec, twiss_segment(backward_LHCB{accel.beam}, \"{self.twiss_backward!s}\", bendLHCB{accel.beam});",
            "",
        ])

        if (self.output_dir / self.corrections_madx).is_file():
            madx_script += "\n".join([
                f"call, file=\"{self.corrections_madx!s}\";",
                f"exec, twiss_segment(forward_LHCB{accel.beam}, "
                f"\"{self.twiss_forward_corrected}\", biniLHCB{accel.beam});",
                f"exec, twiss_segment(backward_LHCB{accel.beam}, "
                f"\"{self.twiss_backward_corrected}\", bendLHCB{accel.beam});",
                "",
            ])

        return madx_script

    def post_run(self):
        check_files = [self.twiss_forward, self.twiss_backward]
        if (self.output_dir / self.corrections_madx).is_file():
            check_files += [self.twiss_backward_corrected, self.twiss_backward_corrected]
        self._check_files_exist(self.accel.model_dir, check_files)
