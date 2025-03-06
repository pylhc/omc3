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

from omc3.correction.constants import ORBIT_DPP
from omc3.model.accelerators.accelerator import (
    AcceleratorDefinitionError,
    AccExcitationMode,
)
from omc3.model.accelerators.lhc import Lhc
from omc3.model.constants import (
    AFS_ACCELERATOR_MODEL_REPOSITORY,
    AFS_B2_ERRORS_ROOT,
    B2_ERRORS_TFS,
    B2_SETTINGS_MADX,
    ERROR_DEFFS_TXT,
    GENERAL_MACROS,
    JOB_MODEL_MADX_BEST_KNOWLEDGE,
    JOB_MODEL_MADX_NOMINAL,
    LHC_REMOVE_TRIPLET_SYMMETRY_RELPATH,
    LHC_MACROS,
    LHC_MACROS_RUN3,
    MACROS_DIR,
    MADX_ENERGY_VAR,
    MODIFIER_TAG,
    OPTICS_SUBDIR,
    TWISS_AC_DAT,
    TWISS_ADT_DAT,
    TWISS_BEST_KNOWLEDGE_DAT,
    TWISS_DAT,
    TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT,
    TWISS_ELEMENTS_DAT,
    Fetcher,
)
from omc3.model.model_creators.abstract_model_creator import (
    CorrectionModelCreator,
    ModelCreator,
    SegmentCreator,
    check_folder_choices,
)
from omc3.utils.iotools import create_dirs, get_check_suffix_func

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

    def __init__(self, accel: Lhc, *args, **kwargs):
        LOGGER.debug("Initializing LHC Model Creator")
        super(LhcModelCreator, self).__init__(accel, *args, **kwargs)

    def prepare_options(self, opt):
        """ Use the fetcher to list choices if requested. """
        accel: Lhc = self.accel

        # Set the fetcher paths ---
        if opt.fetch == Fetcher.PATH:
            if opt.path is None:
                raise AcceleratorDefinitionError(
                    "Path fetcher chosen, but no path provided."
                )
            acc_model_path = Path(opt.path)

        elif opt.fetch == Fetcher.AFS:
            # list 'year' choices ---
            acc_model_path = check_folder_choices(
                AFS_ACCELERATOR_MODEL_REPOSITORY / accel.NAME,
                msg="No optics tag (flag --year) given",
                selection=accel.year,
                list_choices=opt.list_choices,
                predicate=Path.is_dir
            )  # raises AcceleratorDefintionError if not valid choice
        else:
            LOGGER.warning(
                f"{accel.NAME} model creation is expected to run via a fetcher these days. "
                "If you are creating an older model, this might all be correct "
                "and you can ignore this warning. Otherwise you will soon run into "
                "a MAD-X error. In this case, please provide a fetcher for the model via --fetch flag. "
            )
            return

        # list optics choices ---
        if opt.list_choices:  # assumes we want to list optics. Invoked even if modifiers are given!
            check_folder_choices(  
                acc_model_path / OPTICS_SUBDIR,
                msg="No modifier given",
                selection=None,  # TODO: could check if user made valid choice
                list_choices=opt.list_choices,
                predicate=get_check_suffix_func(".madx")
            )  # raises AcceleratorDefinitionError
        
        # Set acc model path to be used in model creator ---
        accel.acc_model_path = acc_model_path

    def prepare_run(self) -> None:
        super().prepare_run()  # create symlink, find modifiers

        accel: Lhc = self.accel

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
                "The accelerator definition is incomplete: No modifiers could be found. "
                "Hint: If the accelerator class was instantiated from a model dir, "
                f"make sure {JOB_MODEL_MADX_NOMINAL} exists and contains the '{MODIFIER_TAG}' tag."
            )

        # hint: if modifiers are given as absolute paths: `path / abs_path` returns `abs_path`  (jdilly)
        # hint: modifiers were at this point probably already checked (and adapted) in `prepare_run()`.
        inexistent_modifiers = [
            m for m in accel.modifiers if not (accel.model_dir / m).exists()]
        if len(inexistent_modifiers):
            raise AcceleratorDefinitionError(
                "The following modifier files do not exist: "
                f"{', '.join([str(accel.model_dir / modifier) for modifier in inexistent_modifiers])}"
            )

    def get_madx_script(self) -> str:  # nominal
        """ Get madx script to create a LHC model."""
        accel: Lhc = self.accel

        madx_script = self.get_base_madx_script()
        madx_script += (
            f"exec, do_twiss_monitors(LHCB{accel.beam}, '{accel.model_dir / TWISS_DAT}', {accel.dpp});\n"
            f"exec, do_twiss_elements(LHCB{accel.beam}, '{accel.model_dir / TWISS_ELEMENTS_DAT}', {accel.dpp});\n"
        )
        if accel.excitation != AccExcitationMode.FREE or accel.drv_tunes is not None:
            # allow user to modify script and enable excitation, if driven tunes are given
            use_acd = accel.excitation == AccExcitationMode.ACD
            use_adt = accel.excitation == AccExcitationMode.ADT
            madx_script += (
                f"use_acd={use_acd:d};\n"
                f"use_adt={use_adt:d};\n"
                f"if(use_acd == 1){{\n"
                f"exec, twiss_ac_dipole({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.drv_tunes[0]}, {accel.drv_tunes[1]}, {accel.beam}, '{accel.model_dir / TWISS_AC_DAT}', {accel.dpp});\n"
                f"}}else if(use_adt == 1){{\n"
                f"exec, twiss_adt({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.drv_tunes[0]}, {accel.drv_tunes[1]}, {accel.beam}, '{accel.model_dir / TWISS_ADT_DAT}', {accel.dpp});\n"
                f"}}\n"
            )
        return madx_script

    def get_base_madx_script(self) -> str:
        """ Returns the base LHC madx script."""
        accel: Lhc = self.accel
        
        madx_script = self._get_sequence_initialize_script()
        if self._uses_ats_knobs():
            madx_script += "\n! ----- Matching Knobs -----\n"
            LOGGER.debug(
                "According to the optics year or the --ats flag being provided, ATS macros and knobs will be used"
            )
            madx_script += f"exec, match_tunes_ats({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.beam});\n"
            madx_script += f"exec, coupling_knob_ats({accel.beam});\n"
        else:
            madx_script += f"exec, match_tunes({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.beam});\n"
            madx_script += f"exec, coupling_knob({accel.beam});\n"

        return madx_script

    def _get_sequence_initialize_script(self) -> str: 
        """ Returns the LHC sequence initialization script.
        
        This is split up here from the matching (in the base-script), 
        to accompany the needs of the Best Knowledge Model Creator, 
        see below.
        """
        accel: Lhc = self.accel

        madx_script = (
            f"{self._get_madx_script_info_comments()}\n\n"
            f"call, file = '{accel.model_dir / MACROS_DIR / GENERAL_MACROS}';\n"
            f"call, file = '{accel.model_dir / MACROS_DIR / LHC_MACROS}';\n"
        )
        madx_script += f"{MADX_ENERGY_VAR} = {accel.energy};\n"
        madx_script += "exec, define_nominal_beams();\n\n"
        if self._uses_run3_macros():
            LOGGER.debug(
                "According to the optics year, Run 3 versions of the macros will be used"
            )
            madx_script += (
                f"call, file = '{accel.model_dir / MACROS_DIR / LHC_MACROS_RUN3}';\n"
            )

        madx_script += "\n! ----- Calling Sequence -----\n"
        madx_script += "option, -echo;  ! suppress output from base sequence loading to keep the log small\n"
        madx_script += self._get_madx_main_sequence_loading()
        madx_script += "\noption, echo;  ! re-enable output to see the optics settings\n"
        
        madx_script += "\n! ---- Call optics and other modifiers ----\n"

        if accel.modifiers is not None:
            # if modifier is an absolute path, go there, otherwise use the path refers from model_dir
            madx_script += "".join(
                f"call, file = '{accel.model_dir / modifier}'; {MODIFIER_TAG}\n"
                for modifier in accel.modifiers
            )

        if accel.year in ['2012', '2015', '2016', '2017', '2018', '2021', 'hllhc1.3']:
            # backwards compatibility with pre acc-models optics
            # WARNING: This might override values extracted via the knob-extractor.
            madx_script += (
                f"\n! ----- Defining Configuration Specifics -----\n"
                f"xing_angles = {'1' if accel.xing else '0'};\n"
                f"if(xing_angles==1){{\n"
                f"    exec, set_crossing_scheme_ON();\n"
                f"}}else{{\n"
                f"    exec, set_default_crossing_scheme();\n"
                f"}}\n"
            )
        
        if accel.acc_model_path is not None:
            remove_symmetry_knob_madx = accel.acc_model_path / LHC_REMOVE_TRIPLET_SYMMETRY_RELPATH
            if remove_symmetry_knob_madx.exists():  # alternatively check if year != 2018/2021
                madx_script += (
                "\n! ----- Remove IR symmetry definitions -----\n"
                f"\ncall, file=\"{remove_symmetry_knob_madx!s}\"; "
                "! removes 'ktqx.r1 := -ktqx.l1'-type issues\n"
                )

        madx_script += (
            "\n! ----- Finalize Sequence -----\n"
            "exec, cycle_sequences();\n"
            f"use, sequence = LHCB{accel.beam};\n"
        )
        return madx_script
    
    def get_update_deltap_script(self, deltap: float | str) -> str:
        """ Update the dpp in the LHC.
        
        Args:
            deltap (float | str): The dpp to update the LHC to.
        """
        accel: Lhc = self.accel
        if not isinstance(deltap, str):
            deltap = f"{deltap:.15e}"

        madx_script = (
            f"twiss, deltap={deltap};\n"
            "correct, mode=svd;\n\n"
            
            "! The same as match_tunes, but include deltap in the matching\n"
            f"exec, find_complete_tunes({accel.nat_tunes[0]}, {accel.nat_tunes[1]}, {accel.beam});\n"
            f"match, deltap={deltap};\n"
        ) # Works better when split up
        madx_script += "\n".join([f"vary, name={knob};" for knob in self.get_tune_knobs()]) + "\n"
        madx_script += (
            "constraint, range=#E, mux=total_qx, muy=total_qy;\n"
            "lmdif, tolerance=1e-10;\n"
            "endmatch;\n"
        )
        return madx_script

    def _get_madx_script_info_comments(self) -> str:
        accel: Lhc = self.accel
        info_comments = (
                f'title, "LHC Model created by omc3";\n'
                f"! Model directory        {Path(accel.model_dir).absolute()}\n"
        )
        if accel.acc_model_path is not None:
            info_comments += (
                f"! Acc-Models             {Path(accel.acc_model_path).absolute()}\n"
            )
        info_comments += (
                f"! LHC year               {accel.year}\n"
                f"! Natural Tune X         {accel.nat_tunes[0]:10.3f}\n"
                f"! Natural Tune Y         {accel.nat_tunes[1]:10.3f}\n"
                f"! Best Knowledge         {'NO' if accel.model_best_knowledge is None else 'YES':>10s}\n"
        )
        if accel.excitation == AccExcitationMode.FREE:
            info_comments += (
                f"! Excitation             {'NO':>10s}\n"
            )
        else:
            info_comments += (
                f"! Excitation             {'ACD' if accel.excitation == AccExcitationMode.ACD else 'ADT':>10s}\n"
                f"! > Driven Tune X        {accel.drv_tunes[0]:10.3f}\n"
                f"! > Driven Tune Y        {accel.drv_tunes[1]:10.3f}\n"
            )
        return info_comments

    def _get_madx_main_sequence_loading(self) -> str:
        accel: Lhc = self.accel

        if accel.acc_model_path is not None:
            main_call = f'call, file = \'{accel.acc_model_path / "lhc.seq"}\';'
            if accel.year.startswith('hl'):
                main_call += f'\ncall, file = \'{accel.acc_model_path / "hllhc_sequence.madx"}\';'
            return main_call
        try:
            return self._get_call_main()
        except AttributeError:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete, mode "
                "has to be specified (--lhcmode option missing?)."
            )
    
    def get_tune_knobs(self) -> tuple[str, str]:
        accel: Lhc = self.accel
        if self._uses_run3_macros():
            return f"dQx.b{accel.beam}_op", f"dQy.b{accel.beam}_op"
        elif self._uses_ats_knobs():
            return f"dQx.b{accel.beam}", f"dQy.b{accel.beam}"
        else:
            return f"KQTD.B{accel.beam}", f"KQTF.B{accel.beam}"

    # Private Methods ##########################################################

    def _uses_ats_knobs(self) -> bool:
        """
        Returns wether the ATS knobs and macros should be used, based on the accel-instance's year.
        If the **--ats** flag was explicitely provided then the returned value will be `True`.
        """
        try:
            if self.accel.ats:
                return True
            return 2018 <= int(self.accel.year) <= 2021  # self.year is always a string
        except ValueError:  # if a "hllhc1.x" version is given
            return False

    def _uses_run3_macros(self) -> bool:
        """Returns whether the Run 3 macros should be called, based on the accel-instance's year."""
        try:
            return int(self.accel.year) >= 2022  # self.year is always a string
        except ValueError:  # if a "hllhc1.x" year is given
            return False

    def _get_call_main(self) -> str:
        accel: Lhc = self.accel

        call_main = f"call, file = '{accel.get_accel_file('main.seq')}';\n"
        if accel.year == "2012":
            call_main += (
                f"call, file = '{accel.get_accel_file('install_additional_elements.madx')}';\n"
            )
        if accel.year == "hllhc1.3":
            call_main += f"call, file = '{accel.get_accel_file('main_update.seq')}';\n"
        return call_main



class LhcBestKnowledgeCreator(LhcModelCreator):  # ---------------------------------------------------------------------
    EXTRACTED_MQTS_FILENAME: str = "extracted_mqts.str"
    CORRECTIONS_FILENAME: str = "corrections.madx"
    jobfile: str = JOB_MODEL_MADX_BEST_KNOWLEDGE
    files_to_check: tuple[str] = (TWISS_BEST_KNOWLEDGE_DAT, TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT)

    def check_options(self, opt) -> bool:
        accel: Lhc = self.accel
        if accel.list_b2_errors:
            errors_dir = AFS_B2_ERRORS_ROOT / f"Beam{accel.beam}"
            for d in errors_dir.iterdir():
                if d.suffix==".errors" and d.name.startswith("MB2022"):
                    print(d.stem)
            return False

        return super().check_options(opt)

    def check_accelerator_instance(self):
        accel: Lhc = self.accel
        super().check_accelerator_instance()
        
        if accel.b2_errors is None:
            raise AcceleratorDefinitionError(
                "No b2 errors specified. These are neccessary for the best knowledge model."
            )
        
        if accel.excitation is not AccExcitationMode.FREE:
            raise AcceleratorDefinitionError(
                "Don't set ACD or ADT for best knowledge model."
            )

    def prepare_run(self):
        accel: Lhc = self.accel
        super().prepare_run()

        LOGGER.debug("Copying B2 error tables")

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

    def get_madx_script(self) -> str:
        accel: Lhc = self.accel
        madx_script = self.get_base_madx_script()

        madx_script += "\n! ----- Load MQTs -----\n"
        mqts_file = accel.model_dir / self.EXTRACTED_MQTS_FILENAME
        if mqts_file.exists():
            madx_script += f"call, file = '{mqts_file}';\n"

        madx_script += "\n! ----- Output Files -----\n"
        madx_script += (
            f"exec, do_twiss_monitors(LHCB{accel.beam}, '{accel.model_dir / TWISS_BEST_KNOWLEDGE_DAT}', {accel.dpp});\n"
            f"exec, do_twiss_elements(LHCB{accel.beam}, '{accel.model_dir / TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT}', {accel.dpp});\n"
        )
        return madx_script
    
    def get_base_madx_script(self) -> str:
        accel: Lhc = self.accel
        
        # don't load the super().get_base_madx_script as this also matches the tunes at the end,
        # which we skip here as we are using the data from the machine.
        madx_script = self._get_sequence_initialize_script()  

        madx_script += (
            f"\n! ----- For Best Knowledge Model -----\n"
            f"readmytable, file = '{accel.model_dir / B2_ERRORS_TFS}', table=errtab;\n"
            f"seterr, table=errtab;\n"
            f"call, file = '{accel.model_dir / B2_SETTINGS_MADX}';\n"
        )
        return madx_script


class LhcCorrectionModelCreator(CorrectionModelCreator, LhcModelCreator):  # -------------------------------------------
    """
    Creates an updated model from multiple changeparameters inputs 
    (used in iterative correction).
    """

    def __init__(self, accel: Lhc, twiss_out: Path | str, corr_files: Sequence[Path | str], update_dpp: bool = False):
        """Model creator for the corrected/matched model of the LHC.

        Inheritance (i.e. __init__ calls) from here should be as follows:
        This super() calls CorrectionModelCreator.
        The super() there calls LhcModelCreator which calls ModelCreator, setting the base-attributes.
        Then Attributes in CorrectionModelCreator are set.

        Args:
            accel (Lhc): Accelerator Class Instance
            twiss_out (Union[Path, str]): Path to the twiss(-elements) file to write
            change_params (Sequence[Path]): Sequence of correction/matching files
        """
        LOGGER.debug("Initializing LHC Correction Model Creator")
        super().__init__(accel, twiss_out, corr_files, update_dpp)  

    def get_madx_script(self) -> str:
        """ Get the madx script for the correction model creator, which updates the model after correcion. """  
        accel: Lhc = self.accel
        madx_script = self.get_base_madx_script()  # do not super().get_madx_script as we don't need the uncorrected output. 

        # First set the dpp to the value in the accelerator model
        madx_script += f"{ORBIT_DPP} = {accel.dpp};\n"

        for corr_file in self.corr_files:  # Load the corrections, can also update ORBIT_DPP
            madx_script += f"call, file = '{str(corr_file)}';\n"
        
        if self.update_dpp: # If we are doing orbit correction, we need to ensure that a correct and a match is done
            madx_script += self.get_update_deltap_script(deltap=ORBIT_DPP)

        madx_script += f'exec, do_twiss_elements(LHCB{accel.beam}, "{str(self.twiss_out)}", {ORBIT_DPP});\n'
        return madx_script
    
    def prepare_run(self) -> None:
        # As the matched/corrected model is created in the same directory as the original model,
        # we do not need to prepare as much.
        self.check_accelerator_instance()
        LOGGER.debug("Preparing model creation structure")
        macros_path = self.accel.model_dir / MACROS_DIR
        if not macros_path.exists():
            raise AcceleratorDefinitionError(f"Folder for the macros does not exist at {macros_path:s}.")
    
    @property
    def files_to_check(self) -> list[str]:
        return [self.twiss_out, self.jobfile, self.logfile]


# LHC Segment Creator ----------------------------------------------------------

class LhcSegmentCreator(SegmentCreator, LhcModelCreator):

    def get_madx_script(self):
        accel: Lhc = self.accel
        madx_script = self.get_base_madx_script()

        madx_script += "\n".join([
            "",
            f"! ----- Segment-by-Segment propagation for {self.segment.name} -----",
            "",
            "! Cycle the sequence to avoid negative length.",
            f"seqedit, sequence=LHCB{accel.beam};",
            "flatten;",
            f"cycle, start={self.segment.start};",
            "endedit;",
            "",
            f"use, period = LHCB{accel.beam};",
            "option, echo;",
            "",
            "twiss;",
            "exec, save_initial_and_final_values(",
            f"    LHCB{accel.beam},",
            f"    {self.segment.start},",
            f"    {self.segment.end}, ",
            f"    \"{accel.model_dir / self.measurement_madx!s}\",",
            f"    biniLHCB{accel.beam},",
            f"    bendLHCB{accel.beam}",
            ");",
            "",
            "exec, extract_segment_sequence(",
            f"    LHCB{accel.beam},",
            f"    forward_LHCB{accel.beam},",
            f"    backward_LHCB{accel.beam},",
            f"    {self.segment.start},",
            f"    {self.segment.end},",
            ");",
            "",
            f"beam, particle = proton, sequence=forward_LHCB{accel.beam}, energy = {MADX_ENERGY_VAR}, bv={accel.beam_direction:d};",
            f"beam, particle = proton, sequence=backward_LHCB{accel.beam}, energy = {MADX_ENERGY_VAR}, bv={accel.beam_direction:d};",
            "",
            f"exec, twiss_segment(forward_LHCB{accel.beam}, \"{self.twiss_forward!s}\", biniLHCB{accel.beam});",
            f"exec, twiss_segment(backward_LHCB{accel.beam}, \"{self.twiss_backward!s}\", bendLHCB{accel.beam});",
            "",
        ])

        if self.corrections is not None:
            madx_script += "\n".join([
                f"call, file=\"{self.corrections_madx!s}\";",
                f"exec, twiss_segment(forward_LHCB{accel.beam}, "
                f"\"{self.twiss_forward_corrected}\", biniLHCB{accel.beam});",
                f"exec, twiss_segment(backward_LHCB{accel.beam}, "
                f"\"{self.twiss_backward_corrected}\", bendLHCB{accel.beam});",
                "",
            ])

        return madx_script
    
    @property
    def files_to_check(self) -> list[str]:
        check_files = [self.twiss_forward, self.twiss_backward]
        if self.corrections is not None:
            check_files += [self.twiss_backward_corrected, self.twiss_backward_corrected]
        return check_files
