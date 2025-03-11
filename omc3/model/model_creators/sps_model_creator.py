""" 
SPS Model Creator
-----------------
"""
from __future__ import annotations

from abc import ABC
from pathlib import Path

from omc3.model.accelerators.accelerator import (
    AccElementTypes,
    AcceleratorDefinitionError,
    AccExcitationMode,
)
from omc3.model.accelerators.sps import Sps
from omc3.model.constants import (
    AFS_ACCELERATOR_MODEL_REPOSITORY,
    MODIFIER_TAG,
    TWISS_AC_DAT,
    TWISS_DAT,
    TWISS_ELEMENTS_DAT,
    Fetcher,
)
from omc3.model.model_creators.abstract_model_creator import (
    CorrectionModelCreator,
    ModelCreator,
    SegmentCreator,
    check_folder_choices,
)
from omc3.utils import logging_tools
from omc3.utils.iotools import get_check_suffix_func

LOGGER = logging_tools.get_logger(__name__)


class SpsModelCreator(ModelCreator, ABC):
    acc_model_name = "sps" 
    _start_bpm = "BPH.13008"  # BPM to cycle to

    def check_accelerator_instance(self) -> None:
        accel: Sps = self.accel
        accel.verify_object()  # should have been done anyway, but cannot hurt (jdilly)

        # Creator specific checks
        if accel.model_dir is None:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete: "
                "Model directory (outputdir option) was not given."
            )
        
        if accel.acc_model_path is None:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete: "
                "SPS model creation only works with acc-models, but `acc-model-path` was not set. "
            )

        if accel.str_file is None:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete: No strength-file. "
            )
    
    def prepare_options(self, opt) -> bool:
        """ Use the fetcher to list choices if requested. """
        accel: Sps = self.accel
        
        if opt.fetch == Fetcher.PATH:
            if opt.path is None:
                raise AcceleratorDefinitionError(
                    "Path fetcher chosen, but no path proivided."
                )
            acc_model_path = Path(opt.path)

        elif opt.fetch == Fetcher.AFS:
            # list 'year' choices ---
            acc_model_path = check_folder_choices(
                AFS_ACCELERATOR_MODEL_REPOSITORY / Sps.NAME,
                msg="No optics tag (flag --year) given",
                selection=accel.year,
                list_choices=opt.list_choices,
                predicate=Path.is_dir
            )  # raises AcceleratorDefintionError if not valid choice
        else:
            LOGGER.warning(
                f"{accel.NAME} model creation is expected to run via a fetcher these days. "
                "If you have set the acc_model_path otherwise manually, this might all be correct "
                "and you can ignore this warning. Otherwise you will soon run into "
                "a MAD-X error. In this case, please provide a fetcher for the model via --fetch flag. "
            )
            return

        str_file = accel.str_file
        if str_file is None or not Path(str_file).is_file():
            str_file = check_folder_choices(
                acc_model_path / "strengths",
                msg="No/Unknown strength file (flag --str_file) selected",
                selection=str_file,
                list_choices=opt.list_choices,
                predicate=get_check_suffix_func(".str")
            )

        if opt.list_choices:
            raise AcceleratorDefinitionError("List Choices Requested, but all choices already set correctly.")  
        
        # Set the found paths ---
        accel.acc_model_path = acc_model_path
        accel.str_file = str_file
    
    def get_base_madx_script(self):
        accel: Sps = self.accel
        use_excitation = accel.excitation != AccExcitationMode.FREE

        # The very basics ---
        madx_script = (
            "option, -echo;  ! suppress output from base sequence loading to keep the log small\n\n"
            "! Load Base Sequence and Strength ---\n"
            f"call, file = '{accel.acc_model_path / 'sps.seq'!s}';\n"
            f"call, file = '{accel.str_file!s}'; {accel.STRENGTH_FILE_TAG}\n"
        )

        if accel.modifiers is not None:
            for modifier in accel.modifiers:
                madx_script += f"call, file = '{accel.acc_model_path / modifier!s}'; {MODIFIER_TAG}\n"


        madx_script += (
            f"call, file ='{accel.acc_model_path / 'toolkit' / 'macro.madx'!s}';\n\n"
            "! Create Beam ---\n"
            "beam;\n\n"
            "twiss;\n\n"  # not sure if needed, but is in the scenarios scripts
            "! Prepare Tunes ---\n"
            f"qx0={accel.nat_tunes[0]:.3f};\n"
            f"qy0={accel.nat_tunes[1]:.3f};\n\n"
        )

        # Install AC dipole ---
        if use_excitation or accel.drv_tunes is not None:
            # allow user to modify script and enable excitation, if driven tunes are given
            madx_script += (
                f"qxd={accel.drv_tunes[0]%1:.3f};\n"
                f"qyd={accel.drv_tunes[1]%1:.3f};\n\n"
                "! Prepare ACDipole Elements ---\n"
                f"use_acd={use_excitation:d}; ! Switch the use of AC dipole\n\n"
                "hacmap21 = 0;\n"
                "vacmap43 = 0;\n"
                "hacmap: matrix, l=0, rm21 := hacmap21;\n"
                "vacmap: matrix, l=0, rm43 := vacmap43;\n\n"
                "ZKH_MARKER: marker;\n"
                "ZKV_MARKER: marker;\n\n"
            )

        madx_script += (
            "! Cycle Sequence ---\n"
            "seqedit, sequence=sps;\n"
            "    flatten;\n"
        )

        if use_excitation or accel.drv_tunes is not None:
            # allow user to modify script and enable excitation, if driven tunes are given
            madx_script += (
                "\n"
                "    ! Install AC dipole ---\n"
                "    if(use_acd == 1){\n"
                "        replace, element=ZKHA.21991, by=ZKH_MARKER;\n"
                "        replace, element=ZKV.21993,  by=ZKV_MARKER;\n"
                "        install, element=hacmap, at=0.0, from=ZKH_MARKER;\n"
                "        install, element=vacmap, at=0.0, from=ZKV_MARKER;\n"
                "    }\n\n"
            )

        madx_script += (
            f"    cycle, start = {self._start_bpm};\n"
             "endedit;\n"
             "use, sequence=sps;\n\n"
        )

        # Match tunes ---
        madx_script += (
             "! Match Tunes ---\n"
            "exec, sps_match_tunes(qx0,qy0);\n\n"
            # "twiss, file = 'sps.tfs';\n"  # also not sure if needed
        )
        return madx_script

    def get_madx_script(self):
        accel: Sps = self.accel
        use_excitation = accel.excitation != AccExcitationMode.FREE

        madx_script = self.get_base_madx_script()
        madx_script += (
             "! Create twiss data files ---\n"
            f"{self._get_select_command(pattern=accel.RE_DICT[AccElementTypes.BPMS])}"
            f"twiss, file = {accel.model_dir / TWISS_DAT};\n"
            "\n"
            f"{self._get_select_command()}"
            f"twiss, file = {accel.model_dir / TWISS_ELEMENTS_DAT};\n"
        )

        if use_excitation or accel.drv_tunes is not None:
            # allow user to modify script and enable excitation, if driven tunes are given
            madx_script += (
                f"if(use_acd == 1){{\n"
                 "    betxac = table(twiss, hacmap, betx);\n"
                 "    betyac = table(twiss, vacmap, bety);\n"
                f"    hacmap21 := 2*(cos(2*pi*qxd)-cos(2*pi*qx0))/(betxac*sin(2*pi*qx0));\n"
                f"    vacmap43 := 2*(cos(2*pi*qyd)-cos(2*pi*qy0))/(betyac*sin(2*pi*qy0));\n"
                "\n"
                f"{self._get_select_command(pattern=accel.RE_DICT[AccElementTypes.BPMS], indent=4)}"
                f"    twiss, file = {accel.model_dir / TWISS_AC_DAT};\n" 
                f"}}\n"
            )
        return madx_script
    

class SpsCorrectionModelCreator(CorrectionModelCreator, SpsModelCreator):
    """
    Creates an updated model from multiple changeparameters inputs 
    (used in iterative correction).
    """
    def prepare_run(self) -> None:
        # As the matched/corrected model is created in the same directory as the original model,
        # we do not need to prepare as much.
        self.check_accelerator_instance()

class SpsSegmentCreator(SegmentCreator, SpsModelCreator):
    _squence_name = "sps"

