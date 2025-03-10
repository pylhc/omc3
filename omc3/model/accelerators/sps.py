""" 
SPS
---

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omc3.model.accelerators.accelerator import (
    AccElementTypes,
    Accelerator,
    find_called_files_with_tag,
)
from omc3.model.constants import JOB_MODEL_MADX_NOMINAL
from omc3.utils import logging_tools
from omc3.utils.iotools import load_multiple_jsons
from omc3.utils.knob_list_manipulations import get_vars_by_classes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from generic_parser import DotDict

LOGGER = logging_tools.get_logger(__name__)

class Sps(Accelerator):
    """ SPS accelerator. """
    NAME: str = "sps"
    LOCAL_REPO_NAME: str = "acc-models-sps"
    RE_DICT: dict[str, str] = {
        AccElementTypes.BPMS: r"BP.*",
        AccElementTypes.MAGNETS: r"M.*",
        AccElementTypes.ARC_BPMS: r"BP.*",
    }
    STRENGTH_FILE_TAG: str = "!@StrengthFile"

    @staticmethod
    def get_parameters():
        params = super().get_parameters()
        params.add_parameter(name="year", type=str, help="Optics tag.")
        params.add_parameter(name="str_file", type=str, help="Strength File")
        return params

    def __init__(self, opt: DotDict):
        super().__init__(opt)
        self.year: str | None = opt.year
        self.str_file: str | None = opt.str_file
    
    def get_variables(self, frm: float | None = None, to: float | None = None, classes: Iterable[str] | None = None):
        correctors_file = self.get_file("correctors.json")
        all_vars_by_class = load_multiple_jsons(correctors_file)
        knobs_to_use = get_vars_by_classes(classes, all_vars_by_class)
        return knobs_to_use
    
    def init_from_model_dir(self, model_dir: Path) -> None:
        super().init_from_model_dir(model_dir)
        
        job_file = model_dir / JOB_MODEL_MADX_NOMINAL
        str_file = find_called_files_with_tag(job_file, self.STRENGTH_FILE_TAG)
        if str_file is not None:
            self.str_file = str_file[0].name  # assumes we always use acc-models, so we don't need the full path
