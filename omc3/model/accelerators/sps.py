""" 
SPS
---

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import (
    AccElementTypes,
    Accelerator,
)
from omc3.model.constants import STRENGTHS_SUBDIR
from omc3.utils import logging_tools
from omc3.utils.iotools import load_multiple_jsons, find_file
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
        AccElementTypes.BPMS: r"^BP.*",
        AccElementTypes.MAGNETS: r".*",  # has a variety of names...
        AccElementTypes.ARC_BPMS: r"^BP.*",
    }

    @staticmethod
    def get_parameters():
        params = Accelerator.get_parameters()
        params.add_parameter(name="year", type=str, help="Optics tag.")
        return params

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
    
    def get_variables(self, frm: float | None = None, to: float | None = None, classes: Iterable[str] | None = None):
        correctors_file = self.get_file("correctors.json")
        all_vars_by_class = load_multiple_jsons(correctors_file)
        knobs_to_use = get_vars_by_classes(classes, all_vars_by_class)
        return knobs_to_use
    
    def init_from_options(self, opt: DotDict) -> None:
        super().init_from_options(opt)
        self.year: str | None = opt.year
        if opt.drv_tunes is not None:
            self.drv_tunes = opt.drv_tunes  # allow giving them even if ACD is deactivated (see madx job-file).
    
    def find_modifier(self, modifier: Path | str):
        """ Try to find a modifier file, which might be given only by its name. 
        This is looking for full-path, model-dir and in the acc-models-path's strength-dir.,
        """
        dirs = []
        if self.model_dir is not None:
            dirs.append(self.model_dir)

        if self.acc_model_path is not None:
            dirs.append(Path(self.acc_model_path) / STRENGTHS_SUBDIR)

        return find_file(modifier, dirs=dirs)
