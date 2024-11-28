"""
PS
--

Accelerator-Class for the ``PS`` machine.

Model Creation Keyword Args:
    *--Optional--*

    - **dpp** *(float)*:

        Deltap/p to use.

        default: ``0.0``


    - **driven_excitation** *(str)*:

        Denotes driven excitation by `AC-dipole` (acd) or by `ADT` (adt)

        choices: ``('acd', 'adt')``


    - **drv_tunes** *(float)*:

        Driven tunes without integer part.


    - **energy** *(float)*:

        Energy in **Tev**.


    - **model_dir** *(str)*:

        Path to model directory; loads tunes and excitation from model!


    - **modifiers** *(str)*:

        Path to the optics file to use (modifiers file).


    - **nat_tunes** *(float)*:

        Natural tunes without integer part.


    - **xing**:

        If True, x-ing angles will be applied to model

        action: ``store_true``
"""
import logging
from pathlib import Path

from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import (
    AccElementTypes,
)
from omc3.utils.parsertools import require_param
from omc3.model.constants import PLANE_TO_HV
from omc3.model.accelerators.psbase import PsBase

LOGGER = logging.getLogger(__name__)
CURRENT_DIR = Path(__file__).parent

# tune matching methods
# check `ps/base.madx`
TUNE_METHODS = {
    "qf": 1,  # main quadrupoles (low energy quads)
    "pfw": 2,  # pole face windings
    "bh": 3,  # combined function magnet quadrupole
    "f8l": 4,  # figure of eight loop
}

class Ps(PsBase):
    """Parent Class for PS-types."""

    NAME = "ps"
    RE_DICT = {AccElementTypes.BPMS: r"PR\.BPM",
               AccElementTypes.MAGNETS: r".*",
               AccElementTypes.ARC_BPMS: r"PR\.BPM"
               }

    # Public Methods ##########################################################
    @staticmethod
    def get_parameters():
        params = super(Ps, Ps).get_parameters()
        params.add_parameter(name="tune_method",
                             choices = list(TUNE_METHODS.keys()),
                             type=str, help="Tune method")
        return params

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)

        if opt.model_dir is not None:
            self.init_from_model_dir(opt.model_dir)
            return  # if we create the model from the model dir, none of the following needs to happen

        require_param("tune_method", Ps.get_parameters(), opt)
        self.tune_method = TUNE_METHODS[opt.tune_method]

    def verify_object(self):
        pass

    def get_file(self, filename: str) -> Path:
        """Get filepaths for PS files."""
        ps_dir = CURRENT_DIR / self.NAME
        ps_year_dir = self.get_dir()

        for dir_ in (ps_year_dir, ps_dir):
            file_path = dir_ / filename
            if file_path.is_file():
                return file_path

        raise NotImplementedError(
            f"File {file_path.name} not available for accelerator {self.NAME}."
        )

    def get_exciter_bpm(self, plane, bpms):
        if not self.excitation:
            return None
        bpms_to_find = ["PR.BPM00", "PR.BPM03"]
        found_bpms = [bpm for bpm in bpms_to_find if bpm in bpms]
        if not len(found_bpms):
            raise KeyError
        return (list(bpms).index(found_bpms[0]), found_bpms[0]), f"{PLANE_TO_HV[plane]}ACMAP"

