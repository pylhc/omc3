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

from omc3.model.accelerators.accelerator import Accelerator, AccElementTypes, AccExcitationMode
from omc3.model.constants import PLANE_TO_HV, MODIFIER_TAG

LOGGER = logging.getLogger(__name__)
CURRENT_DIR = Path(__file__).parent


class Ps(Accelerator):
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
        params.add_parameter(
            name="year",
            type=int,
            default=2021,
            choices=(2018, 2021),
            help="Year of the optics.",
        )
        return params

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
        self.year = opt.year

    def verify_object(self):
        pass

    def get_dir(self):
        return CURRENT_DIR / self.NAME / str(self.year)

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

    def get_base_madx_script(self, best_knowledge=False):
        if best_knowledge:
            raise NotImplementedError(f"Best knowledge model not implemented for accelerator {self.NAME}")

        use_acd = self.excitation == AccExcitationMode.ACD
        replace_dict = {
            "FILES_DIR": str(self.get_dir()),
            "USE_ACD": str(int(use_acd)),
            "NAT_TUNE_X": self.nat_tunes[0],
            "NAT_TUNE_Y": self.nat_tunes[1],
            "KINETICENERGY": self.energy,
            "DRV_TUNE_X": "",
            "DRV_TUNE_Y": "",
            "MODIFIERS": "",
            "BEAM_FILE": "",  # From 2021
        }
        if self.modifiers:
            replace_dict["MODIFIERS"] = '\n'.join([f" call, file = '{m}'; {MODIFIER_TAG}" for m in self.modifiers])
            replace_dict["BEAM_FILE"] = '\n'.join([f" call, file = '{m.with_suffix('.beam')}';" for m in self.modifiers])
        if use_acd:
            replace_dict["DRV_TUNE_X"] = self.drv_tunes[0]
            replace_dict["DRV_TUNE_Y"] = self.drv_tunes[1]
        mask = self.get_file('base.mask').read_text()
        return mask % replace_dict


class _PsSegmentMixin(object):

    def __init__(self):
        self._start = None
        self._end = None
        self.energy = None
