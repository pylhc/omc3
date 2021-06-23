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
    YEAR = 2018
    RE_DICT = {AccElementTypes.BPMS: r"PR\.BPM",
               AccElementTypes.MAGNETS: r".*",
               AccElementTypes.ARC_BPMS: r"PR\.BPM"
               }

    # Public Methods ##########################################################

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)

    def verify_object(self):
        pass

    @classmethod
    def get_dir(cls):
        return CURRENT_DIR / cls.NAME / str(cls.YEAR)

    def get_exciter_bpm(self, plane, bpms):
        if not self.excitation:
            return None
        bpms_to_find = ["PR.BPM00", "PR.BPM03"]
        found_bpms = [bpm for bpm in bpms_to_find if bpm in bpms]
        if not len(found_bpms):
            raise KeyError
        return (list(bpms).index(found_bpms[0]), found_bpms[0]), f"{PLANE_TO_HV[plane]}ACMAP"

    def get_base_madx_script(self, model_directory, best_knowledge=False):
        if best_knowledge:
            raise NotImplementedError(f"Best knowledge model not implemented for accelerator {self.NAME}")

        use_acd = str(int(self.excitation == AccExcitationMode.ACD)),
        replace_dict = {
            "FILES_DIR": str(self.get_dir()),
            "USE_ACD": use_acd,
            "NAT_TUNE_X": self.nat_tunes[0],
            "NAT_TUNE_Y": self.nat_tunes[1],
            "KINETICENERGY": self.energy,
            "DRV_TUNE_X": "",
            "DRV_TUNE_Y": "",
            "MODIFIERS": "",
        }
        if self.modifiers:
            replace_dict["MODIFIERS"] = '\n'.join([f" call, file = '{m}'; {MODIFIER_TAG}" for m in self.modifiers])
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
