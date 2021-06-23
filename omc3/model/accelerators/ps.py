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

from omc3.model.accelerators.accelerator import Accelerator, AccElementTypes
from omc3.model.constants import PLANE_TO_HV

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


class _PsSegmentMixin(object):

    def __init__(self):
        self._start = None
        self._end = None
        self.energy = None
