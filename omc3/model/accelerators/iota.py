"""
Iota
----

Accelerator-Class for the ``Iota`` machine.

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


    - **particle** *(str)*:

        Particle type.

        choices: ``('p', 'e')``


    - **xing**:

        If True, x-ing angles will be applied to model

        action: ``store_true``
"""
import logging
import os

from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import AccElementTypes, Accelerator

LOGGER = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


class Iota(Accelerator):
    NAME = "iota"
    RE_DICT = {AccElementTypes.BPMS: r"IBPM*",
               AccElementTypes.MAGNETS: r"Q*",
               AccElementTypes.ARC_BPMS: r"IBPM*"}
    BPM_INITIAL = 'I'

    @staticmethod
    def get_parameters():
        params = super(Iota, Iota).get_parameters()
        params.add_parameter(name="particle", type=str, choices=('p', 'e'), help="Particle type.")
        return params

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
        self.particle = opt.particle

    @classmethod
    def verify_object(self):
        pass  # TODO
