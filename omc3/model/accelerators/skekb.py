"""
SuperKEKB
---------

Accelerator-Class for the ``SuperKEKB`` machine.


Model Creation Keyword Args:
    *--Required--*

    - **ring** *(str)*:

        HER or LER ring.

        choices: ``('ler', 'her')``


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
from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import (Accelerator,
                                                 AcceleratorDefinitionError)
from omc3.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


class SKekB(Accelerator):
    """KEK's SuperKEKB accelerator."""
    NAME = "skekb"
    RINGS = ("ler", "her")

    @classmethod
    def get_parameters(cls):
        params = super(SKekB, SKekB).get_parameters()
        params.add_parameter(name="ring", type=str, choices=cls.RINGS, required=True,
                             help="HER or LER ring.")
        return params

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
        self.ring = opt.ring
        ring_to_beam_direction = {"ler": 1, "her": -1}
        self.beam_direction = ring_to_beam_direction[self.ring]

    @property
    def ring(self):
        if self._ring is None:
            raise AcceleratorDefinitionError("The accelerator definition is incomplete, ring "
                                             "has to be specified (--ring option missing?).")
        return self._ring

    @ring.setter
    def ring(self, value):
        if value not in self.RINGS:
            raise AcceleratorDefinitionError("Ring parameter has to be one of ('ler', 'her')")
        self._ring = value

    def verify_object(self):
        if self.model_dir is None:  # is the class is used to create full response?
            raise AcceleratorDefinitionError("SuperKEKB doesn't have a model creation, "
                                             "calling it this way is most probably wrong.")
