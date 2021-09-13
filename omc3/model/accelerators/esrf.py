"""
ESRF
----

Accelerator-Class for the ``ESRF`` machine.

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
from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import AccElementTypes, Accelerator


class Esrf(Accelerator):
    NAME = "esrf"
    RE_DICT = {AccElementTypes.BPMS: r"BPM",
               AccElementTypes.MAGNETS: r".*",
               AccElementTypes.ARC_BPMS: r"BPM\.(\d*[02468]\.[1-5]|\d*[13579]\.[3-7])",
               }  # bpms 1-5 in even cells and bpms 3-7 in odd cells.

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
