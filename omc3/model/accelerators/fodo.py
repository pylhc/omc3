from generic_parser import EntryPoint
from omc3.model.accelerators.accelerator import (
    AccElementTypes,
    Accelerator,
)


class Fodo(Accelerator):
    """Parent Class for RDT fodo lattice."""
    NAME = "fodo"
    RE_DICT = {AccElementTypes.BPMS: r"BPM*",
               AccElementTypes.MAGNETS: r".*",
               AccElementTypes.ARC_BPMS: r"BPM*"
               }

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
