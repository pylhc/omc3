
from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import AccElementTypes, Accelerator


class Fccee(Accelerator):
    NAME = "Fccee"
    RE_DICT = {AccElementTypes.BPMS: r"^.*",
               AccElementTypes.MAGNETS: r".*",
               AccElementTypes.ARC_BPMS: r"^.*",
               }

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
