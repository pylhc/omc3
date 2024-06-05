
from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import AccElementTypes, Accelerator


class Generic(Accelerator):
    NAME = "generic"
    RE_DICT = {AccElementTypes.BPMS: r"B",
               AccElementTypes.MAGNETS: r".*",
               AccElementTypes.ARC_BPMS: r"B",
               }

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
