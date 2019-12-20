"""
ESRF
-------------------
"""
from generic_parser import EntryPoint

from model.accelerators.accelerator import Accelerator, AccElementTypes


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
