from __future__ import annotations

from typing import TYPE_CHECKING

from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import AccElementTypes, Accelerator

if TYPE_CHECKING:
    from typing import ClassVar

class Generic(Accelerator):
    NAME = "generic"
    RE_DICT: ClassVar[dict[str, str]] = {
        AccElementTypes.BPMS: r"^B.*",
        AccElementTypes.MAGNETS: r".*",
        AccElementTypes.ARC_BPMS: r"^B.*",
    }

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
