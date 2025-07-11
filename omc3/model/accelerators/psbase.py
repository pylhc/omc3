from __future__ import annotations

from typing import TYPE_CHECKING

from omc3.model.accelerators.accelerator import Accelerator

if TYPE_CHECKING:
    from generic_parser import DotDict

class PsBase(Accelerator):
    """ Base class for Ps and PsBooster"""
    NAME = None

    @staticmethod
    def get_parameters():
        params = super(PsBase, PsBase).get_parameters()
        params.add_parameter(name="year", type=str, help="Optics tag.")
        params.add_parameter(name="scenario", type=str, help="Scenario.")
        params.add_parameter(name="cycle_point", type=str, help="Cycle Point.")
        params.add_parameter(name="str_file", type=str, help="Strength File")
        return params

    def __init__(self, opt: DotDict):
        super().__init__(opt)
        self.year: str | None = opt.year
        self.scenario: str | None = opt.scenario
        self.cycle_point: str | None = opt.cycle_point
        self.beam_file: str | None = None
        self.str_file: str | None = opt.str_file
