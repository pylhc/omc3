"""
Propagables
-----------

In the ``propagables`` module, the parameters that can be propagated
through the segment, are defined.
Each propagable has a corresponding class, which contains the functions that
describe the forward and backward propagation for the respective parameter.

This module exposes the main classes for propagables, to avoid making the
imports so long and complicated.
"""
from __future__ import annotations

from omc3.segment_by_segment.propagables.abstract import Propagable  # noqa: TC001 -> expose
from omc3.segment_by_segment.propagables.alpha import AlphaPhase
from omc3.segment_by_segment.propagables.dispersion import Dispersion, DispersionMomentum  #noqa  -> expose
from omc3.segment_by_segment.propagables.coupling import F1001, F1010  #noqa  -> expose


ALL_PROPAGABLES: tuple[type[Propagable], ...] = (Phase, BetaPhase, AlphaPhase, Dispersion, DispersionMomentum, F1001, F1010)
