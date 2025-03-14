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
from omc3.segment_by_segment.propagables.abstract import Propagable  # probably only used for type-hinting
from omc3.segment_by_segment.propagables.utils import PropagableBoundaryConditions   #noqa  -> expose
from omc3.segment_by_segment.propagables.utils import PropagableColumns   #noqa  -> expose
from omc3.segment_by_segment.propagables.phase import Phase
from omc3.segment_by_segment.propagables.beta import BetaPhase
from omc3.segment_by_segment.propagables.alpha import AlphaPhase

def get_all_propagables() -> tuple[Propagable, ...]:
    """ Return all defined Propagables. """
    return Phase, BetaPhase, AlphaPhase
