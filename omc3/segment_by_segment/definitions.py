"""
Definitions 
-----------

This module provides definitions to be used with segment by segment
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from uncertainties.core import Variable 
from omc3.utils.math_classes import MathMixin


class Measurement(MathMixin, Variable):
    """ Alias for a uncertainties variable.
    
    Additionally has a method to convert it to a tuple and iterate over it.
    """
    
    def __init__(self, *args, **kwargs):
        try: 
            variable_tuple = args[0].nominal_value, args[0].std_dev
        except (IndexError, AttributeError):
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*variable_tuple)
    
    def as_tuple(self) -> tuple[float, float]:
        return (self.nominal_value, self.std_dev)
    
    def __iter__(self):
        return iter(self.as_tuple())


@dataclass 
class MadXBoundaryConditions:
    """Store all boundary conditions for a Mad-X twiss."""
    alfx: float = None
    alfy: float = None
    betx: float = None
    bety: float = None
    dx: float = None
    dy: float = None
    dpx: float = None
    dpy: float = None
    wx: float = None
    wy: float = None
    dphix: float = None
    dphiy: float = None
    r11: float = None
    r12: float = None
    r21: float = None
    r22: float = None

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self) 
                if getattr(self, f.name) is not None}
