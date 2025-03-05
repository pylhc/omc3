"""
Segment by Segment: Definitions 
-------------------------------

This module provides definitions to be used with segment by segment
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from uncertainties.core import Variable 
from omc3.utils.math_classes import MathMixin

from omc3.optics_measurements.constants import ERR
from omc3.segment_by_segment.constants import BACKWARD, CORRECTION, EXPECTED, FORWARD


class PropagableColumns:
    """ Class to define columns for propagables. 
    One could also implicitly define the error-columns,
    either via __getattr__ or as a wrapper, but I decided to
    explicitely name these columns, so that the IDEs can see them 
    and can help in renaming and autocompletion (jdilly 2023).
    """
    def __init__(self, column: str, plane: str = "{}") -> None:
        self._column = column
        self._plane = plane

    def planed(self, plane: str) -> PropagableColumns:
        return PropagableColumns(self._column, plane)

    @property
    def column(self):
        return f"{self._column}{self._plane}"
    
    @property
    def error_column(self):
        return f"{ERR}{self.column}"
    
    # Propagation ---
    @property
    def forward(self):
        return f"{FORWARD}{self.column}"
    
    @property
    def error_forward(self):
        return f"{ERR}{self.forward}"
    
    @property
    def backward(self):
        return f"{BACKWARD}{self.column}"
    
    @property
    def error_backward(self):
        return f"{ERR}{self.backward}"
    
    # Correction ---
    @property
    def forward_correction(self):
        return f"{CORRECTION}{self.forward}"

    @property
    def error_forward_correction(self):
        return f"{ERR}{self.forward_correction}"
        
    @property
    def backward_correction(self):
        return f"{CORRECTION}{self.backward}"
    
    @property
    def error_backward_correction(self):
        return f"{ERR}{self.backward_correction}"

    # Expectation --- 
    @property
    def forward_expected(self):
        return f"{EXPECTED}{self.forward}"

    @property
    def error_forward_expected(self):
        return f"{ERR}{self.forward_expected}"
        
    @property
    def backward_expected(self):
        return f"{EXPECTED}{self.backward}"
    
    @property
    def error_backward_expected(self):
        return f"{ERR}{self.backward_expected}"


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
class PropagableBoundaryConditions:
    """Store boundary conditions with error for propagating."""
    alpha: Measurement = None
    beta: Measurement = None
    dispersion: Measurement = None
    f1001_amplitude: Measurement = None
    f1001_phase: Measurement = None
    f1010_amplitude: Measurement = None
    f1010_phase: Measurement = None


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
