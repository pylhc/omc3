""" 
Utilities for Propagables
-------------------------

This module contains utilities for the ``propagables`` module,
and functions that are common to multiple propagables.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from omc3.optics_measurements.constants import ERR
from omc3.segment_by_segment.constants import BACKWARD, CORRECTION, EXPECTED, FORWARD
from omc3.segment_by_segment.definitions import Measurement


# Functions --------------------------------------------------------------------

def common_indices(*indices: pd.Index) -> pd.Index:
    """Common indices between the sets of given indices."""
    common = indices[0]
    for index in indices[1:]:
        common = common.intersection(index)
    return common


# Classes ----------------------------------------------------------------------

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

