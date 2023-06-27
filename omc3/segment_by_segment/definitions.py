"""
Segment by Segment: Definitions 
-------------------------------

This module provides definitions to be used with segment by segment
"""
from omc3.segment_by_segment.constants import BACKWARD, CORRECTED, FORWARD
from omc3.optics_measurements.constants import ALPHA, BETA, ERR, NAME, PHASE, PHASE_ADV, S


class PropagableColumns:
    """ Class to define columns for propagables. 
    One could also implicitly defile the error-columns,
    either via __getattr__ or as a wrapper, but I decided to
    explicityly define these columns, so that the IDEs can see them 
    and can help in renaming and autocompletion (jdilly 2023).
    """
    def __init__(self, column: str, plane: str = "{}") -> None:
        self._column = column
        self._plane = plane

    def planed(self, plane: str) -> "PropagableColumns":
        return PropagableColumns(self._column, plane)

    @property
    def column(self):
        return f"{self._column}{self._plane}"
    
    @property
    def error_column(self):
        return f"{ERR}{self.column}"
    
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
    
    @property
    def forward_corrected(self):
        return f"{CORRECTED}{self.forward}"

    @property
    def error_forward_corrected(self):
        return f"{ERR}{self.forward_corrected}"
        
    @property
    def backward_corrected(self):
        return f"{CORRECTED}{self.backward}"
    
    @property
    def error_backward_corrected(self):
        return f"{ERR}{self.backward_corrected}"
