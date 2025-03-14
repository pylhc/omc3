
"""
Abstract Propagable
-------------------
In this module the abstract class for a propagable is defined.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
from typing import TYPE_CHECKING

import pandas as pd

from omc3.definitions.constants import PLANES
from omc3.definitions.optics import OpticsMeasurement
from omc3.segment_by_segment import propagables  # don't import alpha/beta etc directly ! cyclic imports !
from omc3.segment_by_segment.definitions import Measurement
from omc3.segment_by_segment.propagables.utils import (
    PropagableBoundaryConditions as BoundaryConditions,
)
from omc3.segment_by_segment.segments import Segment, SegmentDiffs, SegmentModels
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
    IndexType = Sequence[str] | str | slice | pd.Index
    ValueErrorType = tuple[pd.Series, pd.Series] | tuple[float, float]

LOG = logging_tools.get_logger(__name__)

class Propagable(ABC):
    _init_pattern: str  # see init_conditions_dict

    def __init__(self, segment: Segment, meas: OpticsMeasurement):
        self._segment: Segment = segment
        self._meas: OpticsMeasurement = meas
        self._segment_models: SegmentModels = None

    @property
    def segment_models(self):
        """TfsCollection of the segment models."""
        if self._segment_models is None:
            raise ValueError("Segment_models have not been set.")
        return self._segment_models

    @segment_models.setter
    def segment_models(self, segment_models: SegmentModels):
        self._segment_models = segment_models

    def init_conditions_dict(self):
        """Return a dictionary containing the initial values for this propagable at start and end
        of the segment.

        For the naming, see `save_initial_and_final_values` macro in 
        `omc3/model/madx_macros/general.macros.madx`.
        """
        if self._init_pattern is None:
            raise NotImplementedError(
                f"Class {self.__class__.__name__} has no ``_init_pattern`` implemented."
                f"Contact a developer."
            )

        init_dict = {}
        for plane in PLANES:
            # get start value
            start_cond, _ = self.get_at(self._segment.start, self._meas, plane)
            start_name = self._init_pattern.format(plane, "ini")
            init_dict[start_name] = start_cond

            # get end value
            end_cond, _ = self.get_at(self._segment.end, self._meas, plane)
            end_name = self._init_pattern.format(plane, "end")
            init_dict[end_name] = end_cond
        return init_dict
    
    def _init_start(self, plane: str) -> BoundaryConditions:
        """Get the start condition for all propagables at the given plane."""
        return BoundaryConditions(
            alpha=Measurement(*propagables.AlphaPhase.get_at(self._segment.start, self._meas, plane)),
            beta=Measurement(*propagables.BetaPhase.get_at(self._segment.start, self._meas, plane))
        )
    
    def _init_end(self, plane: str) -> BoundaryConditions:
        """Get the end condition for all propagables at the given plane.
        Note: Alpha needs to be "reversed" as the end-condition is only used in backward
              propagation and alpha is anti-symmetric in time.
        """
        return BoundaryConditions(
            alpha=-Measurement(*propagables.AlphaPhase.get_at(self._segment.end, self._meas, plane)),
            beta=Measurement(*propagables.BetaPhase.get_at(self._segment.end, self._meas, plane))
        )
    
    @classmethod
    @abstractmethod
    def get_at(cls, names: IndexType, measurement: OpticsMeasurement, plane: str
               ) -> ValueErrorType:
        """Get corresponding measurement values at the elements ``names``

        Args:
            names: element name(s)
            measurement: Measurement Collection
            plane: plane to use

        Returns:
            Series or float containing the required values at ``names``.
        """
        ...

    @cache
    @abstractmethod
    def measured_forward(self, plane: str):
        """Interpolation of measured deviations to forward propagated model."""
        ...

    @cache
    @abstractmethod
    def measured_backward(self, plane: str):
        """Interpolation of measured deviations to backward propagated model."""
        ...
    
    @cache
    @abstractmethod
    def expected_forward(self, plane: str):
        """Interpolation of measured deviations to corrected forward propagated model."""
        ...

    @cache
    @abstractmethod
    def expected_backward(self, plane: str):
        """Interpolation of measured deviations to corrected backward propagated model."""
        ...

    @cache
    @abstractmethod
    def correction_forward(self, plane: str):
        """Deviations between forward propagated models with and without correction."""
        ...

    @cache
    @abstractmethod
    def correction_backward(self, plane: str):
        """Deviations between backward propagated models with and without correction."""
        ...
    
    @abstractmethod
    def add_differences(self, segment_diffs: SegmentDiffs):
        """This function calculates the differences between the propagated 
        forward and backward models and the measured values.
        It then adds the results to the segment_diffs class 
        (which writes them out, if its ``allow_write`` is set to ``True``)."""
        ...
