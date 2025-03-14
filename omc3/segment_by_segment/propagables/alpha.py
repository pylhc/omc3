
"""
Alpha Propagable
----------------

This module contains the propagable for the alpha parameter.
"""
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import pandas as pd

from omc3.definitions.optics import OpticsMeasurement
from omc3.optics_measurements.constants import ALPHA
from omc3.segment_by_segment.propagables.abstract import Propagable 
from omc3.segment_by_segment.propagables.utils import PropagableColumns as Columns
from omc3.segment_by_segment.segments import SegmentDiffs
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
    IndexType = Sequence[str] | str | slice | pd.Index
    ValueErrorType = tuple[pd.Series, pd.Series] | tuple[float, float]

LOG = logging_tools.get_logger(__name__)


class AlphaPhase(Propagable):

    _init_pattern = "alf{}_{}"
    columns: Columns = Columns(ALPHA)

    @classmethod
    def get_at(cls, names: IndexType, meas: OpticsMeasurement, plane: str) -> ValueErrorType:
        c = cls.columns.planed(plane)
        alpha = meas.beta_phase[plane].loc[names, c.column]
        error = meas.beta_phase[plane].loc[names, c.error_column]
        return alpha, error
    
    def init_conditions_dict(self):
        # alpha needs to be inverted for backward propagation, i.e. the end-init
        init_cond = super().init_conditions_dict()
        for key, value in init_cond.items():
            if "end" in key:
                init_cond[key] = -value
        return init_cond

    @cache
    def measured_forward(self, plane):
        pass

    @cache
    def correction_forward(self, plane):
        pass

    @cache
    def expected_forward(self, plane):
        pass

    @cache
    def measured_backward(self, plane):
        pass

    @cache
    def correction_backward(self, plane):
        pass

    @cache
    def expected_backward(self, plane):
        pass
    
    def add_differences(self, segment_diffs: SegmentDiffs):
        pass
