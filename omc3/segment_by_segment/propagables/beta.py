
"""
Beta Propagable
---------------

This module contains the propagable for the beta parameter.
"""
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import pandas as pd

from omc3.definitions.optics import OpticsMeasurement
from omc3.optics_measurements.constants import (
    BETA,
    PHASE_ADV,
)
from omc3.segment_by_segment import math
from omc3.segment_by_segment.propagables.abstract import Propagable
from omc3.segment_by_segment.propagables.utils import PropagableColumns as Columns, common_indices
from omc3.segment_by_segment.segments import SegmentDiffs
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
    IndexType = Sequence[str] | str | slice | pd.Index
    ValueErrorType = tuple[pd.Series, pd.Series] | tuple[float, float]

LOG = logging_tools.get_logger(__name__)


class BetaPhase(Propagable):

    _init_pattern = "bet{}_{}"
    columns: Columns = Columns(BETA)

    @classmethod
    def get_at(cls, names: IndexType, meas: OpticsMeasurement, plane: str) -> ValueErrorType:
        c = cls.columns.planed(plane)
        beta = meas.beta_phase[plane].loc[names, c.column]
        error = meas.beta_phase[plane].loc[names, c.error_column]
        return beta, error

    @cache
    def measured_forward(self, plane):
        return self._compute_measured(plane, self.segment_models.forward, forward=True)

    @cache
    def correction_forward(self, plane):
        pass

    @cache
    def expected_forward(self, plane):
        pass

    @cache
    def measured_backward(self, plane):
        return self._compute_measured(plane, self.segment_models.backward, forward=False)

    @cache
    def correction_backward(self, plane):
        pass
    
    @cache
    def expected_backward(self, plane):
        pass
    
    def add_differences(self, segment_diffs: SegmentDiffs):
        pass

    def _compute_measured(self, plane: str, seg_model: pd.DataFrame, forward: bool):
        model_beta = seg_model.loc[:, f"{BETA}{plane}"]
        model_phase = seg_model.loc[:, f"{PHASE_ADV}{plane}"]
        init_condition = self._init_start(plane) if forward else self._init_end(plane)
        if not self._segment.element:
            beta, err_beta = BetaPhase.get_at(slice(None), self._meas, plane)

            # filter model indices
            names = common_indices(seg_model.index, beta.index)
            model_beta = model_beta.loc[names]
            model_phase = model_phase.loc[names]

            # calculate beta beating
            beta_beating = (beta - model_beta) / model_beta

            # propagate the error
            err_beta = err_beta / model_beta
            propagated_err = math.propagate_error_beta(model_beta, model_phase, init_condition)
            total_err = math.quadratic_add(err_beta, propagated_err)
            return beta_beating, total_err
        else:
            prop_beta = model_beta.iloc[0]
            propagated_err = math.propagate_error_beta(prop_beta, model_phase.iloc[0], init_condition)
            return prop_beta, propagated_err
