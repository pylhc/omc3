
"""
Phase Propagable
----------------

This module contains the propagable for the phase parameter.
"""
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tfs import TfsDataFrame

from omc3.definitions.constants import PLANE_TO_NUM, PLANES
from omc3.definitions.optics import OpticsMeasurement
from omc3.optics_measurements.constants import (
    NAME,
    PHASE,
    PHASE_ADV,
    TUNE,
    S,
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


class Phase(Propagable):
    columns: Columns = Columns(PHASE)

    def init_conditions_dict(self):
        # The phase is not necessary for the initial conditions.
        return {}

    @classmethod
    def get_at(cls, names: IndexType, meas: OpticsMeasurement, plane: str) -> ValueErrorType:
        columns = cls.columns.planed(plane)
        phase = meas.total_phase[plane].loc[names, columns.column]
        error = meas.total_phase[plane].loc[names, columns.error_column]
        return phase, error

    def _get_s(self, names: IndexType, meas: OpticsMeasurement, plane: str) -> ValueErrorType:
        return meas.total_phase[plane].loc[names, S]

    @cache
    def measured_forward(self, plane):
        return self._compute_measured(
            plane, 
            self.segment_models.forward, 
            forward=True
        )

    @cache
    def correction_forward(self, plane):
        return self._compute_correction(
            plane,
            self.segment_models.forward,
            self.segment_models.forward_corrected,
            forward=True,
        )

    @cache
    def expected_forward(self, plane):
        return self._compute_measured(
            plane, 
            self.segment_models.forward_corrected, 
            forward=True
        )

    @cache
    def measured_backward(self, plane):
        return self._compute_measured(
            plane, 
            self.segment_models.backward, 
            forward=False
        )

    @cache
    def correction_backward(self, plane):
        return self._compute_correction(
            plane,
            self.segment_models.backward,
            self.segment_models.backward_corrected,
            forward=False,
        )
    
    @cache
    def expected_backward(self, plane):
        return self._compute_measured(
            plane, 
            self.segment_models.backward_corrected, 
            forward=False
        )

    def add_differences(self, segment_diffs: SegmentDiffs):
        """ Calculate the differences between the propagated models and the measured values."""
        for plane in PLANES:
            names = common_indices(
                self.segment_models.forward.index, self._meas.total_phase[plane].index
            )  # measurement points/BPMs
            c = self.columns.planed(plane)
            df = pd.DataFrame(index=names)
            df[NAME] = names
            df[S] = self.segment_models.forward.loc[names, S]

            phase, err_phase = Phase.get_at(names, self._meas, plane)
            df.loc[:, c.column] = math.phase_diff(phase, phase.iloc[0])
            df.loc[:, c.error_column] = err_phase

            phase, err_phase = self.measured_forward(plane)
            df.loc[:, c.forward] = phase
            df.loc[:, c.error_forward] = err_phase
            
            phase, err_phase = self.measured_backward(plane)
            df.loc[:, c.backward] = phase
            df.loc[:, c.error_backward] = err_phase

            if self.segment_models.get_path("forward_corrected").exists(): 
                phase, err_phase = self.correction_forward(plane)
                df.loc[:, c.forward_correction] = phase.loc[names]
                df.loc[:, c.error_forward_correction] = err_phase.loc[names]

                phase, err_phase = self.expected_forward(plane)
                df.loc[:, c.forward_expected] = phase
                df.loc[:, c.error_forward_expected] = err_phase

            if self.segment_models.get_path("backward_corrected").exists(): 
                phase, err_phase = self.correction_backward(plane)
                df.loc[:, c.backward_correction] = phase.loc[names]
                df.loc[:, c.error_backward_correction] = err_phase.loc[names]

                phase, err_phase = self.expected_backward(plane)
                df.loc[:, c.backward_expected] = phase
                df.loc[:, c.error_backward_expected] = err_phase

            # save to diffs/write to file (if allow_write is set)
            segment_diffs.phase[plane] = df

    def _compute_measured(self, plane: str, seg_model: TfsDataFrame, forward: bool) -> tuple[pd.Series, pd.Series]:
        """ Compute the difference between the given segment model and the measured values."""
        model_phase = seg_model.loc[:, self._model_column(plane)]
        tune = seg_model.headers[f"{TUNE}{PLANE_TO_NUM[plane]}"]

        init_condition = self._init_start(plane) if forward else self._init_end(plane)
        if not self._segment.element:
            # Segment
            meas_phase, meas_err = Phase.get_at(slice(None), self._meas, plane)  # slice(None) gives all, i.e. `:`
            
            # filter names for segment
            segment_elements = seg_model.index
            if not forward:
                segment_elements = pd.Index(reversed(segment_elements))
            names = common_indices(segment_elements, meas_phase.index)  # keep same order as segment

            meas_phase = meas_phase.loc[names]
            model_phase = model_phase.loc[names]

            # take care of circularity of accelerator (when the segment start is before first bpm in measurement)
            s = self._get_s(names, self._meas, plane)
            meas_phase = meas_phase - np.where(s > s.iloc[-1], tune, 0)

            # calculate phase with reference to segment (start/end)
            reference_element = names[0 if forward else -1]  # start of the propagation
            segment_model_phase = model_phase - model_phase.loc[reference_element]
            segment_meas_phase = meas_phase - meas_phase.loc[reference_element]
            if not forward:
                segment_model_phase = -segment_model_phase  # TODO: Why?

            phase_beating = math.phase_diff(segment_meas_phase, segment_model_phase)
            # propagate the error
            propagated_err = math.propagate_error_phase(model_phase, init_condition)
            total_err = math.quadratic_add(meas_err, propagated_err)
            return phase_beating, total_err

        else:
            # Element segment
            propagated_phase = model_phase.iloc[0]  # no measurement value exists at the element
            propagated_err = math.propagate_error_phase(propagated_phase, init_condition)
            return propagated_phase, propagated_err

    def _compute_correction(self, plane: str, seg_model: pd.DataFrame, seg_model_corr: pd.DataFrame, forward: bool) -> tuple[pd.Series, pd.Series]:
        """Compute the difference between the nominal and the corrected model."""
        model_phase = seg_model.loc[:, self._model_column(plane)]
        corrected_phase = seg_model_corr.loc[:, self._model_column(plane)]
        if self._segment.element:
            model_phase = model_phase.iloc[0]
            corrected_phase = corrected_phase.iloc[0]
        
        init_condition = self._init_start(plane)
        if forward: 
            phase_beating = math.phase_diff(corrected_phase, model_phase)
        else:
            phase_beating = math.phase_diff(model_phase, corrected_phase)
        propagated_err = math.propagate_error_phase(corrected_phase, init_condition)
        return phase_beating, propagated_err
    
    @staticmethod
    def _model_column(plane: str):
        """ Helper function to get the phase-column in the model, as it has a different name as in the measurement."""
        return f"{PHASE_ADV}{plane}"