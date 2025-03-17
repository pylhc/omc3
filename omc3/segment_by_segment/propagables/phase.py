
"""
Phase Propagable
----------------

This module contains the propagable for the phase parameter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tfs import TfsDataFrame

from omc3.definitions.constants import PLANE_TO_NUM
from omc3.definitions.optics import OpticsMeasurement
from omc3.optics_measurements.constants import (
    PHASE,
    PHASE_ADV,
    TUNE,
    S,
)
from omc3.segment_by_segment import math
from omc3.segment_by_segment.propagables.abstract import Propagable
from omc3.segment_by_segment.propagables.utils import PropagableColumns, common_indices
from omc3.segment_by_segment.segments import SegmentDiffs
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
    IndexType = Sequence[str] | str | slice | pd.Index
    ValueErrorType = tuple[pd.Series, pd.Series] | tuple[float, float]

LOG = logging_tools.get_logger(__name__)


class Phase(Propagable):
    columns: PropagableColumns = PropagableColumns(PHASE)

    def init_conditions_dict(self):
        # The phase is not necessary for the initial conditions.
        return {}

    @classmethod
    def get_at(cls, names: IndexType, meas: OpticsMeasurement, plane: str) -> ValueErrorType:
        columns = cls.columns.planed(plane)
        phase = meas.total_phase[plane].loc[names, columns.column]
        error = meas.total_phase[plane].loc[names, columns.error_column]
        return phase, error
    
    def get_segment_observation_points(self, plane: str):
        """ Return the measurement points for the given plane, that are in the segment. """
        return common_indices(
            self.segment_models.forward.index, 
            self._meas.total_phase[plane].index
        )  

    def add_differences(self, segment_diffs: SegmentDiffs):
        """ Calculate the differences between the propagated models and the measured values."""
        dfs = self.get_difference_dataframes()
        for plane, df in dfs.items():
            # subtract reference phase
            columns = self.columns.planed(plane)
            phase = df.loc[:, columns.column]
            df.loc[:, columns.column] = math.phase_diff(phase, phase.iloc[0])
            
            # save to diffs/write to file (if allow_write is set)
            segment_diffs.phase[plane] = df

    def _compute_measured(self, 
            plane: str, 
            seg_model: TfsDataFrame, 
            forward: bool
        ) -> tuple[pd.Series, pd.Series]:
        """ Compute the difference between the given segment model and the measured values."""
        init_condition = self._init_start(plane) if forward else self._init_end(plane)

        # get the measured values
        names = self.get_segment_observation_points(plane)
        meas_phase, meas_err = Phase.get_at(names, self._meas, plane)
        
        # get the propagated values
        model_phase = seg_model.loc[names, self._model_column(plane)]
        tune = seg_model.headers[f"{TUNE}{PLANE_TO_NUM[plane]}"]

        # take care of circularity of accelerator (when the segment start is before first bpm in measurement)
        s = self._elements_model.loc[names, S]
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

    def _compute_correction(
            self,
            plane: str,
            seg_model: pd.DataFrame,
            seg_model_corr: pd.DataFrame,
            forward: bool,
        ) -> tuple[pd.Series, pd.Series]:
        """Compute the difference between the nominal and the corrected model."""
        model_phase = seg_model.loc[:, self._model_column(plane)]
        corrected_phase = seg_model_corr.loc[:, self._model_column(plane)]
        
        init_condition = self._init_start(plane)
        if forward: 
            phase_beating = math.phase_diff(corrected_phase, model_phase)
        else:
            phase_beating = math.phase_diff(model_phase, corrected_phase)
        propagated_err = math.propagate_error_phase(corrected_phase, init_condition)
        return phase_beating, propagated_err
    
    def _compute_elements(self, 
            plane: str, 
            seg_model: pd.DataFrame, 
            forward: bool
        ) -> tuple[pd.Series, pd.Series]:
        """ Compute get the propagated phase values from the segment model and calculate the propagated error."""
        model_phase = seg_model.loc[:, self._model_column(plane)]

        init_condition = self._init_start(plane) if forward else self._init_end(plane)
        propagated_err = math.propagate_error_phase(model_phase, init_condition)
        return model_phase, propagated_err

    @staticmethod
    def _model_column(plane: str) -> str:
        """ Helper function to get the phase-column in the model, as it has a different name as in the measurement."""
        return f"{PHASE_ADV}{plane}"