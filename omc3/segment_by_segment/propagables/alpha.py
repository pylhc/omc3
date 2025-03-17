
"""
Alpha Propagable
----------------

This module contains the propagable for the alpha parameter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from tfs import TfsDataFrame

from omc3.definitions.optics import OpticsMeasurement
from omc3.optics_measurements.constants import ALPHA, PHASE_ADV
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


class AlphaPhase(Propagable):

    _init_pattern = "alf{}_{}"
    columns: PropagableColumns = PropagableColumns(ALPHA)

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

    def get_segment_observation_points(self, plane: str):
        """ Return the measurement points for the given plane, that are in the segment. """
        return common_indices(
            self.segment_models.forward.index, 
            self._meas.beta_phase[plane].index
        )  

    def add_differences(self, segment_diffs: SegmentDiffs):
        """ Calculate the differences between the propagated models and the measured values."""
        dfs = self.get_difference_dataframes()
        for plane, df in dfs.items():
            # save to diffs/write to file (if allow_write is set)
            segment_diffs.alpha_phase[plane] = df

    def _compute_measured(self, 
            plane: str, 
            seg_model: TfsDataFrame, 
            forward: bool
        ) -> tuple[pd.Series, pd.Series]:
        """ Compute the beta-beating between the given segment model and the measured values."""
        init_condition = self._init_start(plane) if forward else self._init_end(plane)

        # get the measured values
        names = self.get_segment_observation_points(plane)
        alpha, err_alpha = self.get_at(names, self._meas, plane)

        # get the propagated values
        model_alpha = seg_model.loc[names, f"{ALPHA}{plane}"]
        model_phase = seg_model.loc[names, f"{PHASE_ADV}{plane}"]

        # calculate beta beating
        alpha_diff = alpha - model_alpha

        # propagate the error
        propagated_err = math.propagate_error_alpha(model_alpha, model_phase, init_condition)
        total_err = math.quadratic_add(err_alpha, propagated_err)
        return alpha_diff, total_err
    
    def _compute_correction(
            self,
            plane: str,
            seg_model: pd.DataFrame,
            seg_model_corr: pd.DataFrame,
            forward: bool,
        ) -> tuple[pd.Series, pd.Series]:
        """Compute the beta-beating between the nominal and the corrected model."""
        init_condition = self._init_start(plane) if forward else self._init_end(plane)

        model_alpha = seg_model.loc[:, f"{ALPHA}{plane}"]
        corrected_alpha = seg_model_corr.loc[:, f"{ALPHA}{plane}"]
        alpha_diff = corrected_alpha - model_alpha
        
        # propagate the error
        model_phase = seg_model.loc[:, f"{PHASE_ADV}{plane}"]
        propagated_err = math.propagate_error_alpha(corrected_alpha, model_phase, init_condition)
        return alpha_diff, propagated_err

    def _compute_elements(self, plane: str, seg_model: pd.DataFrame, forward: bool):
        """ Compute get the propagated beta values from the segment model and calculate the propagated error.  """
        init_condition = self._init_start(plane) if forward else self._init_end(plane)

        model_alpha = seg_model.loc[:, f"{ALPHA}{plane}"]

        # propagate the error
        model_phase = seg_model.loc[:, f"{PHASE_ADV}{plane}"]
        propagated_err = math.propagate_error_alpha(model_alpha, model_phase, init_condition)
        return model_alpha, propagated_err