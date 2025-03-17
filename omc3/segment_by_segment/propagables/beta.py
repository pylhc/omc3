
"""
Beta Propagable
---------------

This module contains the propagable for the beta parameter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from tfs import TfsDataFrame

from omc3.definitions.optics import OpticsMeasurement
from omc3.optics_measurements.constants import BETA
from omc3.segment_by_segment import math
from omc3.segment_by_segment.propagables.abstract import Propagable
from omc3.segment_by_segment.propagables.phase import Phase
from omc3.segment_by_segment.propagables.utils import PropagableColumns, common_indices
from omc3.segment_by_segment.segments import SegmentDiffs
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
    IndexType = Sequence[str] | str | slice | pd.Index
    ValueErrorType = tuple[pd.Series, pd.Series] | tuple[float, float]

LOG = logging_tools.get_logger(__name__)


class BetaPhase(Propagable):

    _init_pattern = "bet{}_{}"
    columns: PropagableColumns = PropagableColumns(BETA)

    @classmethod
    def get_at(cls, names: IndexType, meas: OpticsMeasurement, plane: str) -> ValueErrorType:
        c = cls.columns.planed(plane)
        beta = meas.beta_phase[plane].loc[names, c.column]
        error = meas.beta_phase[plane].loc[names, c.error_column]
        return beta, error
    
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
            segment_diffs.beta_phase[plane] = df

    def _compute_measured(self, 
            plane: str, 
            seg_model: TfsDataFrame, 
            forward: bool
        ) -> tuple[pd.Series, pd.Series]:
        """ Compute the beta-beating between the given segment model and the measured values."""
        init_condition = self._init_start(plane) if forward else self._init_end(plane)

        # get the measured values
        names = self.get_segment_observation_points(plane)
        beta, err_beta = self.get_at(names, self._meas, plane)

        # get the propagated values
        model_beta = seg_model.loc[names, f"{BETA}{plane}"]
        model_phase = Phase.get_segment_phase(seg_model.loc[names, :], plane, forward) 

        # calculate beta beating
        beta_beating = (beta - model_beta) / model_beta

        # propagate the error
        err_beta = err_beta / model_beta
        propagated_err = math.propagate_error_beta(model_beta, model_phase, init_condition)
        total_err = math.quadratic_add(err_beta, propagated_err)
        return beta_beating, total_err
    
    def _compute_correction(
            self,
            plane: str,
            seg_model: pd.DataFrame,
            seg_model_corr: pd.DataFrame,
            forward: bool,
        ) -> tuple[pd.Series, pd.Series]:
        """Compute the beta-beating between the nominal and the corrected model."""
        init_condition = self._init_start(plane) if forward else self._init_end(plane)

        # calculate beta beating
        model_beta = seg_model.loc[:, f"{BETA}{plane}"]
        corrected_beta = seg_model_corr.loc[:, f"{BETA}{plane}"]
        beta_beating = (corrected_beta - model_beta) / model_beta

        # propagate the error
        model_phase = Phase.get_segment_phase(seg_model, plane, forward)
        propagated_err = math.propagate_error_beta(corrected_beta, model_phase, init_condition)
        return beta_beating, propagated_err

    def _compute_elements(self, plane: str, seg_model: pd.DataFrame, forward: bool):
        """ Compute get the propagated beta values from the segment model and calculate the propagated error.
        TODO: Should this be beta-beating with the nominal model (self._elements) ?
        """
        init_condition = self._init_start(plane) if forward else self._init_end(plane)
        
        model_beta = seg_model.loc[:, f"{BETA}{plane}"]

        # propagate the error
        model_phase = Phase.get_segment_phase(seg_model, plane, forward)
        propagated_err = math.propagate_error_beta(model_beta, model_phase, init_condition)
        return model_beta, propagated_err
