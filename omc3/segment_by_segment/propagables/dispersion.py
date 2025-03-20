
"""
Dispersion Propagable
---------------------

This module contains the propagable for the dispersion parameter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from tfs import TfsDataFrame

from omc3.definitions.optics import OpticsMeasurement
from omc3.optics_measurements.constants import DISPERSION
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


class Dispersion(Propagable):

    _init_pattern = "d{}_{}"  # format(plane, ini/end)
    columns: PropagableColumns = PropagableColumns(DISPERSION)

    @classmethod
    def get_at(cls, names: IndexType, meas: OpticsMeasurement, plane: str) -> ValueErrorType:
        c = cls.columns.planed(plane)
        dispersion = meas.dispersion[plane].loc[names, c.column]
        error = meas.dispersion[plane].loc[names, c.error_column]
        return dispersion, error
    
    def get_segment_observation_points(self, plane: str):
        """ Return the measurement points for the given plane, that are in the segment. """
        return common_indices(
            self.segment_models.forward.index, 
            self._meas.dispersion[plane].index
        )  

    def add_differences(self, segment_diffs: SegmentDiffs):
        """ Calculate the differences between the propagated models and the measured values."""
        dfs = self.get_difference_dataframes()
        for plane, df in dfs.items():
            # save to diffs/write to file (if allow_write is set)
            segment_diffs.dispersion[plane] = df

    def _compute_measured(self, 
            plane: str, 
            seg_model: TfsDataFrame, 
            forward: bool
        ) -> tuple[pd.Series, pd.Series]:
        """ Compute the dispersion difference between the given segment model and the measured values."""
        init_condition = self._init_start(plane) if forward else self._init_end(plane)

        # get the measured values
        names = self.get_segment_observation_points(plane)
        disp, err_disp = self.get_at(names, self._meas, plane)

        # get the propagated values
        model_disp = seg_model.loc[names, f"{DISPERSION}{plane}"]
        model_phase = Phase.get_segment_phase(seg_model.loc[names, :], plane, forward) 

        # calculate difference
        disp_diff = disp - model_disp

        # propagate the error
        propagated_err = math.propagate_error_dispersion(model_disp, model_phase, init_condition)
        total_err = math.quadratic_add(err_disp, propagated_err)
        return disp_diff, total_err
    
    def _compute_correction(
            self,
            plane: str,
            seg_model: pd.DataFrame,
            seg_model_corr: pd.DataFrame,
            forward: bool,
        ) -> tuple[pd.Series, pd.Series]:
        """Compute the dispersion difference between the nominal and the corrected model."""
        init_condition = self._init_start(plane) if forward else self._init_end(plane)

        model_disp = seg_model.loc[:, f"{DISPERSION}{plane}"]
        corrected_disp = seg_model_corr.loc[:, f"{DISPERSION}{plane}"]
        disp_diff = corrected_disp - model_disp
        
        # propagate the error
        model_phase = Phase.get_segment_phase(seg_model, plane, forward)
        propagated_err = math.propagate_error_dispersion(corrected_disp, model_phase, init_condition)
        return disp_diff, propagated_err

    def _compute_elements(self, plane: str, seg_model: pd.DataFrame, forward: bool):
        """ Compute get the propagated dispersion values from the segment model and calculate the propagated error.  """
        init_condition = self._init_start(plane) if forward else self._init_end(plane)

        model_disp = seg_model.loc[:, f"{DISPERSION}{plane}"]

        # propagate the error
        model_phase = Phase.get_segment_phase(seg_model, plane, forward)
        propagated_err = math.propagate_error_dispersion(model_disp, model_phase, init_condition)
        return model_disp, propagated_err
