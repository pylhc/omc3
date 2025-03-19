
"""
Coupling Propagables
--------------------

This module contains the propagables for the coupling parameters.
"""
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tfs import TfsDataFrame
import tfs

from omc3.definitions.optics import OpticsMeasurement
from omc3.optics_measurements.constants import AMPLITUDE, IMAG, PHASE, REAL, F1010 as COL_F1010, F1001 as COL_F1001 
from omc3.segment_by_segment import math
from omc3.segment_by_segment.propagables.abstract import Propagable
from omc3.segment_by_segment.propagables.phase import Phase 
from omc3.segment_by_segment.propagables.utils import PropagableColumns, common_indices
from omc3.segment_by_segment.segments import SegmentDiffs
from omc3.utils import logging_tools
from optics_functions.coupling import coupling_via_cmatrix, rmatrix_from_coupling 

if TYPE_CHECKING:
    from collections.abc import Sequence, Callable
    IndexType = Sequence[str] | str | slice | pd.Index
    ValueErrorType = tuple[pd.Series, pd.Series] | tuple[float, float]

LOG = logging_tools.get_logger(__name__)


COMPONENTS: tuple[str, str] = (REAL, IMAG, AMPLITUDE, PHASE)


# Base Class -------------------------------------------------------------------

class Coupling(Propagable):

    columns: PropagableColumns = PropagableColumns("")  # coupling columns don't have prefix
    error_propagation_funcs: dict[str, Callable]  # Need to be defined per RDT

    def __init__(self, segment, meas, twiss_elements):
        super().__init__(segment, meas, twiss_elements)

        rdt = self.__class__.__name__
        self.segment_models.forward = append_rdt_components(self.segment_models.forward, rdt)
        self.segment_models.backward = append_rdt_components(self.segment_models.backward, rdt)
        self.segment_models.forward_corrected = append_rdt_components(self.segment_models.forward_corrected, rdt)
        self.segment_models.backward_corrected = append_rdt_components(self.segment_models.backward_corrected, rdt)
        
    def init_conditions_dict(self):
        return init_conditions_dict(self._segment.start, self._segment.end, self._meas)

    def _compute_measured(self, 
            plane: str, 
            seg_model: TfsDataFrame, 
            forward: bool
        ) -> tuple[pd.Series, pd.Series]:
        """ Compute the dispersion difference between the given segment model and the measured values."""
        init_condition = self._init_start(plane) if forward else self._init_end(plane)

        # get the measured values
        names = self.get_segment_observation_points(plane)
        meas_value, meas_error = self.get_at(names, self._meas, plane)

        # get the propagated values
        model_value = seg_model.loc[names, plane]

        # calculate difference
        if plane == PHASE:
            disp_diff = math.phase_diff(meas_value, model_value)
        else:
            disp_diff = meas_value - model_value

        # propagate the error
        error_propagation = self.error_propagation_funcs[plane]
        model_phase_x = Phase.get_segment_phase(seg_model.loc[names, :], "X", forward) 
        model_phase_y = Phase.get_segment_phase(seg_model.loc[names, :], "Y", forward) 
        propagated_err = error_propagation(model_phase_x, model_phase_y, init_condition)
        total_err = math.quadratic_add(meas_error, propagated_err)
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

        model_disp = seg_model.loc[:, plane]
        corrected_disp = seg_model_corr.loc[:, plane]
        if plane == PHASE:
            model_diff = math.phase_diff(corrected_disp, model_disp)
        else:
            model_diff = corrected_disp - model_disp
        
        # propagate the error
        error_propagation = self.error_propagation_funcs[plane]
        model_phase_x = Phase.get_segment_phase(seg_model, "X", forward) 
        model_phase_y = Phase.get_segment_phase(seg_model, "Y", forward) 
        propagated_err = error_propagation(model_phase_x, model_phase_y, init_condition)
        return model_diff, propagated_err

    def _compute_elements(self, plane: str, seg_model: pd.DataFrame, forward: bool):
        """ Compute get the propagated dispersion values from the segment model and calculate the propagated error.  """
        init_condition = self._init_start(plane) if forward else self._init_end(plane)

        model_value = seg_model.loc[:, plane]

        # propagate the error
        error_propagation = self.error_propagation_funcs[plane]
        model_phase_x = Phase.get_segment_phase(seg_model, "X", forward) 
        model_phase_y = Phase.get_segment_phase(seg_model, "Y", forward) 
        propagated_err = error_propagation(model_phase_x, model_phase_y, init_condition)
        return model_value, propagated_err
    

# Coupling RDTs ----------------------------------------------------------------  

class F1001(Coupling):
    """ Propagable for the F1001 parameter. 
    
    Hint: We use the "plane" parameter to determine the components, 
    i.e. real, imaginary, amplitude or phase.
    """
    error_propagation_funcs: dict[str, Callable] = {
        REAL: math.propagate_error_coupling_1001_re,
        IMAG: math.propagate_error_coupling_1001_im,
        AMPLITUDE: math.propagate_error_coupling_1001_amp,
        PHASE: math.propagate_error_coupling_1001_phase
    }
    
    @classmethod
    def get_at(cls, names: IndexType, meas: OpticsMeasurement, plane: str) -> ValueErrorType:
        c = cls.columns.planed(plane)
        value = meas.f1001.loc[names, c.column]
        error = meas.f1001.loc[names, c.error_column]
        return value, error
    
    def get_segment_observation_points(self):
        """ Return the measurement points for the given plane, that are in the segment. """
        return common_indices(
            self.segment_models.forward.index, 
            self._meas.f1001.index
        )  

    def add_differences(self, segment_diffs: SegmentDiffs):
        """ Calculate the differences between the propagated models and the measured values."""
        dfs = self.get_difference_dataframes(planes=COMPONENTS)
        segment_diffs.f1001 = tfs.concat(dfs)


class F1010(Coupling):
    """ Propagable for the F1010 parameter. 
    
    Hint: We use the "plane" parameter to determine the components, 
    i.e. real, imaginary, amplitude or phase.
    """
    error_propagation_funcs: dict[str, Callable] = {
        REAL: math.propagate_error_coupling_1010_re,
        IMAG: math.propagate_error_coupling_1010_im,
        AMPLITUDE: math.propagate_error_coupling_1010_amp,
        PHASE: math.propagate_error_coupling_1010_phase
    }

    @classmethod
    def get_at(cls, names: IndexType, meas: OpticsMeasurement, plane: str) -> ValueErrorType:
        c = cls.columns.planed(plane)
        value = meas.f1010.loc[names, c.column]
        error = meas.f1010.loc[names, c.error_column]
        return value, error
    
    def get_segment_observation_points(self):
        """ Return the measurement points for the given plane, that are in the segment. """
        return common_indices(
            self.segment_models.forward.index, 
            self._meas.f1010.index
        )  

    def add_differences(self, segment_diffs: SegmentDiffs):
        """ Calculate the differences between the propagated models and the measured values."""
        dfs = self.get_difference_dataframes(planes=COMPONENTS)
        segment_diffs.f1010 = tfs.concat(dfs)


# Helper functions -------------------------------------------------------------

def append_rdt_components(seg_model: pd.DataFrame | None, rdt: str):
    """ Append the RDT components to the segment model."""
    if seg_model is None:
        return None

    rdt_df = coupling_via_cmatrix(seg_model, output=("rdts",))
    seg_model[AMPLITUDE] = np.abs(rdt_df[rdt])
    seg_model[REAL] = np.real(rdt_df[rdt])
    seg_model[IMAG] = np.imag(rdt_df[rdt])
    phase = np.angle(rdt_df[rdt]) / (2*np.pi)
    seg_model[PHASE] = np.where(phase > 0.5, phase - 1, phase)
    return seg_model


@cache  # will only be run once for both Coupling-RDTs
def init_conditions_dict(start: str, end: str, meas: OpticsMeasurement):
    """ Return a dictionary with the initial conditions for the RDTs in R-Matrix form. """
    elements = (start, end)
    f1001_real, _ = F1001.get_at(elements, meas, REAL)
    f1001_imag, _ = F1001.get_at(elements, meas, IMAG)
    f1010_real, _ = F1010.get_at(elements, meas, REAL)
    f1010_imag, _ = F1010.get_at(elements, meas, IMAG)
    rmatrix = rmatrix_from_coupling(
        pd.DataFrame(
            [f1001_real, f1001_imag, f1010_real, f1010_imag],
            index=[f"{COL_F1001}{REAL}", f"{COL_F1001}{IMAG}", f"{COL_F1010}{REAL}", f"{COL_F1010}{IMAG}"]), 
        complex_columns=False)
    return {
        f"{r_component}_{suffix}": rmatrix.loc[element, r_component.upper()] 
            for r_component in ("r11", "r12", "r21", "r22")
            for suffix, element in zip(("start", "end"), elements)
    }