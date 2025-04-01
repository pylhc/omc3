
"""
Coupling Propagables
--------------------

This module contains the propagables for the coupling parameters.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from optics_functions.coupling import coupling_via_cmatrix, rmatrix_from_coupling
from tfs import TfsDataFrame

from omc3.definitions.optics import OpticsMeasurement
from omc3.optics_measurements.constants import ALPHA as COL_ALPHA
from omc3.optics_measurements.constants import AMPLITUDE, IMAG, PHASE, REAL
from omc3.optics_measurements.constants import BETA as COL_BETA
from omc3.optics_measurements.constants import F1001 as COL_F1001
from omc3.optics_measurements.constants import F1010 as COL_F1010
from omc3.segment_by_segment import math
from omc3.segment_by_segment.propagables.abstract import Propagable
from omc3.segment_by_segment.propagables.alpha import AlphaPhase
from omc3.segment_by_segment.propagables.beta import BetaPhase
from omc3.segment_by_segment.propagables.phase import Phase
from omc3.segment_by_segment.propagables.utils import PropagableColumns, common_indices
from omc3.segment_by_segment.segments import SegmentDiffs, SegmentModels
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Callable
    IndexType = list[str] | str | slice | pd.Index
    ValueErrorType = tuple[pd.Series, pd.Series] | tuple[float, float]

LOG = logging_tools.get_logger(__name__)


COMPONENTS: tuple[str, str] = (REAL, IMAG, AMPLITUDE, PHASE)


# Base Class -------------------------------------------------------------------

class Coupling(Propagable):

    columns: PropagableColumns = PropagableColumns("")  # measured columns don't have prefix
    MODEL_COLUMN_PREFIX: str  # needs to be defined per RDT
    error_propagation_funcs: dict[str, Callable]  # Need to be defined per RDT

    @Propagable.segment_models.setter
    def segment_models(self, segment_models: SegmentModels):
        segment_models.forward = append_rdt_components(segment_models.forward, self.MODEL_COLUMN_PREFIX)
        segment_models.backward = append_rdt_components(segment_models.backward, self.MODEL_COLUMN_PREFIX)

        try:
            segment_models.forward_corrected = append_rdt_components(segment_models.forward_corrected, self.MODEL_COLUMN_PREFIX)
            segment_models.backward_corrected = append_rdt_components(segment_models.backward_corrected, self.MODEL_COLUMN_PREFIX)
        except FileNotFoundError:
            LOG.debug(f"No corrected segment models for {self._segment.name} to add {self.MODEL_COLUMN_PREFIX} to.")

        self._segment_models = segment_models
        
    def init_conditions_dict(self):
        elements = [self._segment.start, self._segment.end]

        # build data-frame to be used by rmatrix_from_coupling
        # Note: taking the measured alpha and beta here, as the propagated 
        #       values will also be based on the measured values, not on the model
        #       values. 
        df = pd.DataFrame({
            f"{COL_ALPHA}X": AlphaPhase.get_at(elements, self._meas, "X")[0],
            f"{COL_ALPHA}Y": AlphaPhase.get_at(elements, self._meas, "Y")[0],
            f"{COL_BETA}X": BetaPhase.get_at(elements, self._meas, "X")[0],
            f"{COL_BETA}Y": BetaPhase.get_at(elements, self._meas, "Y")[0],
            f"{COL_F1001}{REAL}": F1001.get_at(elements, self._meas, REAL)[0],
            f"{COL_F1001}{IMAG}": F1001.get_at(elements, self._meas, IMAG)[0],
            f"{COL_F1010}{REAL}": F1010.get_at(elements, self._meas, REAL)[0],
            f"{COL_F1010}{IMAG}": F1010.get_at(elements, self._meas, IMAG)[0],
        })
        
        # invert alpha at the end 
        df.loc[df.index[-1], [f"{COL_ALPHA}X", f"{COL_ALPHA}Y"]] = -df.loc[df.index[-1], [f"{COL_ALPHA}X", f"{COL_ALPHA}Y"]]
        
        rmatrix = rmatrix_from_coupling(df, complex_columns=False)
        return {
            f"{r_component}_{suffix}": rmatrix.loc[element, r_component.upper()] 
                for r_component in ("r11", "r12", "r21", "r22")
                for suffix, element in zip(("ini", "end"), elements)
        }

    def _compute_measured(
            self, 
            plane: str,  # is component here, i..e real, imag, amp, phase
            seg_model: TfsDataFrame, 
            forward: bool
        ) -> tuple[pd.Series, pd.Series]:
        """ Compute the coupling difference between the given segment model and the measured values."""
        # inits contain only rdts, as the error propagation is independent of alpha/beta/disp
        init_condition = self._init_start(None) if forward else self._init_end(None)

        # get the measured values
        names = self.get_segment_observation_points(plane)
        meas_value, meas_error = self.get_at(names, self._meas, plane)

        # get the propagated values
        model_column = f"{self.MODEL_COLUMN_PREFIX}{plane}"
        model_value = seg_model.loc[names, model_column]

        # calculate difference
        if plane == PHASE:
            value_diff = math.phase_diff(meas_value, model_value)
        else:
            value_diff = meas_value - model_value

        # propagate the error
        error_propagation = self.error_propagation_funcs[plane]
        model_phase_x = Phase.get_segment_phase(seg_model.loc[names, :], "X", forward) 
        model_phase_y = Phase.get_segment_phase(seg_model.loc[names, :], "Y", forward) 
        propagated_err = error_propagation(model_phase_x, model_phase_y, init_condition)
        total_err = math.quadratic_add(meas_error, propagated_err)
        return value_diff, total_err
    
    def _compute_correction(
            self,
            plane: str,  # is component here, i..e real, imag, amp, phase
            seg_model: pd.DataFrame,
            seg_model_corr: pd.DataFrame,
            forward: bool,
        ) -> tuple[pd.Series, pd.Series]:
        """Compute the coupling difference between the nominal and the corrected model."""
        init_condition = self._init_start(None) if forward else self._init_end(None)  # only coupling

        model_column = f"{self.MODEL_COLUMN_PREFIX}{plane}"
        model_rdt = seg_model.loc[:, model_column]
        corrected_rdt = seg_model_corr.loc[:, model_column]

        if plane == PHASE:
            model_diff = math.phase_diff(corrected_rdt, model_rdt)
        else:
            model_diff = corrected_rdt - model_rdt
        
        # propagate the error
        error_propagation = self.error_propagation_funcs[plane]
        model_phase_x = Phase.get_segment_phase(seg_model, "X", forward) 
        model_phase_y = Phase.get_segment_phase(seg_model, "Y", forward) 
        propagated_err = error_propagation(model_phase_x, model_phase_y, init_condition)
        return model_diff, propagated_err

    def _compute_elements(
            self, 
            plane: str,  # is component here, i..e real, imag, amp, phase
            seg_model: pd.DataFrame, 
            forward: bool
        ) -> tuple[pd.Series, pd.Series]:
        """ Compute get the propagated coupling values from the segment model and calculate the propagated error.  """
        init_condition = self._init_start(None) if forward else self._init_end(None)  # only coupling

        model_column = f"{self.MODEL_COLUMN_PREFIX}{plane}"
        model_value = seg_model.loc[:, model_column]

        # propagate the error
        error_propagation = self.error_propagation_funcs[plane]
        model_phase_x = Phase.get_segment_phase(seg_model, "X", forward) 
        model_phase_y = Phase.get_segment_phase(seg_model, "Y", forward) 
        propagated_err = error_propagation(model_phase_x, model_phase_y, init_condition)
        return model_value, propagated_err
    
    @classmethod
    def is_rdt(cls) -> bool:
        """ Tells the caller, if the propagable is an RDT. 
        The columns and filenames need to be handled differently in that case. """
        return True 
    

# Coupling RDTs ----------------------------------------------------------------  

class F1001(Coupling):
    """ Propagable for the F1001 parameter. 
    
    Hint: We use the "plane" parameter to determine the components, 
    i.e. real, imaginary, amplitude or phase.
    """
    MODEL_COLUMN_PREFIX = COL_F1001
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

    @classmethod
    def in_measurement(cls, meas: OpticsMeasurement) -> bool:
        """ Check if the coupling rdt is in the measurement data. """
        try:
            meas.f1001
        except FileNotFoundError:
            return False
        return True
    
    def get_segment_observation_points(self, _: str):
        """ Return the measurement points for the given plane, that are in the segment. """
        return common_indices(
            self.segment_models.forward.index, 
            self._meas.f1001.index
        )  

    def add_differences(self, segment_diffs: SegmentDiffs):
        """ Calculate the differences between the propagated models and the measured values."""
        dfs = self.get_difference_dataframes(planes=COMPONENTS)
        segment_diffs.f1001 = concat_dfs_dict_no_duplicates(dfs)


class F1010(Coupling):
    """ Propagable for the F1010 parameter. 
    
    Hint: We use the "plane" parameter to determine the components, 
    i.e. real, imaginary, amplitude or phase.
    """
    MODEL_COLUMN_PREFIX = COL_F1010
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
    
    @classmethod
    def in_measurement(cls, meas: OpticsMeasurement) -> bool:
        """ Check if the coupling rdt is in the measurement data. """
        try:
            meas.f1010
        except FileNotFoundError:
            return False
        return True
    
    def get_segment_observation_points(self, _: str):
        """ Return the measurement points for the given plane, that are in the segment. """
        return common_indices(
            self.segment_models.forward.index, 
            self._meas.f1010.index
        )  

    def add_differences(self, segment_diffs: SegmentDiffs):
        """ Calculate the differences between the propagated models and the measured values."""
        dfs = self.get_difference_dataframes(planes=COMPONENTS)
        segment_diffs.f1010 = concat_dfs_dict_no_duplicates(dfs)


# Helper functions -------------------------------------------------------------

def append_rdt_components(seg_model: pd.DataFrame | None, rdt: str):
    """ Append the RDT components to the segment model."""
    if seg_model is None:
        return None

    rdt_series = coupling_via_cmatrix(seg_model, output=("rdts",))[rdt]

    seg_model[f"{rdt}{AMPLITUDE}"] = np.abs(rdt_series)
    seg_model[f"{rdt}{REAL}"] = np.real(rdt_series)
    seg_model[f"{rdt}{IMAG}"] = np.imag(rdt_series)
    seg_model[f"{rdt}{PHASE}"] = (np.angle(rdt_series) / (2*np.pi)) % 1 
    return seg_model


def concat_dfs_dict_no_duplicates(dfs: dict[str, pd.DataFrame]):
    """ Concatenate the dataframes in the dictionary without duplicate columns. """
    df = pd.concat(dfs.values(), axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df
