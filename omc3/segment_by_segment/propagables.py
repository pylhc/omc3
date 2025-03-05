"""
Segment by Segment: Propagables
-------------------------------

In this module, the propagables, i.e. the parameters that can be propagated
through the segment, are defined. 
Each propagable has a corresponding class, which contains the functions that
describe the forward and backward propagation for the respective parameter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tfs import TfsDataFrame

from omc3.definitions.constants import PLANE_TO_NUM, PLANES
from omc3.definitions.optics import OpticsMeasurement
from omc3.optics_measurements.constants import (
    ALPHA,
    BETA,
    NAME,
    PHASE,
    PHASE_ADV,
    S,
    TUNE,
)
from omc3.segment_by_segment import math
from omc3.segment_by_segment.definitions import Measurement
from omc3.segment_by_segment.definitions import (
    PropagableBoundaryConditions as BoundaryConditions,
)
from omc3.segment_by_segment.definitions import PropagableColumns as Columns
from omc3.segment_by_segment.segments import Segment, SegmentDiffs, SegmentModels
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
    IndexType = Sequence[str] | str | slice | pd.Index
    ValueErrorType = tuple[pd.Series, pd.Series] | tuple[float, float]

LOG = logging_tools.get_logger(__name__)


def get_all_propagables() -> tuple:
    """ Return all defined Propagables. """
    return Phase, BetaPhase, AlphaPhase


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
            alpha=Measurement(*AlphaPhase.get_at(self._segment.start, self._meas, plane)),
            beta=Measurement(*BetaPhase.get_at(self._segment.start, self._meas, plane))
        )
    
    def _init_end(self, plane: str) -> BoundaryConditions:
        """Get the end condition for all propagables at the given plane.
        Note: Alpha needs to be "reversed" as the end-condition is only used in backward
              propagation and alpha is anti-symmetric in time.
        """
        return BoundaryConditions(
            alpha=-Measurement(*AlphaPhase.get_at(self._segment.end, self._meas, plane)),
            beta=Measurement(*BetaPhase.get_at(self._segment.end, self._meas, plane))
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
            names = _common_indices(self.segment_models.forward.index,
                                    self._meas.total_phase[plane].index)  # measurement points/BPMs
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
            names = _common_indices(segment_elements, meas_phase.index)  # keep same order as segment

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
            total_err = _quadratic_add(meas_err, propagated_err)
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
            names = _common_indices(seg_model.index, beta.index)
            model_beta = model_beta.loc[names]
            model_phase = model_phase.loc[names]

            # calculate beta beating
            beta_beating = (beta - model_beta) / model_beta

            # propagate the error
            err_beta = err_beta / model_beta
            propagated_err = math.propagate_error_beta(model_beta, model_phase, init_condition)
            total_err = _quadratic_add(err_beta, propagated_err)
            return beta_beating, total_err
        else:
            prop_beta = model_beta.iloc[0]
            propagated_err = math.propagate_error_beta(prop_beta, model_phase.iloc[0], init_condition)
            return prop_beta, propagated_err


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


# Helper -----------------------------------------------------------------------

def _common_indices(*indices):
    """Common indices between the sets of given indices."""
    common = indices[0]
    for index in indices[1:]:
        common = common.intersection(index)
    return common


def _quadratic_add(*values):
    """Calculate the root-sum-squared of the given values.
    The individual "values" can be ``pd.Series`` and then their 
    elements are summed by indexs."""
    result = 0.
    for value in values:
        result += value ** 2
    return np.sqrt(result)
