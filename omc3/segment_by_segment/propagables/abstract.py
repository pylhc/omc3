
"""
Abstract Propagable
-------------------
In this module the abstract class for a propagable is defined.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
from typing import TYPE_CHECKING

import pandas as pd

from omc3.definitions.constants import PLANES
from omc3.definitions.optics import OpticsMeasurement
from omc3.optics_measurements.constants import AMPLITUDE, NAME, PHASE, S_MODEL, S
from omc3.segment_by_segment import (
    propagables,  # don't import alpha/beta etc directly ! cyclic imports !
)
from omc3.segment_by_segment.math import SegmentBoundaryConditions, Measurement
from omc3.segment_by_segment.propagables.utils import PropagableColumns
from omc3.segment_by_segment.segments import Segment, SegmentDiffs, SegmentModels
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tfs import TfsDataFrame
    IndexType = Sequence[str] | str | slice | pd.Index
    ValueErrorType = tuple[pd.Series, pd.Series] | tuple[float, float]

LOG = logging_tools.get_logger(__name__)

class Propagable(ABC):
    _init_pattern: str  # see init_conditions_dict
    columns: PropagableColumns 

    def __init__(self, segment: Segment, meas: OpticsMeasurement, twiss_elements: TfsDataFrame):
        """ 
        Abstract class to define a propagable, i.e. a parameter that can be/has been 
        propagated through a segment.

        Args:
            segment: The segment to propagate through.
            meas: The OpticsMeasurement that contains the measured values.
            twiss_elements: The twiss-elements model of the full machine.
        """
        self._segment: Segment = segment
        self._meas: OpticsMeasurement = meas
        self._elements_model: TfsDataFrame = twiss_elements
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
    
    def _init_start(self, plane: str) -> SegmentBoundaryConditions:
        """Get the start condition for all propagables at the given plane."""
        return self._get_boundary_condition_at(self._segment.start, plane)
    
    def _init_end(self, plane: str) -> SegmentBoundaryConditions:
        """Get the end condition for all propagables at the given plane.
        Note: Alpha needs to be "reversed" as the end-condition is only used in backward
              propagation and alpha is anti-symmetric in time.
        """
        conditions = self._get_boundary_condition_at(self._segment.end, plane)
        if conditions.alpha is not None:
            conditions.alpha = -conditions.alpha
        return conditions

    def _get_boundary_condition_at(self, position, plane: str | None) -> SegmentBoundaryConditions:
        conditions = SegmentBoundaryConditions(
            f1001_amplitude=Measurement(*propagables.F1001.get_at(position, self._meas, AMPLITUDE)),
            f1001_phase=Measurement(*propagables.F1001.get_at(position, self._meas, PHASE)),
            f1010_amplitude=Measurement(*propagables.F1010.get_at(position, self._meas, AMPLITUDE)),
            f1010_phase=Measurement(*propagables.F1010.get_at(position, self._meas, PHASE)),
        )
        if plane is not None:
            conditions.alpha = Measurement(*propagables.AlphaPhase.get_at(position, self._meas, plane))
            conditions.beta = Measurement(*propagables.BetaPhase.get_at(position, self._meas, plane))
            conditions.dispersion = Measurement(*propagables.Dispersion.get_at(position, self._meas, plane))
        return conditions

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

    @abstractmethod
    def get_segment_observation_points(self, plane: str):
        """Return the measurement points for the given plane, that are in the segment. """
        ...

    def get_difference_dataframes(self, planes: Sequence[str] = PLANES) -> dict[str, pd.DataFrame]:
        """Compute the difference dataframes between the propagated models and the measured values.
        
        As the naming conventions of the columns are not intuitive, when not working with 
        segment-by-segment, here the detailed explanations:

        NAME-Column: The element/observaiton point (BPM) names
        S-Column: The segment model longitudinal value, starting with 0 from the start of the segment.
        S_MODEL-Column: The longitudinal value of the twiss model, starting with 0 from the start of the accelerator.

        Parameter-Column (+ Error):
            The measured value of the parameter, i.e. the same value as in the optics analysis output

        Forward/Backward-Columns (+ Error):
            The DIFFERENCE of the forward/backward propagated value of the parameter to the measured values. 
            In case of Beta, the beating is calculated.
            The error is a combination of the measured error at the element and the propagated error.
        
        Correction-Columns (+ Error):
            The DIFFERENCE of the forward/backward propagated value through the corrected model 
            to the forward/backward propagated value through the nominal model. 
            This compares the two segment models with each other and shows how well the corrected model
            now matches the measured values.
            We want the difference between them to be as close as possible to the Forward/Backward-Column.
            The error is the propagated error from the forward model.

        Expected-Columns (+ Error):
            The DIFFERENCE of the forward/backward propagated value through the corrected model to the measured values. 
            This represents the expected difference between measurement and model after correction, 
            hence we want this value to be as close to zero as possible.
            The error is a combination of the measured error at the element and the propagated error.
        """
        dfs = {}
        for plane in planes:
            names = self.get_segment_observation_points(plane)
            columns = self.columns.planed(plane)
            df = pd.DataFrame(index=names)
            df[NAME] = names
            df[S] = self.segment_models.forward.loc[names, S]
            df[S_MODEL] = self._elements_model.loc[names, S]

            meas_val, meas_err = self.get_at(names, self._meas, plane)
            df.loc[:, columns.column] = meas_val
            df.loc[:, columns.error_column] = meas_err

            meas_val, meas_err = self.measured_forward(plane)
            df.loc[:, columns.forward] = meas_val
            df.loc[:, columns.error_forward] = meas_err
            
            meas_val, meas_err = self.measured_backward(plane)
            df.loc[:, columns.backward] = meas_val
            df.loc[:, columns.error_backward] = meas_err

            if self.segment_models.get_path("forward_corrected").exists(): 
                meas_val, meas_err = self.correction_forward(plane)
                df.loc[:, columns.forward_correction] = meas_val.loc[names]
                df.loc[:, columns.error_forward_correction] = meas_err.loc[names]

                meas_val, meas_err = self.expected_forward(plane)
                df.loc[:, columns.forward_expected] = meas_val
                df.loc[:, columns.error_forward_expected] = meas_err

            if self.segment_models.get_path("backward_corrected").exists(): 
                meas_val, meas_err = self.correction_backward(plane)
                df.loc[:, columns.backward_correction] = meas_val.loc[names]
                df.loc[:, columns.error_backward_correction] = meas_err.loc[names]

                meas_val, meas_err = self.expected_backward(plane)
                df.loc[:, columns.backward_expected] = meas_val
                df.loc[:, columns.error_backward_expected] = meas_err

            dfs[plane] = df
        return dfs
    
    @abstractmethod
    def add_differences(self, segment_diffs: SegmentDiffs):
        """This function calculates the differences between the propagated 
        forward and backward models and the measured values.
        It then adds the results to the segment_diffs class 
        (which writes them out, if its ``allow_write`` is set to ``True``)."""
        ...

    @cache
    def measured_forward(self, plane: str) -> tuple[pd.Series, pd.Series]:
        """Interpolation of measured deviations to forward propagated model."""
        return self._compute_measured(
            plane, 
            self.segment_models.forward, 
            forward=True
        )

    @cache
    def measured_backward(self, plane: str) -> tuple[pd.Series, pd.Series]:
        """Interpolation of measured deviations to backward propagated model."""
        return self._compute_measured(
            plane, 
            self.segment_models.backward, 
            forward=False
        )
    
    @cache
    def expected_forward(self, plane: str) -> tuple[pd.Series, pd.Series]:
        """Interpolation of measured deviations to corrected forward propagated model."""
        return self._compute_measured(
            plane, 
            self.segment_models.forward_corrected, 
            forward=True
        )

    @cache
    def expected_backward(self, plane: str) -> tuple[pd.Series, pd.Series]:
        """Interpolation of measured deviations to corrected backward propagated model."""
        return self._compute_measured(
            plane, 
            self.segment_models.backward_corrected, 
            forward=False
        )

    @cache
    def correction_forward(self, plane: str) -> tuple[pd.Series, pd.Series]:
        """Deviations between forward propagated models with and without correction."""
        return self._compute_correction(
            plane,
            self.segment_models.forward,
            self.segment_models.forward_corrected,
            forward=True,
        )

    @cache
    def correction_backward(self, plane: str) -> tuple[pd.Series, pd.Series]:
        """Deviations between backward propagated models with and without correction."""
        return self._compute_correction(
            plane,
            self.segment_models.backward,
            self.segment_models.backward_corrected,
            forward=False,
        )
    
    def _compute_measured(self, 
            plane: str, 
            seg_model: TfsDataFrame, 
            forward: bool
        ) -> tuple[pd.Series, pd.Series]:
        """ Compute the difference between the given segment model and the measured values."""
        raise NotImplementedError  # only needs to be implemented, if inherited class uses functions declared above (or similar)
    
    def _compute_correction(
            self,
            plane: str,
            seg_model: pd.DataFrame,
            seg_model_corr: pd.DataFrame,
            forward: bool,
        ) -> tuple[pd.Series, pd.Series]:
        """Compute the difference between the nominal and the corrected model."""
        raise NotImplementedError  # only needs to be implemented, if inherited class uses functions declared above (or similar)
    
    def _compute_elements(self, 
            plane: str, 
            seg_model: pd.DataFrame, 
            forward: bool
        ) -> tuple[pd.Series, pd.Series]:
        """ Compute get the propagated phase values from the segment model and calculate the propagated error."""
        raise NotImplementedError  # only needs to be implemented, if inherited class uses functions declared above (or similar)
