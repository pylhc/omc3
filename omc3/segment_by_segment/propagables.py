"""
Segment by Segment: Propagables
-------------------------------

In this module, the propagables, i.e. the parameters that can be propagated
through the segment, are defined. 
Each prpagable has a corresponding class, which contains the functions that
describe the forward and backward propagation for the respective parameter.
"""
from abc import ABC, abstractmethod
from functools import lru_cache as cache  # in 3.9 one could use functools.cache
from typing import Any, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from omc3.definitions.constants import PLANES
from omc3.definitions.optics import OpticsMeasurement
from omc3.optics_measurements.constants import ALPHA, BETA, ERR, NAME, PHASE, PHASE_ADV, S
from omc3.segment_by_segment import math
from omc3.segment_by_segment.segments import Segment, SegmentDiffs, SegmentModels
from omc3.segment_by_segment.definitions import PropagableColumns as Columns
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


def get_all_propagables() -> Tuple:
    """ Return all defined Propagables. """
    return Phase, BetaPhase, AlfaPhase


IndexType = Union[Sequence[str], str, slice, pd.Index]
ValueErrorType = Union[Tuple[pd.Series, pd.Series], Tuple[float, float]]


class Propagable(ABC):
    _init_pattern: str  # see init_conditions_dict

    def __init__(self, segment: Segment, meas: OpticsMeasurement):
        self._segment: Segment = segment
        self._meas: OpticsMeasurement = meas
        self._segment_models: SegmentModels = None

        # Save initial conditions per plane:
        self.beta0, self.alpha0, self.errbeta0, self.erralpha0 = {}, {}, {}, {}
        for plane in PLANES:
            self.beta0[plane], self.errbeta0[plane] = BetaPhase.get_at(self._segment.start, meas, plane)
            self.alpha0[plane], self.erralpha0[plane] = AlfaPhase.get_at(self._segment.start, meas, plane)

    @property
    def segment_models(self):
        """TfsCollection of the segment models."""
        if self._segment_models is None:
            raise ValueError("segment_models have not been set.")
        return self._segment_models

    @segment_models.setter
    def segment_models(self, segment_models: SegmentModels):
        self._segment_models = segment_models

    def init_conditions_dict(self):
        """Return a dictionary containing the inital values at start and end
        of the segment.

        For the naming, see `save_initial_and_final_values` macro in 
        `omc3/model/madx_macros/geeral.macros.madx`.
        """
        if self._init_pattern is None:
            raise NotImplementedError(
                f"Class {self.__class__.__name__} has no ``_init_pattern`` implemented."
                f"Contact a developer."
            )

        init_dict = {}
        for plane in PLANES:
            # get start value
            init_cond, _ = self.get_at(self._segment.start, self._meas, plane)
            init_name = self._init_pattern.format(plane, "ini")

            # get end value
            init_dict[init_name] = init_cond
            end_cond, _ = self.get_at(self._segment.end, self._meas, plane)
            end_name = self._init_pattern.format(plane, "end")
            init_dict[end_name] = end_cond
        return init_dict

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
        """Interpolation or measured deviations to forward propagated model."""
        ...

    @cache
    @abstractmethod
    def measured_backward(self, plane: str):
        """Interpolation or measured deviations to backward propagated model."""
        ...

    @cache
    @abstractmethod
    def corrected_forward(self, plane: str):
        """Interpolation or corrected deviations to forward propagated model."""
        ...

    @cache
    @abstractmethod
    def corrected_backward(self, plane: str):
        """Interpolation or corrected deviations to backward propagated model."""
        ...

    @abstractmethod
    def add_differences(self):
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

    @cache
    def measured_forward(self, plane):
        return self._compute_measured(plane, self._segment_models.forward, 1)

    @cache
    def corrected_forward(self, plane):
        return self._compute_corrected(plane,
                                       self.segment_models.forward,
                                       self.segment_models.forward_corrected)

    @cache
    def measured_backward(self, plane):
        return self._compute_measured(plane, self._segment_models.backward, -1)

    @cache
    def corrected_backward(self, plane):
        return self._compute_corrected(plane,
                                       self.segment_models.backward,
                                       self.segment_models.backward_corrected)

    def add_differences(self, segment_diffs: SegmentDiffs):
        for plane in PLANES:
            names = _common_indices(self.segment_models.forward.index,
                                    self._meas.total_phase[plane].index)
            c = self.columns.planed(plane)
            df = pd.DataFrame(index=names)
            df[NAME] = names
            df[S] = self.segment_models.forward.loc[names, S]

            meas_ph, err_meas_ph = Phase.get_at(names, self._meas, plane)
            df.loc[:, c.column] = meas_ph
            df.loc[:, c.error_column] = err_meas_ph

            phs, err_phs = self.measured_forward(plane)
            df.loc[:, c.forward] = phs
            df.loc[:, c.error_forward] = err_phs
            
            phs, err_phs = self.measured_backward(plane)
            df.loc[:, c.backward] = phs
            df.loc[:, c.error_backward] = err_phs

            if self.segment_models.get_path("forward_corrected").exists(): 
                phs, err_phs = self.corrected_forward(plane)
                df.loc[:, c.forward_corrected] = phs
                df.loc[:, c.error_forward_corrected] = err_phs

            if self.segment_models.get_path("backward_corrected").exists(): 
                phs, err_phs = self.corrected_backward(plane)
                df.loc[:, c.backward_corrected] = phs
                df.loc[:, c.error_backward_corrected] = err_phs

            # ============== Plot for Debugging ================================
            # import matplotlib.pyplot as plt
            # df.loc[:, f"{FORWARD}{PHASE}{plane}"].plot()
            # plt.show()
            # ==================================================================

            # save to diffs/write to file (if allow_write is set)
            segment_diffs.phase[plane] = df

    def _compute_measured(self, plane, seg_model, sign):
        model_phase = seg_model.loc[:, f"{PHASE_ADV}{plane}"]
        init_condition = self.beta0[plane], self.errbeta0[plane], self.alpha0[plane], self.erralpha0[plane]
        if not self._segment.element:
            # Segment
            meas_phase, meas_err = Phase.get_at(slice(None), self._meas, plane)  # slice(None) gives all, i.e. `:`
            names = _common_indices(seg_model.index, meas_phase.index)
            model_phase = model_phase.loc[names]

            # calculate phase beating
            segment_meas_phase = sign * (meas_phase - meas_phase.iloc[0]) % 1.
            phase_beating = (segment_meas_phase - model_phase) % 1.
            phase_beating[phase_beating > 0.5] = phase_beating[phase_beating > 0.5] - 1

            # propagate the error
            propagated_err = math.propagate_error_phase(model_phase, *init_condition)
            total_err = _quadratic_add(meas_err, propagated_err)
            return phase_beating, total_err
        else:
            # Element segment
            propagated_phase = model_phase.iloc[0]
            propagated_err = math.propagate_error_phase(propagated_phase, *init_condition)
            return propagated_phase, propagated_err

    def _compute_corrected(self, plane, seg_model, seg_model_corr):
        model_phase = seg_model.loc[:, f"{PHASE_ADV}{plane}"]
        corrected_phase = seg_model_corr.loc[:, f"{PHASE_ADV}{plane}"]
        init_condition = self.beta0[plane], self.errbeta0[plane], self.alpha0[plane], self.erralpha0[plane]
        if not self._segment.element:
            phase_beating = (corrected_phase - model_phase) % 1.
            propagated_err = math.propagate_error_phase(model_phase, *init_condition)
            return phase_beating, propagated_err
        else:
            propagated_phase = model_phase.iloc[0]
            propagated_err = math.propagate_error_phase(propagated_phase, *init_condition)
            return propagated_phase, propagated_err


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
        return self._compute_measured(plane, self.segment_models.forward)

    @cache
    def corrected_forward(self, plane):
        pass

    @cache
    def measured_backward(self, plane):
        return self._compute_measured(plane, self.segment_models.backward)

    @cache
    def corrected_backward(self, plane):
        pass
    
    def add_differences(self, segment_diffs: SegmentDiffs):
        pass

    def _compute_measured(self, plane, seg_model):
        model_beta = seg_model.loc[:, f"{BETA}{plane}"]
        model_phase = seg_model.loc[:, f"{PHASE_ADV}{plane}"]
        init_condition = self.beta0[plane], self.errbeta0[plane], self.alpha0[plane], self.erralpha0[plane]
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
            propagated_err = math.propagate_error_beta(model_beta, model_phase, *init_condition)
            total_err = _quadratic_add(err_beta, propagated_err)
            return beta_beating, total_err
        else:
            prop_beta = model_beta.iloc[0]
            propagated_err = math.propagate_error_beta(prop_beta, model_phase.iloc[0], *init_condition)
            return prop_beta, propagated_err


class AlfaPhase(Propagable):

    _init_pattern = "alf{}_{}"
    columns: Columns = Columns(ALPHA)

    @classmethod
    def get_at(cls, names: IndexType, meas: OpticsMeasurement, plane: str) -> ValueErrorType:
        c = cls.columns.planed(plane)
        beta = meas.beta_phase[plane].loc[names, c.column]
        error = meas.beta_phase[plane].loc[names, c.error_column]
        return beta, error

    @cache
    def measured_forward(self, plane):
        pass

    @cache
    def corrected_forward(self, plane):
        pass

    @cache
    def measured_backward(self, plane):
        pass

    @cache
    def corrected_backward(self, plane):
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
    """Calculate the root-sum-squared of the given values."""
    return np.sqrt((np.array(values) ** 2).sum())
