from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tfs.collection import Tfs, TfsCollection

from omc3.optics_measurements.constants import (
    AMP_BETA_NAME,
    BETA_NAME,
    DISPERSION_NAME,
    EXT,
    KMOD_BETA_NAME,
    NORM_DISP_NAME,
    PHASE_NAME,
)
from omc3.segment_by_segment.constants import (
    TWISS_BACKWARD,
    TWISS_BACKWARD_CORRECTED,
    TWISS_FORWARD,
    TWISS_FORWARD_CORRECTED,
)


@ dataclass
class Segment:
    name: str
    start: str
    end: str
    element: str = None
    init_conds: str = None

    @classmethod
    def init_from_element_name(cls, element_name: str):
        """ Initialize from the string representation for elements 
        as used in inputs."""
        segment = cls(element_name, element_name, element_name)
        segment.element = element_name
        return segment
    
    @classmethod
    def init_from_segment_definition(cls, segment: str):
        """ Initialize from the string representation for segments
        as used in inputs."""
        return cls(*segment.split(","))
    
    @classmethod
    def init_from_input(cls, input_str: str):
        """ Initialize from the string representation for segments or elements
        as used in inputs."""
        try:
            return cls.init_from_segment_definition(input_str)
        except ValueError:
            return cls.init_from_element_name(input_str)

    def to_input_string(self):
        """ String representation of the segment as used in inputs."""
        if self.element is not None:
            return self.element
        return f"{self.name},{self.start},{self.end}"

    def __str__(self):
        """String representation of the segment."""
        return f"{self.name} ({self.start} - {self.end})"


class SegmentModels(TfsCollection):
    """
    Class to hold and load the models of the segments created by MAD-X.
    The filenames need to be the same as in 
    :class:`omc3.model.model_creators.abstract_model_creator.SegmentCreator`.

    Arguments:
        directory: The path where to find the models.
        segment: A segment instance corresponding to the model to load.
    """
    forward = Tfs(TWISS_FORWARD, two_planes=False)
    backward = Tfs(TWISS_BACKWARD, two_planes=False)
    forward_corrected = Tfs(TWISS_FORWARD_CORRECTED, two_planes=False)
    backward_corrected = Tfs(TWISS_BACKWARD_CORRECTED, two_planes=False)

    def __init__(self, directory: Path, segment: Segment):
        super(SegmentModels, self).__init__(directory)
        self.segment = segment

    def _get_filename(self, template: str):
        return template.format(self.segment.name)


class SegmentDiffs(TfsCollection):
    """
    TfsCollection of segment-by-segment outputfiles for the differences
    between propagated model and measurements.

    Arguments:
        directory: The path where to write the files to/find the files.
        segment_name: Name of the segment corresponding to the model to load.
    """
    PREFIX = "sbs_"

    phase = Tfs(f"{PREFIX}{PHASE_NAME}{{plane}}_{{name}}{EXT}")
    beta_phase = Tfs(f"{PREFIX}{BETA_NAME}{{plane}}_{{name}}{EXT}")
    beta_kmod = Tfs(f"{PREFIX}{KMOD_BETA_NAME}{{plane}}_{{name}}{EXT}")
    beta_amp = Tfs(f"{PREFIX}{AMP_BETA_NAME}{{plane}}_{{name}}{EXT}")
    dispersion = Tfs(f"{PREFIX}{DISPERSION_NAME}{{plane}}_{{name}}{EXT}")
    norm_dispersion = Tfs(f"{PREFIX}{NORM_DISP_NAME}{{plane}}_{{name}}{EXT}")
    # TODO: Add coupling!

    def __init__(self, directory: Path, segment_name: str, *args, **kwargs):
        super(SegmentDiffs, self).__init__(directory, *args, **kwargs)
        self.segment_name = segment_name 

    def _get_filename(self, template: str, plane: str=None):
        if plane is None:
            return template.format(name=self.segment_name)
        return template.format(plane=plane.lower(), name=self.segment_name)


class SbsDefinitionError(Exception):
    """ Exception to be raised when the sbs definition is invalid."""
    pass
