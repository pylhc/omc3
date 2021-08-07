from pathlib import Path

from tfs.collection import TfsCollection, Tfs

from omc3.segment_by_segment.constants import twiss_forward, twiss_backward, twiss_forward_corrected, \
    twiss_backward_corrected


class Segment:

    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end
        self.element = None
        self.ini_conds = None

    @staticmethod
    def init_from_element(element_name):
        fake_segment = Segment(element_name, element_name, element_name)
        fake_segment.element = element_name
        return fake_segment


class SegmentModels(TfsCollection):  # write_to does not need to be implemented
    """
    Class to hold and load the models of the segments created by MAD-X.

    Arguments:
        directory: The path where to find the models.
        segment: A segment instance corresponding to the model to load.
    """

    forward = Tfs(twiss_forward, two_planes=False)
    backward = Tfs(twiss_backward, two_planes=False)
    forward_corrected = Tfs(twiss_forward_corrected, two_planes=False)
    backward_corrected = Tfs(twiss_backward_corrected, two_planes=False)

    def __init__(self, directory: Path, segment: Segment):
        super(SegmentModels, self).__init__(directory)
        self.segment = segment

    def get_filename(self, template: str):
        return template.format(self.segment.name)


class SegmentBeatings(TfsCollection):  # write_to does not need to be implemented
    """
    TfsCollection of segment-by-segment outputfiles for the differences
    between propagated model and measurements.

    Arguments:
        directory: The path where to write the files to/find the files.
        seg_name: A segment corresponding to the model to load.
    """

    beta_phase = Tfs("sbsbetabeating{plane}_{name}.out")
    beta_kmod = Tfs("sbskmodbetabeat{plane}_{name}.out")
    beta_amp = Tfs("sbsampbetabeat{plane}_{name}.out")
    phase = Tfs("sbsphase{plane}_{name}.out")
    coupling = Tfs("sbscouple_{name}.out", two_planes=False)
    disp = Tfs("sbsD{plane}_{name}.out")
    norm_disp = Tfs("sbsNDx_{name}.out", two_planes=False)

    def __init__(self, directory: Path, segment: Segment):
        super(SegmentBeatings, self).__init__(directory)
        self.segment = segment

    def get_filename(self, template: str, plane: str=None):
        if plane is None:
            return template.format(name=self.segment.name)
        return template.format(plane=plane.lower(), name=self.segment.name)


class SbsDefinitionError(Exception):
    """
    TODO
    """
    pass
