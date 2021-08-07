import argparse
import os
import sys
from collections import OrderedDict, Iterable
from pathlib import Path
from shutil import copyfile

from os.path import abspath, join, dirname, pardir
from typing import List, Tuple

from pandas import DataFrame

from omc3.model.accelerators.accelerator import Accelerator
from omc3.segment_by_segment.segments import SbsDefinitionError, Segment, SegmentBeatings, SegmentModels


from omc3.model.model_creators.lhc_model_creator import LhcSegmentCreator

from omc3.model import manager
from omc3.utils import logging_tools
from omc3.optics_measurements.io_filehandler import OpticsMeasurement
import tfs
from omc3.segment_by_segment.propagables import get_all_propagables, Propagable
from omc3 import model_creator
from generic_parser import entrypoint, EntryPointParameters, DotDict

# TODO: Remove debug and set up log file
import logging
DEBUG = logging.DEBUG
LOGGER = logging_tools.get_logger(__name__, level_console=DEBUG)

PLANES = ("x", "y")

CREATORS = {
    "lhc": LhcSegmentCreator,
}


def get_parameters():
    return EntryPointParameters(
        measurement_dir=dict(
            help="Path to the measurement files.",
            required=True,
            type=Path,
        ),
        elements=dict(
            help="Element name list to run in element mode.",
            nargs="+",
            type=str,
        ),
        segments=dict(
            help=("Segments to run in segment mode with format: "
                  "``segment_name,start,end``, where start and end "
                  "must be existing BPM names."),
            nargs="+",
            type=str,
        ),
        outputdir=dict(
            help="Output directory. If not given, the model-directory is used.",
            type=Path,
        )

    )


@entrypoint(get_parameters())
def segment_by_segment(opt, accel_opt):
    """
    TODO
    """
    accel_inst = _get_accelerator_instance(accel_opt, opt.outputdir)
    if opt.outputdir is None:
        opt.outputdir = accel_inst.model_dir

    measurement = OpticsMeasurement(opt.measurement)
    segments, elements = _check_segments_and_elements(opt.segments, opt.elements)
    for segment in segments + elements:
        propagable = run_for_segment(accel_inst, segment, measurement, opt.outputdir)
        write_beatings(segment, propagable, opt.outputdir)


def run_for_segment(accel: Accelerator, segment_in: Segment, measurement: OpticsMeasurement, output: Path
                    ) -> List[Propagable]:
    """Perform the computations on the segment.
    The segment is adapted fist, so that the given start and end bpms a in the measurement.
    Then madx is run to create the specified segments and the output files
    are made accessible by a SegmentModels TfsCollection by all propagables, which
    are then returned.
    
    Args:
        accel (Accelerator): Accelerator instance. Needs to be loaded from model directory
        segment_in (Segment): Rough Segment specification. Might be refined later.
        measurement (OpticsMeasurement): TfsCollection of the optics measurments files.
        output (Path): Path to the output directory.

    Returns:
        List of propagables with access to the created segment models.
    """
    segment = improve_segment(segment_in, accel.elements, measurement, eval_funct=_bpm_is_in_beta_meas)
    propagables = [propg(segment, measurement) for propg in get_all_propagables()]
    propagables = [measbl for measbl in propagables if measbl]

    LOGGER.info(
        f"Evaluating segment {segment.name} ({segment.start}, {segment.end}).\n"
        f"This has been input as {segment_in.name} ({segment_in.start}, {segment_in.end})."
    )

    # Create the segment via madx
    segment_creator = CREATORS[accel.NAME](
        accel, segment, propagables, logfile=f"{segment.name}_madx.log"
    )
    segment_creator.full_run()

    # Make the created segments accessible via SegmentModels
    seg_models = SegmentModels(output, segment)
    for propagable in propagables:
        propagable.segment_models = seg_models
    return propagables


def improve_segment(segment: Segment, model: DataFrame, measurement: OpticsMeasurement, eval_funct) -> Segment:
    """Returns a new segment with elements that satisfies eval_funct.

    This function takes a segment with start and end that might not
    satisfy 'eval_funct' and searches the next element that satisfies
    it, returning a new segment with the new start and end elements.

    Args:
        segment (Segment): The segment to be processed (see Segment class).
        model (DataFrame): The model where to take all the element names from. Both the
            start and end of the segment have to be present in this model
            NAME attribute.
        measurement (OpticsMeasurement): An instance of the Measurement class
             that will be passed to 'eval_funct' to check elements for validity.
        eval_funct: An user-provided function that takes an element name as
            first argument and an instance of the Measurement class as second,
            and returns True only if the element is evaluated as good start or
            end for the segment, usually checking for presence in the
            measurement and not too large error.

    Returns:
        A new segment with generally different start and end but always the
        same name and element attributes.
    """
    names = list(model.NAME)
    for name in (segment.start, segment.end):
        if name not in names:
            raise SbsDefinitionError(f"Element name {name} not in the input model.")

    def eval_funct_meas(name):
        return eval_funct(name, measurement)

    new_start = _select_closest(segment.start, names, eval_funct_meas, back=True)
    new_end = _select_closest(segment.end, names, eval_funct_meas, back=False)
    new_segment = Segment(segment.name, new_start, new_end)
    new_segment.element = segment.element
    return new_segment


def write_beatings(segment: Segment, propagables: Iterable[Propagable], output: Path):
    """
    Calculate the differences of the propagated model and the measurement and write
    them out into files.
    """
    seg_beats = SegmentBeatings(output, segment)
    seg_beats.allow_write = True
    for propagable in propagables:
        try:
            propagable.write_to_file(seg_beats)  # computations are done in here
        except NotImplementedError:
            pass


def _get_accelerator_instance(accel_opt: dict, outputdir: Path) -> Accelerator:
    """Get accelerator instance from ``accel_opt`` and create a nominal model if not present."""
    accel_inst = manager.get_accelerator(accel_opt)
    if accel_inst.model_dir is None:
        if outputdir is None:
            raise SbsDefinitionError(
                "Give either a valid ``model_dir`` with pre-created model or an ``outputdir``"
            )
        LOGGER.info(f"Creating Model in {outputdir}")
        creator_cls = model_creator.CREATORS[accel_inst.NAME]['nominal']
        accel_inst.model_dir = outputdir
        creator = creator_cls(accel_inst)
        creator.full_run()

        # Initialize from this model dir, so that elements are loaded
        accel_inst = accel_inst.__class__(DotDict(model_dir=outputdir))

    return accel_inst


def _check_segments_and_elements(segments: List[str], elements: List[str]) -> Tuple[List[Segment], List[Segment]]:
    """Convert segments and elements to Segments and check for duplicate names."""
    if not segments and not elements:
        raise SbsDefinitionError("No segments or elements provided in the input.")

    segments = _parse_segments(segments)
    elements = _parse_elements(elements)

    if not set(s.name for s in segments).isdisjoint(e.name for e in elements):
        raise SbsDefinitionError("Duplicated names in segments and elements.")

    return segments, elements


def _parse_segments(segment_definitions) -> List[Segment]:
    if segment_definitions is None:
        return []

    segments = {}
    for segment_def in segment_definitions:
        try:
            name, start, end = segment_def.split(",")
        except ValueError:
            raise SbsDefinitionError(f"Unable to parse segment string {segment_def}.")

        if name in segments.keys():
            raise SbsDefinitionError(f"Duplicated segment name '{name}'.")
        segments[name] = Segment(name, start, end)
    return list(segments.values())


def _parse_elements(elements) -> List[Segment]:
    if elements is None:
        return []
    if len(set(elements)) != len(elements):
        raise SbsDefinitionError("Duplicated names in element list.")
    elem_segments = [Segment.init_from_element(name) for name in elements]
    return elem_segments


def _select_closest(name, all_names, eval_cond, back=False):
    new_name = name
    while not eval_cond(new_name):
        delta = 1 if not back else -1
        next_index = (all_names.index(new_name) + delta) % len(all_names)
        new_name = all_names[next_index]
        if name == new_name:
            raise SbsDefinitionError(
                "No elements remaining after filtering. "
                "Probably wrong model or bad measurement."
            )
    return new_name


def _bpm_is_in_beta_meas(bpm_name, meas):
    return bpm_name in meas.beta_x and bpm_name in meas.beta_y


if __name__ == "__main__":
    segment_by_segment()
