""" 
Segment-by-Segment Correction
-----------------------------

TODO

"""
import functools
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union

import pandas as pd
from generic_parser import EntryPointParameters, entrypoint
from generic_parser.entrypoint_parser import add_to_arguments
from pandas import DataFrame

from omc3 import model_creator
from omc3.definitions.optics import OpticsMeasurement
from omc3.model import manager
from omc3.model.accelerators.accelerator import Accelerator, AcceleratorDefinitionError
from omc3.model.constants import ACC_MODELS_PREFIX, MACROS_DIR, TWISS_ELEMENTS_DAT
from omc3.model.model_creators.abstract_model_creator import MADXInputType
from omc3.model.model_creators.lhc_model_creator import LhcSegmentCreator
from omc3.segment_by_segment.constants import logfile
from omc3.segment_by_segment.propagables import Propagable, get_all_propagables
from omc3.segment_by_segment.segments import (SbsDefinitionError, Segment, SegmentDiffs,
                                              SegmentModels)
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, PathOrStrOrDict

LOGGER = logging_tools.get_logger(__name__)


CREATORS = {
    "lhc": LhcSegmentCreator,
}


def get_parameters():
    return EntryPointParameters(
        measurement_dir=dict(
            help="Path to the measurement files.",
            required=True,
            type=PathOrStr,
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
        corrections=dict(
            help=("Corrections to use. Can be a dict of knob-value pairs, "
                 "a MAD-X string or a path to a file. "
                 "Note: all segements will have these corrections."),
            type=PathOrStrOrDict,
        ),
        output_dir=dict(
            help="Output directory. If not given, the model-directory is used.",
            type=PathOrStr,
        )
    )


@entrypoint(get_parameters(), strict=False)
def segment_by_segment(opt, accel_opt) -> Dict[str, SegmentDiffs]:
    """
    TODO
    """
    accel = _get_accelerator_instance(accel_opt, opt.output_dir)
    if opt.output_dir is not None and  Path(opt.output_dir) != accel.model_dir:
        _copy_needed_model_files(accel.model_dir, opt.output_dir)
        accel.model_dir = opt.output_dir

    measurement = OpticsMeasurement(opt.measurement_dir)
    segments, elements = _check_segments_and_elements(opt.segments, opt.elements)

    results: Dict[str, SegmentDiffs] = {}
    for segment in segments + elements:
        propagables = create_segment(accel, segment, measurement, opt.corrections)
        results[segment.name] = get_differences(propagables, segment.name, accel.model_dir)

    return results


def create_segment(accel: Accelerator, segment_in: Segment, 
                   measurement: OpticsMeasurement, corrections: MADXInputType) -> List[Propagable]:
    """Perform the computations on the segment.
    The segment is adapted fist, so that the given start and end bpms a in the measurement.
    Then madx is run to create the specified segments and the output files
    are made accessible by a SegmentModels TfsCollection by all propagables, which
    are then returned.
    
    Args:
        accel (Accelerator): Accelerator instance. Needs to be loaded from model directory
        segment_in (Segment): Rough Segment specification. Might be refined later.
        measurement (OpticsMeasurement): TfsCollection of the optics measurments files.
        corrections (MADXInputType): Corrections to use. Can be a dict of knob-value pairs, 
                                     a MAD-X string or a path to a file.
        output (Path): Path to the output directory.

    Returns:
        List of propagables with access to the created segment models.
    """
    segment = extend_segment(segment_in, accel.elements, measurement)
    
    LOGGER.info(
        f"Evaluating segment {segment!s}.\n" + 
        "" if segment == segment_in else
        f"This has been input as {segment_in!s}."
    )

    propagables = [propg(segment, measurement) for propg in get_all_propagables()]
    propagables = [measbl for measbl in propagables if measbl]


    # Create the segment via madx
    segment_creator = CREATORS[accel.NAME](
        accel, segment, propagables, 
        corrections=corrections,
        logfile=accel.model_dir / logfile.format(segment.name),
    )
    segment_creator.full_run()

    # Make the created segment accessible to all propagables via SegmentModels
    seg_models = SegmentModels(accel.model_dir, segment)
    for propagable in propagables:
        propagable.segment_models = seg_models
    return propagables


def extend_segment(segment: Segment, model: DataFrame, measurement: OpticsMeasurement) -> Segment:
    """Returns a new segment with elements that are contained in the measurement.

    This function takes a segment with start and end that might not
    be in the measurement and searches the next element that satisfies
    it, returning a new segment with the new start and end elements.

    Args:
        segment (Segment): The segment to be processed (see Segment class).
        model (DataFrame): The model where to take all the element names from. Both the
            start and end of the segment have to be present in this model
            NAME attribute.

    Returns:
        A new segment with generally different start and end but always the
        same name and element attributes.
    """
    for name in (segment.start, segment.end):
        if name not in model.index:
            raise SbsDefinitionError(f"Element name {name} not in the input model.")

    eval_funct = functools.partial(_bpm_is_in_meas, meas=measurement)

    new_start = _select_closest(segment.start, model.index, eval_funct, back=True)
    new_end = _select_closest(segment.end, model.index, eval_funct, back=False)
    new_segment = Segment(segment.name, new_start, new_end)
    new_segment.element = segment.element
    return new_segment


def get_differences(propagables: List[Propagable], segment_name: str = "", output_dir: Path = None) -> SegmentDiffs:
    """Calculate the differences of the propagated model and the measurement and write
    them out into files (if ``output`` had been given).

    Args:
        propagables (Iterable[Propagable]): List of propagables to calculate the differences.
        segment_name (str): Name of the segment to calculate the differences for.
        output_dir (Path, optional): Output directory. Defaults to None.

    Returns:
        SegmentDiffs: TfsCollection of the differences per propagable.    
    """
    segment_diffs = SegmentDiffs(
        directory=output_dir, 
        segment_name=segment_name, 
        allow_write=output_dir is not None
    )
    for propagable in propagables:
        try:
            propagable.add_differences(segment_diffs)
        except NotImplementedError:
            pass
    return segment_diffs 


def _get_accelerator_instance(accel_opt: dict, output_dir: Union[Path, str]) -> Accelerator:
    """Get accelerator instance from ``accel_opt`` and create a nominal model if not present."""
    try:
        accel_inst = manager.get_accelerator(accel_opt)
    except AcceleratorDefinitionError:
        accel_inst = None
    
    if accel_inst is None or accel_inst.model_dir is None:
        return model_creator.create_instance_and_model(
            add_to_arguments(accel_opt, model_creator._get_params(), outputdir=output_dir)
        )
    
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


def _parse_segments(segment_definitions: Sequence[Union[Segment, str]]) -> List[Segment]:
    """Convert all segment definitions to Segments.     

    Args:
        segment_definitions (Sequence[str | Segment]): List of segment definitions or Segments. 

    Raises:
        SbsDefinitionError: When there are duplicated names or when the definition is not recognized.

    Returns:
        List[Segment]: List of parsed segments. 
    """
    if segment_definitions is None:
        return []

    segments = {}
    for segment_def in segment_definitions:
        if isinstance(segment_def, Segment):
            segment = segment_def
        else:
            try:
                segment = Segment.init_from_segment_definition(segment_def)
            except ValueError:
                raise SbsDefinitionError(f"Unable to parse segment string {segment_def}.")

        if segment.name in segments.keys():
            raise SbsDefinitionError(f"Duplicated segment name '{segment.name}'.")
        segments[segment.name] = segment
    return list(segments.values())


def _parse_elements(elements: Sequence[Union[Segment, str]]) -> List[Segment]:
    """Convert all elements to Segments.

    Args:
        elements (Sequence[str | Segment]): Elements to be parsed or Segment-instaces.

    Raises:
        SbsDefinitionError: When there are duplicated names.

    Returns:
        List[Segment]: List of the parsed segments. 
    """
    if elements is None:
        return []
    if len(set(elements)) != len(elements):
        raise SbsDefinitionError("Duplicated names in element list.")
    elem_segments = [name if isinstance(name, Segment) else Segment.init_from_element_name(name) for name in elements]
    return elem_segments


def _select_closest(name: str, all_names: pd.Index, eval_cond: Callable[[str], bool], back: bool = False) -> str:
    """Select the closest element to the given name, going forward or backward, until the condition is met.

    Args:
        name (str): Name to start  
        all_names (pd.Index): Pandas Index of all element names (e.g. from model) 
        eval_cond (Callable[[str], bool]): Function taking a single argument (the name) and returning a boolean, 
                                           whether this name can be used or not (e.g. is in the measurement).
        back (bool, optional): Search direction, False for forwards and True for backwards. Defaults to False.

    Raises:
        SbsDefinitionError: Is raised when all names have been checked, but no suitable element is found.

    Returns:
        str: Name of the closest element fulfilling the condition.
    """
    new_name = name
    n_names = len(all_names)
    checked_names = []
    while not eval_cond(new_name):
        checked_names.append(new_name)
        delta = 1 if not back else -1
        next_index = (all_names.get_loc(new_name) + delta) % n_names
        new_name = all_names[next_index]
        if new_name in checked_names:
            raise SbsDefinitionError(
                "No elements found fulfilling the condition. "
                "Probably wrong model or bad measurement."
            )
    return new_name


def _copy_needed_model_files(model_dir: Path, output_dir: Path) -> None:
    """Copy the required model files from the model-dir to the output dir.
    This is done, as the paths are usually relative to the model-dir.

    Args:
        model_dir (Path): Path to the model directory.
        output_dir (Path): Path to the output directory.
    """
    LOGGER.debug("Copying model files...")
    shutil.copy(model_dir / TWISS_ELEMENTS_DAT, output_dir / TWISS_ELEMENTS_DAT)
    for file in model_dir.glob("*"):
        if file.name.startswith(ACC_MODELS_PREFIX) or file.name == MACROS_DIR:
            dst_file = output_dir / file.name
            if dst_file.exists():
                LOGGER.debug(f"Skipping {dst_file!s}, already exists.")
                continue
            shutil.copy(file, dst_file, follow_symlinks=False)


def _bpm_is_in_meas(bpm_name: str, meas: OpticsMeasurement) -> bool:
    """ Check if the bpm_name is in the measurement in both planes.
    Possible improvement: Check if the error is too high?
    """
    return bpm_name in meas.beta_phase_x.index and bpm_name in meas.beta_phase_y.index


if __name__ == "__main__":
    segment_by_segment()
