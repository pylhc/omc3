"""
Abstract Model Creator Class
----------------------------

This module provides the template for all model creators.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence, Union, Iterable

from omc3.madx_wrapper import run_string
from omc3.model.accelerators.accelerator import Accelerator, AccExcitationMode
from omc3.model.constants import TWISS_AC_DAT, TWISS_ADT_DAT, TWISS_DAT, TWISS_ELEMENTS_DAT
from omc3.model.constants import JOB_MODEL_MADX
import logging

from omc3.optics_measurements.io_filehandler import OpticsMeasurement
from omc3.segment_by_segment.constants import jobfile, twiss_forward_corrected, twiss_backward_corrected, \
    twiss_backward, twiss_forward, corrections_madx, measurement_madx
from omc3.segment_by_segment.propagables import Propagable
from omc3.segment_by_segment.segments import Segment

LOG = logging.getLogger(__name__)


class ModelCreator(ABC):
    jobfile = JOB_MODEL_MADX
    """
    Abstract class for the implementation of a model creator. All mandatory methods and convenience
    functions are defined here.
    """
    def __init__(self, accel: Accelerator, logfile: Path = None, *args, **kwargs):
        """
        Initialize the Model Creator.

        Args:
            accel (Accelerator): Accelerator Instance
        """
        self.accel = accel
        self.logfile = logfile
        self.output_dir = accel.model_dir

        cleaned_args = [arg for arg in args if arg is not None]
        if len(cleaned_args):
            LOG.warning(f"Unknown args for Model Creator: {', '.join(cleaned_args)}")

        cleaned_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if len(cleaned_kwargs):
            LOG.warning(f"Unknown kwargs for Model Creator: {cleaned_kwargs!s}")

    def full_run(self):
        """ Does the full run: preparation, running madx, post_run. """
        # Prepare model-dir output directory
        self.prepare_run()

        # get madx-script with relative output-paths
        self.accel.model_dir = Path()
        madx_script = self.get_madx_script()
        self.accel.model_dir = self.output_dir

        # Run madx to create model
        run_string(
            madx_script,
            output_file=self.accel.model_dir / self.jobfile,
            log_file=self.logfile,
            cwd=self.accel.model_dir
        )

        # Check output and return accelerator instance
        self.post_run()

    @abstractmethod
    def get_madx_script(self) -> str:
        """
        Returns:
            The string of the ``MAD-X`` script used to used to create the model (directory).
        """
        pass

    @abstractmethod
    def prepare_run(self) -> None:
        """
        Prepares the model creation ``MAD-X`` run. It should check that the appropriate directories
        are created, and that macros and other files are in place.
        Should also check that all necessary data for model creation is available in the accelerator
        instance.
        """
        pass

    def post_run(self) -> None:
        """
        Checks that the model creation ``MAD-X`` run was successful. It should check that the
        appropriate directories are created, and that macros and other files are in place.
        Checks the accelerator instance.
        """
        # These are the default files for most model creators for now.
        files_to_check: List[str] = [TWISS_DAT, TWISS_ELEMENTS_DAT]
        if self.accel.excitation == AccExcitationMode.ACD:
            files_to_check += [TWISS_AC_DAT]
        elif self.accel.excitation == AccExcitationMode.ADT:
            files_to_check += [TWISS_ADT_DAT]
        self._check_files_exist(self.accel.model_dir, files_to_check)

    @staticmethod
    def _check_files_exist(dir_: Union[Path, str], files: Sequence[str]) -> None:
        """
        Convenience function to loop over files supposed to be in a locatioin and raise an error if
        one or more of these files does not exist.

        Args:
            dir_ (Union[Path, str]): Path object or string of the absolute path of the directory in
                which to check for the files.
            files (Sequence[str]): the names of files to check the presence of in the directory.

        Raises:
            Raises an ``FileNotFoundError``
        """
        for out_file in files:
            file_path = Path(dir_) / out_file
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Model Creation Failed. The file '{file_path.absolute()}' was not created."
                )


class SegmentCreator(ModelCreator, ABC):
    jobfile = None  # set in init

    """ Creates Segment of a model. """
    def __init__(self, accel: Accelerator, segment: Segment, measurables: Iterable[Propagable], *args, **kwargs):
        super(SegmentCreator, self).__init__(accel, *args, **kwargs)
        self.segment = segment
        self.measurables = measurables

        # Filenames
        self.jobfile = jobfile.format(segment.name)
        self.measurement_madx = measurement_madx.format(segment.name)
        self.corrections_madx = corrections_madx.format(segment.name)
        self.twiss_forward = twiss_forward.format(segment.name)
        self.twiss_backward = twiss_backward.format(segment.name)
        self.twiss_forward_corrected = twiss_forward_corrected.format(segment.name)
        self.twiss_backward_corrected = twiss_backward_corrected.format(segment.name)

    def prepare_run(self) -> None:
        super(SegmentCreator, self).prepare_run()
        self._create_measurement_file()

    def _create_measurement_file(self):
        meas_dict = {}
        for measurable in self.measurables:
            meas_dict.update(measurable.init_conditions_dict())
        meas_file_content = "\n".join(f"{k} = {v};" for k, v in meas_dict.items())
        output_file = self.output_dir / self.measurement_madx
        output_file.write_text(meas_file_content)
