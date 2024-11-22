"""
Abstract Model Creator Class
----------------------------

This module provides the template for all model creators.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import tfs

from omc3.madx_wrapper import run_string
from omc3.model.accelerators.accelerator import Accelerator, AccExcitationMode
from omc3.model.constants import (JOB_MODEL_MADX, TWISS_AC_DAT, TWISS_ADT_DAT,
                                  TWISS_DAT, TWISS_ELEMENTS_DAT)
from omc3.optics_measurements.constants import NAME
from omc3.segment_by_segment.constants import (corrections_madx, 
                                               jobfile, 
                                               measurement_madx,
                                               TWISS_BACKWARD,
                                               TWISS_BACKWARD_CORRECTED,
                                               TWISS_FORWARD,
                                               TWISS_FORWARD_CORRECTED)
from omc3.utils import iotools, logging_tools
from omc3.segment_by_segment.propagables import Propagable
from omc3.segment_by_segment.segments import Segment

if TYPE_CHECKING:
    from collections.abc import Sequence, Iterable

LOGGER = logging_tools.get_logger(__file__)

MADXInputType = Path | str | dict[str, str]


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
            LOGGER.warning(f"Unknown args for Model Creator: {', '.join(cleaned_args)}")

        cleaned_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if len(cleaned_kwargs):
            LOGGER.warning(f"Unknown kwargs for Model Creator: {cleaned_kwargs!s}")

    @classmethod
    @abstractmethod
    def check_options(cls, accel_inst: Accelerator, options) -> bool:
        """
        Parses additional commandline options (if any) and checks if they are valid.
        If there are options missing, return False. 
        This function is different from the normal parsing of options, as it allows the 
        model creator to print possible choices for the user.        

        Args:
            options: The remaining options (i.e. those not yet consumed by the model creator)

        Returns: True if enough options are given to provide a valid model

        """
        return True

    @classmethod
    @abstractmethod
    def get_correction_check_script(cls, accel: Accelerator, corr_file: str, chrom: bool) -> str:
        """
        Returns the ``MAD-X`` script used to verify global corrections. This script should create twiss
        files for before (``twiss_no.dat``) and after (``twiss_corr.dat``) correction.

        Args:
            accel (Accelerator): Accelerator Instance used for the model creation.
            corr_file (str): File containing the corrections (madx-readable).
            chrom (bool): Flag for chromatic corrections deltapm and deltapp.

        Returns:
            The string of the ``MAD-X`` script used to verify global corrections.
        """
        pass

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
        Returns the ``MAD-X`` script used to create the model (directory).

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

        Args:
            accel (Accelerator): Accelerator Instance used for the model creation.
        """
        pass

    def post_run(self) -> None:
        """
        Checks that the model creation ``MAD-X`` run was successful. It should check that the
        appropriate directories are created, and that macros and other files are in place.
        Also assings created models to the accelerator instance.
        """
        # These are the default files for most model creators for now.
        files_to_check: list[str] = [TWISS_DAT, TWISS_ELEMENTS_DAT]
        self.accel.model = tfs.read(self.accel.model_dir / TWISS_DAT, index=NAME)
        self.accel.elements = tfs.read(self.accel.model_dir / TWISS_ELEMENTS_DAT, index=NAME)
        
        if self.accel.excitation == AccExcitationMode.ACD:
            files_to_check += [TWISS_AC_DAT]
            self.accel.model_driven = tfs.read(self.accel.model_dir / TWISS_AC_DAT, index=NAME)
        elif self.accel.excitation == AccExcitationMode.ADT:
            files_to_check += [TWISS_ADT_DAT]
            self.accel.model_driven = tfs.read(self.accel.model_dir / TWISS_ADT_DAT, index=NAME)

        self._check_files_exist(self.accel.model_dir, files_to_check)

        

    @staticmethod
    def _check_files_exist(dir_: Path | str, files: Sequence[str]) -> None:
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

    @staticmethod
    def prepare_symlink(accel: Accelerator):
        """Prepare the acc-models-symlink.
        Create symlink if it does not yet exist or points the wrong way.
        Use the symlink from here on instead of the acc-model-path, also in the modifiers.

        This functions can be used by all model creators supporting the acc-models creation.

        Args:
            accel (Accelerator): Accelerator instance 
        """
        if accel.acc_model_path is None:
            return

        target = accel.acc_model_path
        link = Path(accel.model_dir) / accel.REPOSITORY

        if link.is_symlink() or link.exists():
            # something is here
            if not link.resolve().samefile(target.resolve()):
                # and it's not pointing at the right target
                LOGGER.warning(
                    f"{accel.REPOSITORY} already exists in model dir {accel.model_dir}. "
                    f"It will be reset to {target}."
                )
                link.unlink()
                link.absolute().symlink_to(target)
            # else: link already points to the right target -> leave as is

        else:
            # no symlink so we create one
            link.absolute().symlink_to(target)
        
        # use the link from now on as model path and for modifiers
        accel.acc_model_path = link.absolute()
        if accel.modifiers is not None:
            accel.modifiers = [
                iotools.replace_in_path(m.absolute(), target.absolute(), link.absolute()) 
                for m in accel.modifiers
            ]


def check_folder_choices(parent: Path, msg: str,
                         selection: str,
                         list_choices: bool = False,
                         predicate=iotools.always_true) -> Path:
    """
    A helper function that scans a selected folder for children, which will then be displayed as possible choices.
    This funciton allows the model-creator to get only the file/folder names, check
    in the desired folder if the choice is present and return the full path to the selected folder.
    
    Args:
        parent (Path): The folder to scan.
        msg (str): The message to display, on failure.
        selection (str): The current selection.
        list_choices (bool): Whether to just list the choices. 
                             In that case `None` is returned, instead of an error
        predicate (callable): A function that takes a path and returns True.
                              if the path results in a valid choice.
    
    Returns:
       Path: Full path of the selected choice in `parent`.

    Examples:
        Let's say we expect a choice for a sequence file in the folder `model_root`.

        ```
        check_folder_choices(model_root, "Expected sequence file", predicate=lambda p: p.suffix == ".seq")
        ```

        Or we want all subfolder of `scenarios`

        ```
        check_folder_choices(scenarios, "Expected scenario folder", predicate=lambda p: p.is_dir())
        ```
    """
    choices = [d.name for d in parent.iterdir() if predicate(d)]

    if selection is not None and selection in choices:
        return parent / selection

    if list_choices:
        for choice in choices:
            print(choice)
        return None
    raise AttributeError(f"{msg}.\nSelected: '{selection}'.\nChoices: [{', '.join(choices)}]")



class SegmentCreator(ModelCreator, ABC):
    jobfile = None  # set in init

    def __init__(self, accel: Accelerator, segment: Segment, measurables: Iterable[Propagable], 
                 corrections: MADXInputType = None, *args, **kwargs):
        """ Creates Segment of a model. """
        super(SegmentCreator, self).__init__(accel, *args, **kwargs)
        self.segment = segment
        self.measurables = measurables
        self.corrections = corrections

        # Filenames
        self.jobfile = jobfile.format(segment.name)
        self.measurement_madx = measurement_madx.format(segment.name)
        self.corrections_madx = corrections_madx.format(segment.name)
        self.twiss_forward = TWISS_FORWARD.format(segment.name)
        self.twiss_backward = TWISS_BACKWARD.format(segment.name)
        self.twiss_forward_corrected = TWISS_FORWARD_CORRECTED.format(segment.name)
        self.twiss_backward_corrected = TWISS_BACKWARD_CORRECTED.format(segment.name)

    def prepare_run(self) -> None:
        super(SegmentCreator, self).prepare_run()
        self._create_measurement_file()
        self._create_corrections_file()

    def _create_measurement_file(self):
        meas_dict = {}
        for measurable in self.measurables:
            meas_dict.update(measurable.init_conditions_dict())
        meas_file_content = "\n".join(f"{k!s} = {v!s};" for k, v in meas_dict.items())
        output_file = self.output_dir / self.measurement_madx
        output_file.write_text(meas_file_content)

    def _create_corrections_file(self):
        if self.corrections is None:
            return

        output_file = self.output_dir / self.corrections_madx
        if output_file.exists():
            LOGGER.warning(f"Segment corrections file {output_file!s} already exists. Overwriting.")
        output_file.write_text(self._get_corrections_text())
        
    
    def _get_corrections_text(self) -> str:
        if self.corrections is None:
            return ""

        if isinstance(self.corrections, dict):
            return "\n".join(f"{k!s} = {v!s};" for k, v in self.corrections.items())
        
        input_file = Path(self.corrections)
        if input_file.exists():
            return input_file.read_text()
        
        return self.corrections
