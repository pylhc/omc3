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
from omc3.model.constants import (
    JOB_MODEL_MADX_NOMINAL,
    OPTICS_SUBDIR,
    TWISS_AC_DAT,
    TWISS_ADT_DAT,
    TWISS_BEST_KNOWLEDGE_DAT,
    TWISS_DAT,
    TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT,
    TWISS_ELEMENTS_DAT,
)
from omc3.optics_measurements.constants import NAME
from omc3.segment_by_segment.constants import (
    TWISS_BACKWARD,
    TWISS_BACKWARD_CORRECTED,
    TWISS_FORWARD,
    TWISS_FORWARD_CORRECTED,
    corrections_madx,
    jobfile,
    measurement_madx,
)
from omc3.segment_by_segment.propagables import Propagable
from omc3.segment_by_segment.segments import Segment
from omc3.utils import iotools, logging_tools

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    MADXInputType = Path | str | dict[str, str]

LOGGER = logging_tools.get_logger(__file__)



class ModelCreator(ABC):
    """
    Abstract class for the implementation of a model creator. All mandatory methods and convenience
    functions are defined here.
    """
    jobfile: str = JOB_MODEL_MADX_NOMINAL

    def __init__(self, accel: Accelerator, logfile: Path = None):
        """
        Initialize the Model Creator.

        Args:
            accel (Accelerator): Accelerator Instance
        """
        self.accel = accel
        self.logfile = logfile
        self.output_dir = accel.model_dir
    
    @abstractmethod
    def check_options(self, opt) -> bool:
        """
        Parses additional commandline options (if any) and checks if they are valid.
        If there are options missing, return False. 
        This function is different from the normal parsing of options, as it allows the 
        model creator to print possible choices for the user.        

        Args:
            opt: The remaining options (i.e. those not yet consumed by the model creator)

        Returns: True if enough options are given to provide a valid model

        """
        return True

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
    def get_base_madx_script(self) -> str:
        """
        Returns the ``MAD-X`` script used to set-up the machine, without creating the model (twiss-output).

        Returns:
            The string of the ``MAD-X`` script used to used to set-up the machine.
        """
        pass
    
    def prepare_run(self) -> None:
        """
        Prepares the model creation ``MAD-X`` run. It should check that the appropriate directories
        are created, and that macros and other files are in place.
        Should also check that all necessary data for model creation is available in the accelerator
        instance. 

        Here implemented are some usual defaults, so that an implementation of the model-creator
        might run these easily with `super()` if desired.

        Args:
            accel (Accelerator): Accelerator Instance used for the model creation.
        """
        LOGGER.info("Preparing MAD-X run for model creation.")
        # adjust modifier paths, to allow giving only filenames in default directories (e.g. optics)
        if self.accel.modifiers is not None:
            self.accel.modifiers = [self._find_modifier(m) for m in self.accel.modifiers]
        
        # prepare the acc-models-symlink and replace paths to use the symlink
        self.prepare_symlink()

    def post_run(self) -> None:
        """
        Checks that the model creation ``MAD-X`` run was successful. It should check that the
        appropriate directories are created, and that macros and other files are in place.
        Also assings created models to the accelerator instance.

        Hint: If you only need to check a different set of files, you can simply override the `files_to_check` property,
              instead of this whole function.
        """
        LOGGER.info("Checking output from MAD-X run for model creation.")
        self._check_files_exist(self.accel.model_dir, self.files_to_check)
        
        # Load the twiss files
        attribute_map = {
            TWISS_DAT: "model",
            TWISS_ELEMENTS_DAT: "elements",
            TWISS_BEST_KNOWLEDGE_DAT: "model_best_knowledge",
            TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT: "elements_best_knowledge",
            TWISS_AC_DAT: "model_driven",
            TWISS_ADT_DAT: "model_driven",
        }
        for filename in self.files_to_check:
            try: 
                setattr(self.accel, attribute_map[filename], tfs.read(self.accel.model_dir / filename, index=NAME))
            except KeyError:
                pass  # just a file to check, not a file with attribute

    
    @property
    def files_to_check(self) -> list[str]:
        """
        Returns the list of files to check after model creation, 
        should only be used in `post_run`.
        Override in subclass if you need to check a different set of files.
        """
        check_files = [TWISS_DAT, TWISS_ELEMENTS_DAT]  # default for most accelerators
        excitation_map = {
            AccExcitationMode.FREE: [],
            AccExcitationMode.ACD: [TWISS_AC_DAT],
            AccExcitationMode.ADT: [TWISS_ADT_DAT],
        }
        check_files = check_files + excitation_map[self.accel.excitation]
        return check_files 

    @staticmethod
    def _check_files_exist(dir_: Path | str, files: Sequence[str]) -> None:
        """
        Convenience function to loop over files supposed to be in a location and raise an error if
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
    
    def prepare_symlink(self):
        """Prepare the acc-models-symlink.
        Create symlink if it does not yet exist or points the wrong way.
        Use the symlink from here on instead of the acc-model-path, also in the modifiers.

        This functions can be used by all model creators supporting the acc-models creation.
        """
        accel = self.accel
        if accel.acc_model_path is None or accel.REPOSITORY is None:
            LOGGER.debug(f"No symlink required for accel {accel.NAME}.")
            return

        LOGGER.debug("Preparing acc-models-symlink")
        target = accel.acc_model_path
        link: Path = Path(accel.model_dir) / accel.REPOSITORY

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
        
        # use the link from now on as model path and for modifiers;
        # this converts all modifiers to absolute paths ... maybe not desired? (jdilly, 2024)
        accel.acc_model_path = link.absolute()
        if accel.modifiers is not None:
            accel.modifiers = [
                iotools.replace_in_path(m.absolute(), target.absolute(), link.absolute()) 
                for m in accel.modifiers
            ]
    
    def _find_modifier(self, modifier: Path):
        # first case: if modifier exists as is, take it
        if modifier.exists():
            return modifier

        # second case: try if it is already in the output dir
        model_dir_path: Path = self.accel.model_dir / modifier
        if model_dir_path.exists():
            return model_dir_path.absolute()

        # and last case, try to find it in the acc-models rep
        if self.accel.acc_model_path is not None:
            optics_path: Path = self.accel.acc_model_path / OPTICS_SUBDIR / modifier
            if optics_path.exists():
                return optics_path.absolute()

        # if you are here, all attempts failed
        msg = (
            f"couldn't find modifier {modifier}. "
            f"Tried in {self.accel.model_dir} and {self.accel.acc_model_path}/{OPTICS_SUBDIR}"
        )
        raise FileNotFoundError(msg)


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


class CorrectionModelCreator(ModelCreator, ABC):
    jobfile = None  # set in __init__ 
    
    def __init__(self, accel: Accelerator, twiss_out: Path | str, corr_files: Sequence[Path | str], update_dpp: bool = False):
        """Model creator for the corrected/matched model of the LHC.

        Args:
            accel (Accelerator): Accelerator Class Instance.
            twiss_out (Path | str): Path to the twiss(-elements) file to write.
            change_params (Sequence[Path]): Sequence of correction/matching files.
            update_dpp (bool): Whether to update the dpp in the machine.
        """
        super(CorrectionModelCreator, self).__init__(accel)
        self.twiss_out = Path(twiss_out)

        # use absolute paths to force files into twiss_out directory instead of model-dir
        self.jobfile = self.twiss_out.parent.absolute() / f"job.create_{self.twiss_out.stem}.madx"
        self.logfile= self.twiss_out.parent.absolute() / f"job.create_{self.twiss_out.stem}.log"
        self.corr_files = corr_files
        self.update_dpp = update_dpp

    @abstractmethod
    def get_update_deltap_script(self, deltap: float | str) -> str:
        """ Get the madx script that updates the dpp in the machine.
        
        Args:
            deltap (float | str): The dpp to update the machine to.
         """
        pass