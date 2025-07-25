"""
Abstract Model Creator Class
----------------------------

This module provides the template for all model creators.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

import tfs

from omc3.madx_wrapper import run_string
from omc3.model.accelerators.accelerator import Accelerator, AccExcitationMode, AcceleratorDefinitionError
from omc3.model.constants import (
    GENERAL_MACROS,
    JOB_MODEL_MADX_NOMINAL,
    MACROS_DIR,
    OMC3_MADX_MACROS_DIR,
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
    MADXInputType = Path | str | dict[str, str] | None

LOGGER = logging_tools.get_logger(__file__)


class ModelCreator(ABC):
    """
    Abstract class for the implementation of a model creator. 
    All mandatory methods and convenience functions are defined here.
    """
    jobfile: str = JOB_MODEL_MADX_NOMINAL  # lowercase as it might be changed in subclasses __init__

    def __init__(self, accel: Accelerator, logfile: Path = None, acc_models_path: Path = None):
        """
        Initialize the Model Creator.

        Args:
            accel (Accelerator): Accelerator Instance
        """
        LOGGER.debug("Initializing Model Creator Base Attributes")
        self.accel: Accelerator = accel
        self.acc_models_path: Path = acc_models_path
        self.logfile: Path = logfile
        self.output_dir: Path = accel.model_dir
    
    @abstractmethod
    def prepare_options(self, opt):
        """
        Checks the options specific to the accelerator and 
        applies them to the instance, if they are valid.
        If there are options missing or wrongly set, raise an AcceleratorDefinitionError.

        This function should different from the normal parsing of options, 
        as it is used to print possible choices for the user.
        In particular it is used for the "fetcher" and sets up the paths to be
        used later by the model creator.

        Args:
            opt: The remaining options (i.e. those not yet consumed by the model creator)

        """
        pass

    def full_run(self):
        """ Does the full run: preparation, running madx, post_run. """
        # Prepare model-dir output directory
        self.prepare_run()
        
        # get madx-script with relative output-paths 
        # (model_dir is base-path in get_madx_script)
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
        Returns the ``MAD-X`` script used to set-up the basic accelerator in MAD-X, without actually creating the twiss-output,
        as some modifications to the accelerator may come afterwards (depending on which model-creator is calling this).

        Returns:
            The string of the ``MAD-X`` script used to used to set-up the machine.
        """
        pass
    
    def get_update_deltap_script(self, deltap: float | str) -> str:
        """ Get the madx snippet that updates the dpp in the machine.
        
        Args:
            deltap (float | str): The dpp to update the machine to.
         """
        raise NotImplementedError("Update dpp not implemented for this model creator.")

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
        self.prepare_modifiers()  # adjust modifier paths, allowing giving filenames in default directories (e.g. optics)
        self.prepare_symlink()  # prepare the acc-models-symlink and replace paths to use the symlink

    def post_run(self) -> None:
        """
        Checks that the model creation ``MAD-X`` run was successful. It should check that the
        appropriate directories are created, and that macros and other files are in place.
        Also assigns the created model output to the accelerator instance (e.g. `elements`, `model`, ...).

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
                    f"Model Creation Failed: The file '{out_file}' was not found in the directory '{dir_}', "
                    f"implying the expected file '{file_path.absolute()}' was not created by MAD-X."
                )
    
    def prepare_symlink(self):
        """Prepare the acc-models-symlink.
        Create symlink if it does not yet exist or points the wrong way.
        Use the symlink from here on instead of the acc-model-path, also in the modifiers.

        This functions can be used by all model creators supporting the acc-models creation.
        """
        accel = self.accel
        if accel.acc_model_path is None or accel.LOCAL_REPO_NAME is None:
            LOGGER.debug(f"No symlink required for accel {accel.NAME}.")
            return

        LOGGER.debug("Preparing acc-models-symlink")
        target = accel.acc_model_path
        link: Path = Path(accel.model_dir) / accel.LOCAL_REPO_NAME

        if link.is_symlink() or link.exists():
            # something is here
            if not link.resolve().samefile(target.resolve()):
                # and it's not pointing at the right target
                LOGGER.warning(
                    f"{accel.LOCAL_REPO_NAME} already exists in model dir {accel.model_dir}. "
                    f"It will be reset to {target}."
                )
                link.unlink()
                link.absolute().symlink_to(target)
            # else: link already points to the right target (or is the target) -> leave as is

        else:
            # no symlink so we create one
            link.parent.mkdir(parents=True, exist_ok=True)
            link.absolute().symlink_to(target)
        
        # use the link from now on as model path and for modifiers;
        # this converts all modifiers to absolute paths ... maybe not desired? (jdilly, 2024)
        accel.acc_model_path = link.absolute()
        if accel.modifiers is not None:
            accel.modifiers = [
                iotools.replace_in_path(m.absolute(), target.absolute(), link.absolute()) 
                for m in accel.modifiers
            ]
        
    def prepare_modifiers(self):
        """ Loop over the modifiers and make them full paths if found. """
        accel: Accelerator = self.accel
        if accel.modifiers is not None:
            accel.modifiers = [accel.find_modifier(m) for m in accel.modifiers]
    
    @staticmethod
    def _get_select_command(pattern: str | None = None, indent: int = 0):
        """ Returns a basic select command with the given pattern, the default columns and correct indentation. """
        space = " " * indent
        pattern_str = f" pattern=\"{pattern}\"," if pattern is not None else ""
        return (
            f"{space}select, flag=twiss, clear;\n"
            f"{space}select, flag=twiss,{pattern_str} column="
            "name, s, keyword, l, betx, bety, mux, muy, angle, "
            "k1l, k2l, k3l, x, y, r11, r12, r21, r22, "
            "alfx, alfy, dx, dpx, dy, dpy, px, py, phix, phiy;\n"
        )


class SegmentCreator(ModelCreator, ABC):
    """ Model creator for Segments, to be used in the Segment-by-Segment algorithm. 
    These segments propagate the measured values from the beginning of the segment to the end.

    This only handles the MAD-X part of things. 
    The input is determined by the passed measurables (Propagable objects), 
    which provide the values via `init_conditions_dict()` method.

    The output is stored in the `twiss_forward` and `twiss_backward` files,
    which in turn can be used for further processing by the implemented Propagables.
    """
    jobfile = None  # set in init
    _sequence_name: str = None  # to be set by any accelerator using the default `get_madx_script`

    def __init__(self, accel: Accelerator, segment: Segment, measurables: Iterable[Propagable], 
                 corrections: MADXInputType = None, *args, **kwargs):
        """ Creates Segment of a model. """
        LOGGER.debug("Initializing Segment Creator")
        super(SegmentCreator, self).__init__(accel, *args, **kwargs)
        self.segment = segment
        self.measurables = measurables
        self.corrections = corrections

        # Filenames
        self.jobfile = jobfile.format(segment.name)
        self.corrections_madx = corrections_madx  # use same correction file for all segments.
        self.measurement_madx = measurement_madx.format(segment.name)
        self.twiss_forward = TWISS_FORWARD.format(segment.name)
        self.twiss_backward = TWISS_BACKWARD.format(segment.name)
        self.twiss_forward_corrected = TWISS_FORWARD_CORRECTED.format(segment.name)
        self.twiss_backward_corrected = TWISS_BACKWARD_CORRECTED.format(segment.name)

    def prepare_run(self) -> None:
        super(SegmentCreator, self).prepare_run()
        self._clean_models()
        self._create_measurement_file()
        self._create_corrections_file()
        self._create_general_macros()

    def _create_general_macros(self):
        accel: Accelerator = self.accel
        macros_path = accel.model_dir / MACROS_DIR
        macros_path.mkdir(parents=True, exist_ok=True)

        general_macros_path = macros_path / GENERAL_MACROS
        shutil.copy(OMC3_MADX_MACROS_DIR / GENERAL_MACROS, general_macros_path)
    
    def _clean_models(self):
        """ Remove models from previous runs. """
        for twiss_file in (self.twiss_forward, self.twiss_forward_corrected, self.twiss_backward, self.twiss_backward_corrected):
            output_twiss: Path = self.output_dir / twiss_file
            output_twiss.unlink(missing_ok=True)

    def _create_measurement_file(self):
        meas_dict = {}
        for measurable in self.measurables:
            meas_dict.update(measurable.init_conditions_dict())
        meas_file_content = "\n".join(f"{k!s} = {v!s};" for k, v in meas_dict.items())
        output_file = self.output_dir / self.measurement_madx
        output_file.write_text(meas_file_content)

    def _create_corrections_file(self):
        output_file = self.output_dir / self.corrections_madx
        if self.corrections is None:
            output_file.unlink(missing_ok=True)
            return

        if isinstance(self.corrections, dict):
            content = "\n".join(f"{k!s} = {v!s};" for k, v in self.corrections.items())
            output_file.write_text(content)
            return

        if isinstance(self.corrections, str | Path):
            corrections = Path(self.corrections)
            if not corrections.exists():
                raise FileNotFoundError(f"File {corrections!s} does not exist.")

            if corrections.resolve() == output_file.resolve():
                return  # already exists -> do nothing
            
            output_file.write_text(corrections.read_text())  # copy
            return

        raise NotImplementedError("Could not determine type of corrections. Aborting.")
    
    def get_madx_script(self):
        accel: Accelerator = self.accel
        madx_script = self.get_base_madx_script()
        
        if self._sequence_name is None:
            raise ValueError(
                "To get the default Segment-by-Segment MAD-X script, "
                f"the derived class '{self.__class__.__name__}'"
                " must set the '_sequence_name' attribute.\n"
                "This error should only be encountered during development. "
                "If you encounter it later, please open an issue!")

        madx_script += "\n".join([
             "",
            f"! ----- Segment-by-Segment propagation for {self.segment.name} -----",
             "",
            f"call, file = '{accel.model_dir / MACROS_DIR / GENERAL_MACROS}';",
             "",
             "! Cycle the sequence to avoid negative length.",
            f"seqedit, sequence={self._sequence_name};",
             "    flatten;",
            f"    cycle, start={self.segment.start};",
             "endedit;",
             "",
            f"use, sequence = {self._sequence_name};",
             "",
             "twiss;",
             "exec, save_initial_and_final_values(",
            f"    {self._sequence_name},",
            f"    {self.segment.start},",
            f"    {self.segment.end}, ",
            f"    \"{accel.model_dir / self.measurement_madx!s}\",",
             "    biniSbSParams,",
             "    bendSbSParams",
             ");",
             "",
             "exec, extract_segment_sequence(",
            f"    {self._sequence_name},",
             "    forward_SbSSEQ,",
             "    backward_SbSSEQ,",
            f"    {self.segment.start},",
            f"    {self.segment.end},",
             ");",
             "",
             "beam, particle = proton, sequence=forward_SbSSEQ;",
             "beam, particle = proton, sequence=backward_SbSSEQ;",
            "",
            f"exec, twiss_segment(forward_SbSSEQ, \"{self.twiss_forward!s}\", biniSbSParams);",
            f"exec, twiss_segment(backward_SbSSEQ, \"{self.twiss_backward!s}\", bendSbSParams);",
             "",
        ])

        if self.corrections is not None:
            madx_script += "\n".join([
                f"call, file=\"{self.corrections_madx!s}\";",
                f"exec, twiss_segment(forward_SbSSEQ, "
                f"\"{self.twiss_forward_corrected}\", biniSbSParams);",
                f"exec, twiss_segment(backward_SbSSEQ, "
                f"\"{self.twiss_backward_corrected}\", bendSbSParams);",
                "",
            ])

        return madx_script
    
    @property
    def files_to_check(self) -> list[str]:
        check_files = [self.twiss_forward, self.twiss_backward]
        if self.corrections is not None:
            check_files += [self.twiss_backward_corrected, self.twiss_backward_corrected]
        return check_files


class CorrectionModelCreator(ModelCreator):
    jobfile = None  # set in __init__ 
    
    def __init__(self, accel: Accelerator, twiss_out: Path | str, corr_files: Sequence[Path | str], update_dpp: bool = False):
        """Model creator for the corrected/matched model of the LHC.

        Args:
            accel (Accelerator): Accelerator Class Instance.
            twiss_out (Path | str): Path to the twiss(-elements) file to write.
            corr_files (Sequence[Path]): Sequence of correction/matching files.
            update_dpp (bool): Whether to update the dpp in the machine.
        """
        LOGGER.debug("Initializing Correction Model Creator Base Attributes")
        super(CorrectionModelCreator, self).__init__(accel)
        self.twiss_out = Path(twiss_out)

        # use absolute paths to force files into twiss_out directory instead of model-dir
        self.jobfile = self.twiss_out.parent.absolute() / f"job.create_{self.twiss_out.stem}.madx"
        self.logfile= self.twiss_out.parent.absolute() / f"job.create_{self.twiss_out.stem}.log"
        self.corr_files = corr_files
        self.update_dpp = update_dpp
    
    def get_madx_script(self) -> str:
        """ Get the madx script for the correction model creator, which updates the model after correcion. 
        
        This is a basic implementation which does not update the dpp, but should work for generic accelerators.
        """  
        if self.update_dpp:
            raise NotImplementedError(
                f"Updating the dpp is not implemented for correction model creator of {self.accel.NAME}."
            )

        madx_script = self.get_base_madx_script()  # do not get_madx_script as we don't need the uncorrected output. 

        for corr_file in self.corr_files:  # Load the corrections
            madx_script += f"call, file = '{corr_file!s}';\n"

        madx_script += (
            f"{self._get_select_command()}\n"
            f"twiss, file = {self.twiss_out!s};\n"
        )
        return madx_script
    
    
    @property
    def files_to_check(self) -> list[str]:
        return [self.twiss_out, self.jobfile, self.logfile]


# Helper functions -------------------------------------------------------------

def check_folder_choices(
    parent: Path,
    msg: str,
    selection: str,
    list_choices: bool = False,
    predicate=iotools.always_true,
    stem_only: bool = False,
    ) -> Path | str:
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
        stem_only (bool): If True, only the stem of the path is checked.
    
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
    choices = [d.stem if stem_only else d.name for d in parent.iterdir() if predicate(d)]

    if selection in choices:
        return parent / selection

    if list_choices:
        for choice in choices:
            print(choice)
    raise AcceleratorDefinitionError(f"{msg}.\nSelected: '{selection}'.\nChoices: [{', '.join(choices)}]")
