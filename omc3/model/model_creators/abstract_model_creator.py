"""
Abstract Model Creator Class
----------------------------

This module provides the template for all model creators.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence, Union

from omc3.model.accelerators.accelerator import Accelerator, AccExcitationMode
from omc3.model.constants import TWISS_AC_DAT, TWISS_ADT_DAT, TWISS_DAT, TWISS_ELEMENTS_DAT
import logging

LOG = logging.getLogger(__name__)


class ModelCreator(ABC):
    """
    Abstract class for the implementation of a model creator. All mandatory methods and convenience
    functions are defined here.
    """
    def __init__(self, accel: Accelerator, *args, **kwargs):
        """
        Initialize the Model Creator.

        Args:
            accel (Accelerator): Accelerator Instance
        """
        self.accel = accel
        cleaned_args = [arg for arg in args if arg is not None]
        cleaned_kwargs = {k: v for k, v in kwargs if v is not None}
        if len(cleaned_args):
            LOG.warning(f"Unknown args for Model Creator: {', '.join(cleaned_args)}")

        if len(cleaned_kwargs):
            LOG.warning(f"Unknown kwargs for Model Creator: {cleaned_kwargs!s}")

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
        instance. Called by the ``model_creator.create_accel_and_instance``

        Args:
            accel (Accelerator): Accelerator Instance used for the model creation.
        """
        pass

    def post_run(self) -> None:
        """
        Checks that the model creation ``MAD-X`` run was successful. It should check that the
        appropriate directories are created, and that macros and other files are in place.
        Checks the accelerator instance. Called by the ``model_creator.create_instance_and_model``

        Args:
            opt:
            accel (Accelerator): Accelerator Instance used for the model creation.
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
