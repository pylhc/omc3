"""
Abstract Model Creator Class
----------------------------

This module provides the template for all model creators.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from omc3.model.accelerators.accelerator import Accelerator, AccExcitationMode
from omc3.model.constants import TWISS_AC_DAT, TWISS_DAT, TWISS_ELEMENTS_DAT, TWISS_ADT_DAT


class ModelCreator(ABC):

    @classmethod
    @abstractmethod
    def get_correction_check_script(cls, accel: Accelerator, corr_file: str, chrom: bool) -> str:
        """ Returns the madx-script used to verify global corrections.
        This script should create twissfiles for before (``twiss_no.dat``) and after (``twiss_corr.dat``)
        correction.

        Args:
            accel (Accelerator): Accelerator Instance
            corr_file (str): File containing the corrections (madx-readable)
            chrom (bool): Flag for chromatic corrections deltapm and deltapp

        Returns:
            madx-script used to verify global corrections
        """
        pass

    @classmethod
    @abstractmethod
    def get_madx_script(cls, accel: Accelerator) -> str:
        """ Returns the madx-script used to create the model (directory).

        Args:
            accel (Accelerator): Accelerator Instance

        Returns:
            madx-script used to used to create the model
        """
        pass

    @classmethod
    @abstractmethod
    def prepare_run(cls, accel: Accelerator):
        """ Prepares the model-creation-madx-run.
        I.e. checks that the directories are created, macros and other files are
        in place.
        Should also check that all necessary data for model creation is
        available in the accelerator instance.
        Called by the ``model_creator.create_accel_and_instance``

        Args:
            accel (Accelerator): Accelerator Instance

        """
        pass

    @classmethod
    def check_run_output(cls, accel: Accelerator):
        """ Checks that the model-creation-madx-run was successful.
        I.e. checks that the directories are created, macros and other files are
        in place. Checks accelerator instance.
        Called by the ``model_creator.create_instance_and_model``

        Args:
            accel (Accelerator): Accelerator Instance

        """
        # These are the default files for most model creators for now.
        to_check = [TWISS_DAT, TWISS_ELEMENTS_DAT]
        if accel.excitation == AccExcitationMode.ACD:
            to_check += [TWISS_AC_DAT]
        elif accel.excitation == AccExcitationMode.ADT:
            to_check += [TWISS_ADT_DAT]
        cls._check_files_exist(accel.model_dir, to_check)

    @staticmethod
    def _check_files_exist(dir_: Path, files: Sequence[str]):
        """ Convenience function to loop over files and raise an IOError if they
        do not exist. """
        for out_file in files:
            file_path = dir_ / out_file
            if not file_path.exists():
                raise IOError(f"Model Creation Failed. "
                              f"The file '{file_path}' was not created.")

