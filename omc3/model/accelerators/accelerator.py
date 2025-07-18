"""
Accelerator
-----------

This module provides high-level classes to define most functionality of ``model.accelerators``.
It contains entrypoint the parent `Accelerator` class as well as other support classes.
"""
from __future__ import annotations

import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tfs
from generic_parser.entrypoint_parser import EntryPointParameters

from omc3.harpy.constants import COL_NAME as NAME
from omc3.model.constants import (
    ERROR_DEFFS_TXT,
    JOB_MODEL_MADX_NOMINAL,
    MODIFIER_TAG,
    MODIFIERS_MADX,
    TWISS_AC_DAT,
    TWISS_ADT_DAT,
    TWISS_BEST_KNOWLEDGE_DAT,
    TWISS_DAT,
    TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT,
    TWISS_ELEMENTS_DAT,
)
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, find_file
from generic_parser.entry_datatypes import get_multi_class


LOG = logging_tools.get_logger(__name__)
CURRENT_DIR = Path(__file__).parent


class AccExcitationMode:  # TODO: use enum! (jdilly, 2025)
    # it is very important that FREE = 0
    FREE, ACD, ADT = range(3)


DRIVEN_EXCITATIONS = dict(acd=AccExcitationMode.ACD, adt=AccExcitationMode.ADT)


class AccElementTypes:
    """Defines the strings for the element types ``BPMS``, ``MAGNETS`` and ``ARC_BPMS``."""

    BPMS: str = "bpm"
    MAGNETS: str = "magnet"
    ARC_BPMS: str = "arc_bpm"


class Accelerator:
    """
    Abstract class to serve as an interface to implement the rest of the accelerators.
    """
    NAME: str
    LOCAL_REPO_NAME: str | None = None
    # RE_DICT needs to use MAD-X compatible regex patterns (jdilly, 2021)
    RE_DICT: dict[str, str] = {
        AccElementTypes.BPMS: r"^B.*",
        AccElementTypes.MAGNETS: r".*",
        AccElementTypes.ARC_BPMS: r"^B.*",
    }

    @staticmethod
    def get_parameters():
        params = EntryPointParameters()
        params.add_parameter(
            name="model_dir",
            type=PathOrStr,
            help="Path to model directory; loads tunes and excitation from model!",
        )
        params.add_parameter(
            name="nat_tunes",
            type=float,
            nargs=2,
            help="Natural tunes without integer part.",
        )
        params.add_parameter(
            name="drv_tunes",
            type=float,
            nargs=2,
            help="Driven tunes without integer part.",
        )
        params.add_parameter(
            name="driven_excitation",
            type=str,
            choices=("acd", "adt"),
            help="Denotes driven excitation by AC-dipole (acd) or by ADT (adt)",
        )
        params.add_parameter(
            name="dpp",
            default=0.0,
            type=float,
            help="Delta p/p to use.",
        )
        params.add_parameter(
            name="energy",
            type=get_multi_class(float, int),
            help="Energy in GeV.",
        )
        params.add_parameter(
            name="modifiers",
            type=PathOrStr,
            nargs="*",
            help="Path to the optics file to use (modifiers file).",
        )
        params.add_parameter(
            name="xing", action="store_true", help="If True, x-ing angles will be applied to model"
        )
        return params

    def __init__(self, opt) -> None:
        self.model_dir = None
        self.nat_tunes = None
        self.drv_tunes = None
        self.excitation = AccExcitationMode.FREE
        self.model = None
        self._model_driven = None
        self.model_best_knowledge = None
        self.elements = None
        self.elements_best_knowledge = None
        self.error_defs_file = None
        self.modifiers = None
        self.acc_model_path = None
        self._beam_direction = 1
        self._beam = None
        self._ring = None
        self.energy = None
        self.dpp = 0.0
        self.xing = None

        if opt.model_dir:
            if (opt.nat_tunes is not None) or (opt.drv_tunes is not None):
                raise AcceleratorDefinitionError(
                    "Arguments 'nat_tunes' and 'driven_tunes' are "
                    "not allowed when loading from model directory."
                )

            self.init_from_model_dir(Path(opt.model_dir))

            if opt.modifiers is not None:
                if self.modifiers is not None:
                    LOG.warning(
                        "Modifier definitions given but also found in model directory. Using the user-given ones."
                    )  # useful if model was copied to a new location, but modifier paths have changed
                self.modifiers = opt.modifiers

        else:
            self.init_from_options(opt)

    def init_from_options(self, opt):
        self.nat_tunes = opt.nat_tunes

        if opt.driven_excitation is not None:
            self.drv_tunes = opt.drv_tunes
            self.excitation = DRIVEN_EXCITATIONS[opt.driven_excitation]

        # optional with default
        self.dpp = opt.dpp

        # optional no default
        self.energy = opt.get("energy", None)
        self.xing = opt.get("xing", None)
        self.modifiers = opt.get("modifiers", None)

    def init_from_model_dir(self, model_dir: Path) -> None:
        LOG.debug("Creating accelerator instance from model dir")
        self.model_dir = model_dir
        # Elements #####################################
        elements_path = model_dir / TWISS_ELEMENTS_DAT
        if not elements_path.is_file():
            raise AcceleratorDefinitionError("Elements twiss not found")
        self.elements = tfs.read(elements_path, index=NAME)

        LOG.debug(f"  model path = {model_dir / TWISS_DAT}")
        try:
            self.model = tfs.read(model_dir / TWISS_DAT, index=NAME)
        except IOError:
            bpm_mask = self.elements.index.str.match(self.RE_DICT[AccElementTypes.BPMS])
            self.model = self.elements.loc[bpm_mask, :]
        self.nat_tunes = [float(self.model.headers["Q1"]), float(self.model.headers["Q2"])]
        try:
            self.energy = float(self.model.headers["ENERGY"])  # always 450GeV because we do not set it anywhere properly...
        except KeyError: #KEK model does not have energy in the header
            pass

        # Excitations #####################################
        driven_filenames = dict(acd=model_dir / TWISS_AC_DAT, adt=model_dir / TWISS_ADT_DAT)
        if driven_filenames["acd"].is_file() and driven_filenames["adt"].is_file():
            raise AcceleratorDefinitionError("ADT as well as ACD models provided. Choose only one.")
        for key in driven_filenames.keys():
            if driven_filenames[key].is_file():
                self.excitation = DRIVEN_EXCITATIONS[key]
                self.model_driven = tfs.read(driven_filenames[key], index=NAME)

        if not self.excitation == AccExcitationMode.FREE:
            self.drv_tunes = [self.model_driven.headers["Q1"], self.model_driven.headers["Q2"]]

        # Best Knowledge #####################################
        best_knowledge_path = model_dir / TWISS_BEST_KNOWLEDGE_DAT
        if best_knowledge_path.is_file():
            self.model_best_knowledge = tfs.read(best_knowledge_path, index=NAME)

        best_knowledge_elements_path = model_dir / TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT
        if best_knowledge_elements_path.is_file():
            self.elements_best_knowledge = tfs.read(best_knowledge_elements_path, index=NAME)

        # Base Model ########################################
        if self.LOCAL_REPO_NAME is not None:
            acc_models = model_dir / self.LOCAL_REPO_NAME

            if acc_models.is_dir():
                if acc_models.is_symlink():
                    self.acc_model_path = Path(os.readlink(acc_models)).absolute()
                else:
                    self.acc_model_path = acc_models
            # else this wasn't an acc-models based model

        # Modifiers #########################################
        self.modifiers = _get_modifiers_from_modeldir(model_dir)

        # Error Def #####################################
        errordefspath = self.model_dir / ERROR_DEFFS_TXT
        if errordefspath.is_file():
            self.error_defs_file = errordefspath

    def find_modifier(self, modifier: Path | str):
        """ Try to find a modifier file, which might be given only by its name.
        By default this is looking for full-path, model-dir and in the acc-models-path,
        but should probably be overwritten by the accelerator sub-classes.
        """
        dirs = []
        if self.model_dir is not None:
            dirs.append(self.model_dir)

        if self.acc_model_path is not None:
            dirs.append(self.acc_model_path)

        return find_file(modifier, dirs=dirs)

    @property
    def beam_direction(self) -> int:
        return self._beam_direction

    @beam_direction.setter
    def beam_direction(self, value: int) -> None:
        if value not in (1, -1):
            raise AcceleratorDefinitionError("Beam direction has to be either 1 or -1")
        self._beam_direction = value

    def verify_object(self):
        """
        Verifies that this instance of an `Accelerator` is properly instantiated.
        """
        # since we removed `required` args, we check here if everything has been passed
        if self.model_dir is None:
            if self.nat_tunes is None:
                raise AttributeError("Natural tunes not set (missing `--nat_tunes` flag?)")
            if self.excitation != AccExcitationMode.FREE and self.drv_tunes is None:
                raise AttributeError("Driven excitation selected but no driven tunes given (missing `--drv_tunes` flag?)")

    def get_exciter_bpm(self, plane: str, commonbpms: list[str]):
        """
        Returns the BPM next to the exciter.
        The `Accelerator` instance knows already which excitation method is used.

        Args:
            plane: **X** or **Y**.
            commonbpms: list of common BPMs (e.g. intersection of input BPMs.

        Returns:
            `((index, bpm_name), exciter_name): tuple(int, str), str)`
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    def important_phase_advances(self):
        return []

    @property
    def model_driven(self) -> tfs.TfsDataFrame:
        if self._model_driven is None:
            raise AttributeError("No driven model given in this accelerator instance.")
        return self._model_driven

    @model_driven.setter
    def model_driven(self, value):
        if self.excitation == AccExcitationMode.FREE:
            raise AcceleratorDefinitionError("Driven model cannot be set for accelerator with free excitation mode.")
        self._model_driven = value

    # Class methods ###########################################

    @classmethod
    def get_element_types_mask(cls, list_of_elements: list[str], types) -> np.ndarray:
        """
        Returns a boolean mask for elements in ``list_of_elements`` that belong to any of the
        specified types.
        Needs to handle: `bpm`, `magnet`, `arc_bpm` (see :class:`AccElementTypes`)

        Args:
            list_of_elements: list of elements.
            types: the kinds of elements to look for.

        Returns:
            A boolean array of elements of specified kinds.
        """
        unknown_elements = [ty for ty in types if ty not in cls.RE_DICT]
        if len(unknown_elements):
            raise TypeError(f"Unknown element(s): '{unknown_elements}'")
        series = pd.Series(list_of_elements)
        mask = series.str.match(cls.RE_DICT[types[0]], case=False)
        for ty in types[1:]:
            mask = mask | series.str.match(cls.RE_DICT[ty], case=False)
        return mask.to_numpy()

    @classmethod
    def get_variables(cls, frm=None, to=None, classes=None):
        """
        Gets the variables with elements in the given range and the given classes. ``None`` means
        everything.
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    @classmethod
    def get_dir(cls) -> Path:
        """Default directory for accelerator. Should be overwritten if more specific."""
        dir_path = CURRENT_DIR / cls.NAME
        if dir_path.exists():
            return dir_path
        raise NotImplementedError(f"No directory for accelerator {cls.NAME} available.")

    @classmethod
    def get_file(cls, filename: str) -> Path:
        """Default location for accelerator files. Should be overwritten if more specific."""
        file_path = CURRENT_DIR / cls.NAME / filename
        if file_path.exists():
            return file_path
        raise NotImplementedError(
            f"File {file_path.name} not available for accelerator {cls.NAME}."
        )

    ##########################################################################


class AcceleratorDefinitionError(Exception):
    """
    Raised when an `Accelerator` instance is wrongly used, for example by calling a method that
    should have been overwritten.
    """

    pass


# Helper ----


def _get_modifiers_from_modeldir(model_dir: Path) -> list[Path]:
    """Parse modifiers from job.create_model.madx or use modifiers.madx file."""
    job_file = model_dir / JOB_MODEL_MADX_NOMINAL
    if job_file.is_file():
        return find_called_files_with_tag(job_file, MODIFIER_TAG)

    # Legacy
    modifiers_file = model_dir / MODIFIERS_MADX
    if modifiers_file.exists():  # legacy
        return [modifiers_file]

    return None


def find_called_files_with_tag(madx_file: Path, tag: str) -> list[Path] | None:
    """ Parse lines that call a file and are tagged with the given tag and return
    a list of paths to these files.

    This is mainly used to find the modifier tag in lines and return called file in these lines.

    The modifier tag is used by the model creator to mark which line defines modifiers
    see e.g. `get_base_madx_script()` in `lhc.py`

    An example for a match to the regex: `call, file = 'modifiers.madx'; !@modifiers`.
    """
    if not madx_file.is_file():
        return None

    job_madx = madx_file.read_text()

    called_files = re.findall(
        fr"\s*call,\s*file\s*=\s*[\"\']?([^;\'\"]+)[\"\']?\s*;\s*{tag}",
        job_madx,
        flags=re.IGNORECASE,
    )
    called_files = [Path(m) for m in called_files]
    return called_files or None
