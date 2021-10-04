"""
Accelerator
-----------

This module provides high-level classes to define most functionality of ``model.accelerators``.
It contains entrypoint the parent `Accelerator` class as well as other support classes.
"""
import re
from pathlib import Path
from typing import List

import numpy
import pandas as pd
import tfs
from generic_parser.entrypoint_parser import EntryPointParameters

from omc3.model.constants import (
    ERROR_DEFFS_TXT,
    JOB_MODEL_MADX,
    MODIFIER_TAG,
    MODIFIERS_MADX,
    TWISS_AC_DAT,
    TWISS_ADT_DAT,
    TWISS_BEST_KNOWLEDGE_DAT,
    TWISS_DAT,
    TWISS_ELEMENTS_DAT,
)
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr

LOG = logging_tools.get_logger(__name__)
CURRENT_DIR = Path(__file__).parent


class AccExcitationMode:
    # it is very important that FREE = 0
    FREE, ACD, ADT = range(3)


DRIVEN_EXCITATIONS = dict(acd=AccExcitationMode.ACD, adt=AccExcitationMode.ADT)


class AccElementTypes:
    """Defines the strings for the element types ``BPMS``, ``MAGNETS`` and ``ARC_BPMS``."""

    BPMS = "bpm"
    MAGNETS = "magnet"
    ARC_BPMS = "arc_bpm"


class Accelerator:
    """
    Abstract class to serve as an interface to implement the rest of the accelerators.
    """
    # RE_DICT needs to use MAD-X compatible patterns (jdilly, 2021)
    RE_DICT = {
        AccElementTypes.BPMS: r".*",
        AccElementTypes.MAGNETS: r".*",
        AccElementTypes.ARC_BPMS: r".*",
    }
    BPM_INITIAL = "B"

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
            type=float,
            help="Energy in Tev.",
        )
        params.add_parameter(
            name="modifiers",
            type=Path,
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
        self.error_defs_file = None
        self.modifiers = None
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

        else:
            self.init_from_options(opt)

    def init_from_options(self, opt) -> None:
        if opt.nat_tunes is None:
            raise AcceleratorDefinitionError("Argument 'nat_tunes' is required.")
        if (opt.drv_tunes is None) and (opt.driven_excitation is not None):
            raise AcceleratorDefinitionError("Argument 'drv_tunes' is required.")
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
        self.elements = tfs.read(elements_path, index="NAME")

        LOG.debug(f"  model path = {model_dir / TWISS_DAT}")
        try:
            self.model = tfs.read(model_dir / TWISS_DAT, index="NAME")
        except IOError:
            bpm_index = [
                idx for idx in self.elements.index.to_numpy() if idx.startswith(self.BPM_INITIAL)
            ]
            self.model = self.elements.loc[bpm_index, :]
        self.nat_tunes = [float(self.model.headers["Q1"]), float(self.model.headers["Q2"])]
        # self.energy = float(self.model.headers["ENERGY"]) * 1e-3  # TODO not the same Energy (jdilly, 2021)

        # Excitations #####################################
        driven_filenames = dict(acd=model_dir / TWISS_AC_DAT, adt=model_dir / TWISS_ADT_DAT)
        if driven_filenames["acd"].is_file() and driven_filenames["adt"].is_file():
            raise AcceleratorDefinitionError("ADT as well as ACD models provided. Choose only one.")
        for key in driven_filenames.keys():
            if driven_filenames[key].is_file():
                self._model_driven = tfs.read(driven_filenames[key], index="NAME")
                self.excitation = DRIVEN_EXCITATIONS[key]

        if not self.excitation == AccExcitationMode.FREE:
            self.drv_tunes = [self.model_driven.headers["Q1"], self.model_driven.headers["Q2"]]

        # Best Knowledge #####################################
        best_knowledge_path = model_dir / TWISS_BEST_KNOWLEDGE_DAT
        if best_knowledge_path.is_file():
            self.model_best_knowledge = tfs.read(best_knowledge_path, index="NAME")

        # Modifiers #########################################
        self.modifiers = _get_modifiers_from_modeldir(model_dir)

        # Error Def #####################################
        errordefspath = self.model_dir / ERROR_DEFFS_TXT
        if errordefspath.is_file():
            self.error_defs_file = errordefspath

    # Class methods ###########################################

    @classmethod
    def get_element_types_mask(cls, list_of_elements: List[str], types) -> numpy.ndarray:
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
    def get_correctors_variables(cls, frm=None, to=None, classes=None):
        """
        Returns the set of corrector variables between ``frm`` and ``to``, with classes in
        classes. ``None`` means select all.
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

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
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    def get_exciter_bpm(self, plane, distance):
        """
        Returns the BPM next to the exciter.
        The `Accelerator` instance knows already which excitation method is used.

        Args:
            plane: **X** or **Y**.
            distance: 1=nearest bpm 2=next to nearest bpm.
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    def important_phase_advances(self):
        return []

    @property
    def model_driven(self) -> tfs.TfsDataFrame:
        if self._model_driven is None:
            raise AttributeError("No driven model given in this accelerator instance.")
        return self._model_driven

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

    # Jobs ###################################################################

    def get_update_correction_script(self, tiwss_out_path, corrections_file_path):
        """
        Returns job (string) to create an updated model from changeparameters input (used in
        iterative correction).
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    def get_base_madx_script(self, best_knowledge=False):
        """
        Returns job (string) to create the basic accelerator sequence.
        """
        raise NotImplementedError("A function should have been overwritten, check stack trace.")

    ##########################################################################


class Variable:
    """
    Generic corrector variable class that holds `name`, `position (s)` and physical elements it
    affects. These variables should be logical variables that have and effect in the model if
    modified.
    """

    def __init__(self, name, elements, classes):
        self.name = name
        self.elements = elements
        self.classes = classes


class Element:
    """
    Generic corrector element class that holds `name` and `position (s)` of the corrector. This
    element should represent a physical element of the accelerator.
    """

    def __init__(self, name, s):
        self.name = name
        self.s = s


class AcceleratorDefinitionError(Exception):
    """
    Raised when an `Accelerator` instance is wrongly used, for example by calling a method that
    should have been overwritten.
    """

    pass


# Helper ----


def _get_modifiers_from_modeldir(model_dir: Path) -> List[Path]:
    """Parse modifiers from job.create_model.madx or use modifiers.madx file."""
    job_file = model_dir / JOB_MODEL_MADX
    if job_file.exists():
        job_madx = job_file.read_text()

        # find modifier tag in lines and return called file in these lines
        modifiers = re.findall(
            fr"\s+call,\s*file\s*=\s*[\"\']?([^;\'\"]+)[\"\']?\s*;\s*{MODIFIER_TAG}",
            job_madx,
            flags=re.IGNORECASE,
        )
        modifiers = [Path(m) for m in modifiers]
        return modifiers or None

    # Legacy
    modifiers_file = model_dir / MODIFIERS_MADX
    if modifiers_file.exists():  # legacy
        return [modifiers_file]

    return None
