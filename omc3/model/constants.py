"""
Constants
---------

This module provides high-level functions to manage most functionality of ``model``.
Specific constants to be used in ``model``, to help with consistency.
"""
from pathlib import Path

from omc3.utils.misc import StrEnum

PLANE_TO_HV: dict[str, str] = dict(X="H", Y="V")

# Job Creation Files ---
OBS_POINTS: str = "observation_points.def"
MODIFIERS_MADX: str = 'modifiers.madx'
ERROR_DEFFS_TXT: str = "error_deffs.txt"
JOB_MODEL_MADX_MASK: str = "job.create_model_{}.madx"
JOB_MODEL_MADX_NOMINAL: str = JOB_MODEL_MADX_MASK.format("nominal")
JOB_MODEL_MADX_BEST_KNOWLEDGE: str = JOB_MODEL_MADX_MASK.format("best_knowledge")

# Variables and Content ---
MADX_ENERGY_VAR: str = "omc3_beam_energy"
MODIFIER_TAG: str = "!@modifier"

# Twiss Output Files ---
TWISS_BEST_KNOWLEDGE_DAT: str = "twiss_best_knowledge.dat"
TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT: str = "twiss_elements_best_knowledge.dat"
TWISS_ADT_DAT: str = "twiss_adt.dat"
TWISS_AC_DAT: str = "twiss_ac.dat"
TWISS_ELEMENTS_DAT: str = "twiss_elements.dat"
TWISS_DAT: str = "twiss.dat"

# Fetcher command names
class Fetcher(StrEnum):
    PATH = "path"
    AFS = "afs"
    GIT = "git"
    LSA = "lsa"

# Macros ---
MACROS_DIR: str = "macros"
GENERAL_MACROS: str = "general.macros.madx"
LHC_MACROS: str = "lhc.macros.madx"
LHC_MACROS_RUN3: str = "lhc.macros.run3.madx"
OMC3_MADX_MACROS_DIR: Path = Path(__file__).parent / "madx_macros"

# b2 Errors ---
AFS_B2_ERRORS_ROOT = Path("/afs/cern.ch/eng/sl/lintrack/error_tables/")
B2_SETTINGS_MADX: str = "b2_settings.madx"
B2_ERRORS_TFS: str = "b2_errors.tfs"

# Acc Models ---
AFS_ACCELERATOR_MODEL_REPOSITORY: Path = Path("/afs/cern.ch/eng/acc-models")
ACC_MODELS_PREFIX: str = AFS_ACCELERATOR_MODEL_REPOSITORY.name
OPTICS_SUBDIR: Path = Path("operation/optics")
STRENGTHS_SUBDIR: Path = Path("strengths")
LHC_REMOVE_TRIPLET_SYMMETRY_RELPATH: Path = Path("toolkit/remove-triplet-symmetry-knob.madx")

