"""
Constants
---------

This module provides high-level functions to manage most functionality of ``model``.
Specific constants to be used in ``model``, to help with consistency.
"""

from pathlib import Path

MACROS_DIR = "macros"
OBS_POINTS = "observation_points.def"
MODIFIERS_MADX = 'modifiers.madx'
MODIFIER_TAG = "!@modifier"
TWISS_BEST_KNOWLEDGE_DAT = "twiss_best_knowledge.dat"
TWISS_ELEMENTS_BEST_KNOWLEDGE_DAT = "twiss_elements_best_knowledge.dat"
TWISS_ADT_DAT = "twiss_adt.dat"
TWISS_AC_DAT = "twiss_ac.dat"
TWISS_ELEMENTS_DAT = "twiss_elements.dat"
TWISS_DAT = "twiss.dat"
ERROR_DEFFS_TXT = "error_deffs.txt"
JOB_MODEL_MADX = "job.create_model.madx"

GENERAL_MACROS = "general.macros.madx"
LHC_MACROS = "lhc.macros.madx"
LHC_MACROS_RUN3 = "lhc.macros.run3.madx"

B2_SETTINGS_MADX = "b2_settings.madx"
B2_ERRORS_TFS = "b2_errors.tfs"
PLANE_TO_HV = dict(X="H", Y="V")

ACCELERATOR_MODEL_REPOSITORY = Path("/afs/cern.ch/eng/acc-models/lhc")