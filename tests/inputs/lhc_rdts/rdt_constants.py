import os
from pathlib import Path

import omc3

PACKAGE_DIR = Path(omc3.__file__).parent.parent.absolute()
TEST_DIR = PACKAGE_DIR / "tests" / "inputs" / "lhc_rdts"

ANALYSIS_DIR = TEST_DIR / "analysis"
FREQ_OUT_DIR = ANALYSIS_DIR / "lin_files"
DATA_DIR = TEST_DIR / "data"
ACC_MODELS = TEST_DIR / "acc-models-lhc"
if not ACC_MODELS.exists():
    os.system(f"ln -s /afs/cern.ch/eng/acc-models/lhc/2024/ {ACC_MODELS}")

COUPLING_RDTS = [
    "f1001",
    "f1010",
]

# The RDTs here are all the normal and skew RDTs for sextupoles and octupoles
# that are not on any of the tune lines, as these are 0 when calculated from OMC3.
NORMAL_RDTS3 = [  # Normal Sextupole
    "f1200_x",
    "f3000_x",
    "f1002_x",
    "f1020_x",
    "f0111_y",
    "f0120_y",
    "f1011_y",
    "f1020_y",
]
SKEW_RDTS3 = [  # Skew Sextupole
    "f0012_y",
    "f0030_y",
    "f1101_x",
    "f1110_x",
    "f2001_x",
    "f2010_x",
    "f0210_y",
    "f2010_y",
]
NORMAL_RDTS4 = [  # Normal Octupole
    "f1300_x",
    "f4000_x",
    "f0013_y",
    "f0040_y",
    "f1102_x",
    "f1120_x",
    "f2002_x",
    "f2020_x",
    "f0211_y",
    "f0220_y",
    "f2011_y",
    "f2020_y",
]
SKEW_RDTS4 = [  # Skew Octupole
    "f0112_y",
    "f0130_y",
    "f0310_y",
    "f1003_x",
    "f1012_y",
    "f1030_x",
    "f1030_y",
    "f1201_x",
    "f1210_x",
    "f3001_x",
    "f3010_x",
    "f3010_y",
]

MODEL_NG_PREFIX = "model_ng"
MODEL_X_PREFIX = "model_x"
MODEL_ANALYTICAL_PREFIX = "analytical_model"

# RUN SETTINGS
NTURNS = 1000
KICK_AMP = 1e-3
SEXTUPOLE_STRENGTH = 3e-5
OCTUPOLE_STRENGTH = 3e-3
