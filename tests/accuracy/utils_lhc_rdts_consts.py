from pathlib import Path

import omc3

PACKAGE_DIR: Path = Path(omc3.__file__).parent.parent.absolute()
LHC_RDTS_TEST_DIR: Path = PACKAGE_DIR / "tests" / "inputs" / "lhc_rdts"

ANALYSIS_DIR : Path = LHC_RDTS_TEST_DIR / "analysis"
DATA_DIR     : Path = LHC_RDTS_TEST_DIR / "data"
FREQ_OUT_DIR : Path = ANALYSIS_DIR / "lin_files"

# The RDTs here are all the normal and skew RDTs for sextupoles and octupoles
# that are not on any of the tune lines, as these will not be calculated in OMC3 correctly.
NORMAL_SEXTUPOLE_RDTS: tuple[str] = (
    "f1200_x",
    "f3000_x",
    "f1002_x",
    "f1020_x",
    "f0111_y",
    "f0120_y",
    "f1011_y",
    "f1020_y",
)
SKEW_SEXTUPOLE_RDTS: tuple[str] = (
    "f0012_y",
    "f0030_y",
    "f1101_x",
    "f1110_x",
    "f2001_x",
    "f2010_x",
    "f0210_y",
    "f2010_y",
)
NORMAL_OCTUPOLE_RDTS: tuple[str] = (
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
)
SKEW_OCTUPOLE_RDTS: tuple[str] = (
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
)

MODEL_NG_PREFIX: str = "model_ng"
MODEL_X_PREFIX : str = "model_x"
MODEL_ANALYTICAL_PREFIX: str = "analytical_model"

# RUN SETTINGS
NTURNS: int = 1000
KICK_AMP: float = 1e-3
SEXTUPOLE_STRENGTH: float = 3e-5
OCTUPOLE_STRENGTH : float = 3e-3
