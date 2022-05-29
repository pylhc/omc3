"""
Constants
---------

General constants to use throughout ``omc3``, so they don't have to be redefined all the time.
Also helps with consistency.
"""
from typing import Dict, Tuple

import numpy as np

PI: float = np.pi
PI2: float = 2 * np.pi
PI2I: float = 2j * np.pi
PLANES: Tuple[str, str] = ("X", "Y")
PLANE_TO_NUM: Dict[str, int] = dict(X=1, Y=2)
PLANE_TO_HV: Dict[str, str] = dict(X="H", Y="V")

UNIT_IN_METERS: Dict[str, float] = dict(
    km=1e3, m=1e0, mm=1e-3, um=1e-6, nm=1e-9, pm=1e-12, fm=1e-15, am=1e-18
)
