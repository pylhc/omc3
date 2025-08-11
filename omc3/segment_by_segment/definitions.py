"""
Definitions
-----------

This module provides definitions to be used with segment by segment
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class MadXBoundaryConditions:
    """Store all boundary conditions for a Mad-X twiss."""
    alfx: float = None
    alfy: float = None
    betx: float = None
    bety: float = None
    dx: float = None
    dy: float = None
    dpx: float = None
    dpy: float = None
    wx: float = None
    wy: float = None
    dphix: float = None
    dphiy: float = None
    r11: float = None
    r12: float = None
    r21: float = None
    r22: float = None

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) is not None}
