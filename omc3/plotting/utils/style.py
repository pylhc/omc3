"""
Plotting Utilities: Style
-------------------------

Helper functions to style plots.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib import rcParams

from omc3.utils.iotools import PathOrStr

REMOVE_ENTRY = "REMOVE ENTRY"  # id to remove entries in manual style

STYLES_DIR = Path(__file__).parent.parent / 'styles'


# Style ######################################################################


def omc3_styles():
    return {p.with_suffix('').name: p for p in STYLES_DIR.glob('*.mplstyle')}


def set_style(styles: Path | str | Sequence[Path | str] = 'standard',
              manual: dict[str, Any] = None):
    """
    Sets the style for all following plots.

    Args:
        styles: `List` of styles (or single string), either path to style-file, name of style in
            styles or from the mpl styles
        manual: `Dict` of manual parameters to update. Convention: ``REMOVE_ENTRY`` removes entry
                from given styles, i.e. falls back to mpl default.
    """
    if isinstance(styles, PathOrStr):
        styles = (styles,)

    local_styles = omc3_styles()
    styles = [local_styles.get(style, style) for style in styles]

    if manual:
        for key, value in manual.items():
            if value == REMOVE_ENTRY:
                manual[key] = rcParams[key]
        styles.append(manual)
    plt.style.use(styles)
