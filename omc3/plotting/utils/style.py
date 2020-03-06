"""
Plotting Utilities: Style
-----------------------------------

Helper functions to make the most awesome* plots out there.

* please feel free to add more stuff

:module: omc3.plotting.utils.style

"""
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path


REMOVE_ENTRY = "REMOVE ENTRY"  # id to remove entries in manual style


STYLES = dict(standard=Path(__file__).parent / 'standard.mplstyle',
              presentation=Path(__file__).parent / 'presentation.mplstyle'
              )

# Style ######################################################################


def set_style(styles=('standard',), manual=None):
    """Sets the style for all following plots.

    Args:
        styles: List of styles, either 'standard', 'presentation' or one of the mpl styles
        manual: Dict of manual parameters to update. Convention: "REMOVE_ENTRY" removes entry
    """
    styles = [STYLES.get(style, style) for style in styles]
    if manual:
        for key, value in manual.items():
            if value == REMOVE_ENTRY:
                manual[key] = matplotlib.rcParams[key]
        styles.append(manual)
    plt.style.use(styles)
