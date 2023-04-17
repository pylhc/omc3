"""
Plotting Utilities: Lines
-------------------------

Line-plotting related functionality.
"""
import matplotlib
import matplotlib.transforms as mtrans
import numpy as np
from matplotlib import transforms
from matplotlib.markers import MarkerStyle
from matplotlib.patches import PathPatch

VERTICAL_LINES_TEXT_LOCATIONS = {
    'bottom': dict(y=-0.01, va='top', ha='center'),
    'top': dict(y=1.01, va='bottom', ha='center'),
    'line bottom': dict(y=0.01, va='bottom', ha='right', rotation=90),
    'line top': dict(y=0.99, va='top', ha='right', rotation=90),
}
VERTICAL_LINES_ALPHA = 0.5


class MarkerList(object):
    """Create a list of predefined markers."""
    # markers = ["s", "o", ">", "D", "v", "*", "h", "^", "p", "X", "<", "P"]  # matplotlib 2.++
    markers = ["s", "o", ">", "D", "v", "*", "h", "^", "p", "<"]

    def __init__(self):
        self.idx = 0

    @classmethod
    def get_marker(cls, marker_num):
        """ Return marker of index marker_num

         Args:
             marker_num (int): return maker at this position in list (mod len(list))
        """
        return cls.markers[marker_num % len(cls.markers)]

    def get_next_marker(self):
        """ Return the next marker in the list (circularly wrapped) """
        marker = self.get_marker(self.idx)
        self.idx += 1
        return marker


def text_to_marker(text: str, center: bool = True) -> np.ndarray:
    """
    Convert the given `text` to path
    which can be used as a marker in matplotlib plots,
    including the options of setting `markerfacecolor` and `markeredgecolor`.

    Args:
        text (str): Text to use as a marker.
        center (bool): Center the path around the origin (otherwise bottom left is anchor).

    Returns:
        Array of the path vertices.
    """
    path = MarkerStyle(fr'$\mathrm{{{text}}}$').get_path()

    if center:
        # center path: remove any offsets (extend.min) and move to center (-size[0]/2)
        extend = path.get_extents()
        t = mtrans.Affine2D().translate(-extend.min[0] - extend.size[0] / 2, -extend.min[1] - extend.size[1] / 2)
        path = path.transformed(t)

    pp = PathPatch(path, transform=mtrans.IdentityTransform())
    return pp.get_path()


def plot_vertical_lines_fast(ax, x, y=(0, 1), **kwargs):
    """
    Plots vertical lines, similar to axvline, but also in 3D, all at once and with only one label.

    Args:
        ax: Axis to plot on.
        x: array of x-positions (data coordinates).
        y: tuple of y-limits (axis coordinates, default (0,1) i.e. bottom to top).
        kwargs: kwargs passed on to ax.plot.
    """
    trans = mtrans.blended_transform_factory(ax.transData, ax.transAxes)  # x is data, y is axes
    ax.plot(np.repeat(x, 3), np.tile([*y, np.nan], len(x)), transform=trans, **kwargs)


def plot_vertical_line(ax, axvline_args: dict, text: str = None, text_loc: str = None,
                       label_size: float = matplotlib.rcParams['font.size']):
    """
    Plot a vertical line into the plot, where mline is a dictionary with arguments for ``axvline``.
    Advanced capabilities include: Automatic alpha value (if not overwritten), automatic
    zorder = -1 (if not overwritten), and adding a label to the line, where the location is given
    by text_loc.
    """
    axvline_args.setdefault('alpha', VERTICAL_LINES_ALPHA)
    axvline_args.setdefault('marker', 'None')
    axvline_args.setdefault('zorder', -1)
    if 'linestyle' not in axvline_args and 'ls' not in axvline_args:
        axvline_args['linestyle'] = '--'

    label = axvline_args.get('label', None)
    if label is None:
        axvline_args['label'] = "__nolegend__"

    line = ax.axvline(**axvline_args)

    if text is not None and text_loc is not None:
        if text_loc not in VERTICAL_LINES_TEXT_LOCATIONS:
            raise ValueError(f"Unknown value '{text_loc}' for label location.")

        ax.text(x=axvline_args['x'], s=text,
                transform=transforms.blended_transform_factory(ax.transData, ax.transAxes),
                color=line.get_color(),
                fontdict={'size': label_size}, **VERTICAL_LINES_TEXT_LOCATIONS[text_loc])
