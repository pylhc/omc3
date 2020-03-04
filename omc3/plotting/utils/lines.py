"""
Plotting Utilities: Lines
---------------------------------

Line-plotting related functionality.

:module: omc3.plotting.utils.lines

"""
import matplotlib.transforms as mtrans
import numpy as np


class MarkerList(object):
    """ Create a list of predefined markers """
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


def vertical_lines(ax, x, y=(0, 1), **kwargs):
    """ Plots vertical lines, similar to axvline, but also in 3D, all at once and with only one label.

        Args:
            ax: Axis to plot on
            x: array of x-positions (data coordinates)
            y: tuple of y-limits (axis coordinates, default (0,1) i.e. bottom to top)
            kwargs: kwargs passed on to ax.plot
    """
    trans = mtrans.blended_transform_factory(ax.transData, ax.transAxes)  # x is data, y is axes
    ax.plot(np.repeat(x, 3), np.tile([*y, np.nan], len(x)), transform=trans, **kwargs)
