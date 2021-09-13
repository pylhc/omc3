"""
Plotting Utilities: Colors
--------------------------

Helper functions to handle colors in plots.
"""
import colorsys
from itertools import cycle

import matplotlib
from matplotlib import colors as mc


def get_mpl_color(idx=None):
    """Gets the 'new' default ``matplotlib`` colors by index, or the whole cycle."""
    c = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf',  # blue-teal
    ]
    if idx is None:
        return cycle(c)
    return c[idx % len(c)]


def rgb_plotly_to_mpl(rgb_string):
    """Helper function to transforn plotly rbg to ``matplotlib`` rgb."""
    if rgb_string.startswith('#'):
        return rgb_string

    rgb_string = rgb_string.replace("rgba", "").replace("rgb", "")
    rgb = eval(rgb_string)
    rgb_norm = [c/255. for c in rgb]
    return rgb_norm


def change_color_brightness(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount. Input can be
    ``matplotlib`` color string, hex string, or RGB tuple. An amount of 1 equals to no change. 0
    is very bright (white) and 2 is very dark. Original code by Ian Hincks
    Source: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    """
    if not (0<=amount<=2):
        raise ValueError("The brightness change has to be between 0 and 2."
                         " Instead it was {}".format(amount))
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color

    try:
        c = colorsys.rgb_to_hls(*mc.ColorConverter().to_rgb(c))  # matplotlib 1.5
    except AttributeError:
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))  # matplotlib > 2
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def change_ebar_alpha_for_line(ebar, alpha):
    """
    Changes the alpha value of an error-bar container.
    Loop through bars (ebar[1]) and caps (ebar[2]) and set the alpha value.
    """
    for bars_or_caps in ebar[1:]:
        for bar_or_cap in bars_or_caps:
            bar_or_cap.set_alpha(alpha)


def change_ebar_alpha_for_axes(ax, alpha):
    """Wrapper for change_ebar_alpha_for_line to change all in one axes."""
    for ebar in ax.containers:
        if isinstance(ebar, matplotlib.container.ErrorbarContainer):
            change_ebar_alpha_for_line(ebar, alpha)