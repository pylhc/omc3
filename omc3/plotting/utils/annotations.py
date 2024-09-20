"""
Plotting Utilities: Annotations
-------------------------------

Helper functions to create annotations, legends as well as style labels in plots.
"""
from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING

import matplotlib
import pandas as pd
import tfs
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path
    from matplotlib.axes import Axes

# List of common y-labels. Sorry for the ugly.
ylabels: dict[str, str] = {
    "beta":               r'$\beta_{{{0}}} \quad [m]$',
    "betabeat":           r'$\Delta \beta_{{{0}}} \; / \; \beta_{{{0}}}$',
    "betabeat_permile":   r'$\Delta \beta_{{{0}}} \; / \; \beta_{{{0}}} [$'u'\u2030'r'$]$',
    "dbeta":              r"$\beta'_{{{0}}} \quad [m]$",
    "dbetabeat":          r'$1 \; / \; \beta_{{{0}}} \cdot \partial\beta_{{{0}}} \; / \; \partial\delta_{{{0}}}$',
    "norm_dispersion":    r'D$_{{{0}}} \; / \; \sqrt{{\beta_{{{0}}}}} \quad \left[\sqrt{{\rm m}}\right]$',
    "norm_dispersion_mu": r'D$_{{{0}}} \; / \; \sqrt{{\beta_{{{0}}}}} \quad \left[\sqrt{{\rm \mu m}}\right]$',
    "phase":              r'$\phi_{{{0}}} \quad [2\pi]$',
    "phasetot":           r'$\phi_{{{0}}} \quad [2\pi]$',
    "phase_milli":        r'$\phi_{{{0}}} \quad [2\pi\cdot10^{{-3}}]$',
    "dispersion":         r'D$_{{{0}}}$ [m]',
    "dispersion_mm":      r'D$_{{{0}}}$ [mm]',
    "co":                 r'Orbit {0} [m]',
    "co_mm":              r'Orbit {0} [mm]',
    "tune":               r'Q$_{{{0}}}$',
    "nattune":            r'Nat. Q$_{{{0}}}$',
    "chromamp":           r'W$_{{{0}}}$',
    "real":               r'$\Re({0})$',
    "imag":               r'$\Im({0})$',
    "absolute":           r'$\left|{0}\right|$',
}


def set_yaxis_label(param: str, plane: str, ax=None, delta: bool = False, chromcoup: bool = False) -> None:  # plot x and plot y
    """
    Set y-axis labels.

    Args:
        param: One of the labels defined in ``ylabel`` in this module.
        plane: Usually x or y, but can be any string actually to be placed into the label ({0}).
        ax: Axes to put the label on. Defaults to `` gca()``.
        delta: If True adds a Delta before the label. Defaults to ``False``.
    """
    if not ax:
        ax = plt.gca()
    try:
        label = ylabels[param].format(plane)
    except KeyError:
        raise ValueError(f"Label '{param}' not found.")

    if delta:
        if param.startswith("beta") or param.startswith("norm"):
            label = fr'$\Delta({label[1:-1]})$'
        else:
            label = fr'$\Delta {label[1:]}'

    if chromcoup:
        label = fr'{label[:-1] }/\Delta\delta$'

    ax.set_ylabel(label)


def set_xaxis_label(ax=None) -> None:
    """
    Sets the standard x-axis label

    Args:
        ax: Axes to put the label on. Defaults to ``gca()``.
    """
    if not ax:
        ax = plt.gca()
    ax.set_xlabel(r'Longitudinal location [m]')


def show_ir(ip_dict, ax: Axes = None, mode: str = "inside") -> None:
    """
    Plots the interaction regions into the background of the plot.

    Args:
        ip_dict: `dict`, `dataframe` or `series` containing ``IPLABEL`` : ``IP_POSITION``.
        ax:  Axes to put the irs on. Defaults to gca()``.
        mode: ``inside``, ``outside`` + ``nolines`` or just ``lines``.
    """
    if ax is None:
        ax = plt.gca()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    lines = 'nolines' not in mode
    inside = 'inside' in mode
    lines_only = 'inside' not in mode and 'outside' not in mode and 'lines' in mode

    if isinstance(ip_dict, (pd.DataFrame, pd.Series)):
        if isinstance(ip_dict, pd.DataFrame):
            ip_dict = ip_dict.iloc[:, 0]
        d = {}
        for ip in ip_dict.index:
            d[ip] = ip_dict.loc[ip]
        ip_dict = d

    for ip in ip_dict.keys():
        if xlim[0] <= ip_dict[ip] <= xlim[1]:
            xpos = ip_dict[ip]

            if lines:
                ax.axvline(xpos, linestyle=':', color='grey', marker='', zorder=0)

            if not lines_only:
                ypos = ylim[not inside] + (ylim[1] - ylim[0]) * 0.01
                c = 'grey' if inside else matplotlib.rcParams["text.color"]
                ax.text(xpos, ypos, ip, color=c, ha='center', va='bottom')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def move_ip_labels(value, ax: Axes = None) -> None:
    """Moves IP labels according to max y * value."""
    if ax is None:
        ax = plt.gca()

    y_max = ax.get_ylim()[1]
    for t in ax.texts:
        if re.match(r"^IP\s*\d$", t.get_text()):
            x = t.get_position()[0]
            t.set_position((x, y_max * value))


def get_ip_positions(path: Path) -> dict[str, float]:
    """
    Returns a `dict` of IP positions from tfs-file of path.

    Args:
        path (str): `Path` to the tfs-file containing IP-positions.
    """
    df = tfs.read_tfs(path).set_index('NAME')
    ip_names = [f"IP{i:d}" for i in range(1, 9)]
    ip_pos = df.loc[ip_names, 'S'].to_numpy()
    return dict(zip(ip_names, ip_pos))


def set_name(name: str, fig_or_ax=None) -> None:
    """
    Sets the name of the figure or axes.

    Args:
        name (str): string to set as name.
        fig_or_ax: `Figure` or `Axes` to to use. If ``None`` is given, takes the current figure.
            Defaults to ``None``.
    """
    if not fig_or_ax:
        fig_or_ax = plt.gcf()
    try:
        fig_or_ax.figure.canvas.manager.set_window_title(name)
    except AttributeError:
        fig_or_ax.canvas.manager.set_window_title(name)


def get_name(fig_or_ax=None) -> str:
    """
    Returns the name of the figure or axes.

    Args:
        fig_or_ax: `Figure` or `Axes` to to use.If ``None`` is given, takes the current figure.
            Defaults to ``None``.
    """
    if not fig_or_ax:
        fig_or_ax = plt.gcf()
    try:
        return fig_or_ax.figure.canvas.get_window_title()
    except AttributeError:
        return fig_or_ax.canvas.get_window_title()


def set_annotation(text: str, ax: Axes = None, position: str = "right", pad: float = 0.02) -> None:
    """
    Writes an annotation on the top right of the axes.

    Args:
        text: The annotation.
        ax: `Axes` to set annotation on. If ``None`` is given, takes the current figure. Defaults
            to ``None``.
        position: ``left`` or ``right``.
        pad: padding to the axes.
    """
    if ax is None:
        ax = plt.gca()

    annotation = get_annotation(ax, by_reference=True)

    xpos = float(position == 'right')

    if annotation is None:
        ax.text(xpos, 1.0 + pad, text,
                ha=position,
                va='bottom',
                transform=ax.transAxes,
                label='plot_style_annotation')
    else:
        annotation.set_text(text)
        plt.setp(annotation, ha=position, x=xpos, y=1.0+pad)


def get_annotation(ax: Axes = None, by_reference: bool = True):
    """
    Returns the annotation set by ``set_annotation()``.

    Args:
        ax: `Axes` to get annotation from. If ``None`` is given, takes the current figure.
            Defaults to ``None``.
        by_reference (bool): If ``True``, returns the reference to the annotation, otherwise the
            text as string. Defaults to ``True``.
    """
    if not ax:
        ax = plt.gca()

    for c in ax.get_children():
        if c.get_label() == 'plot_style_annotation':
            if by_reference:
                return c
            else:
                return c.get_text()
    return None


def small_title(ax: Axes = None) -> None:
    """
    Alternative to annotation, which lets you use the title-functions.

    Args:
        ax: `Axes` to use. If ``None`` is given, takes the current axes. Defaults to ``None``.
    """
    if not ax:
        ax = plt.gca()

    # could not get set_title() to work properly, so one parameter at a time
    ax.title.set_position([1.0, 1.02])
    ax.title.set_transform(ax.transAxes)
    ax.title.set_fontsize(matplotlib.rcParams['font.size'])
    ax.title.set_fontweight(matplotlib.rcParams['font.weight'])
    ax.title.set_verticalalignment('bottom')
    ax.title.set_horizontalalignment('right')


def figure_title(text: str, ax: Axes = None, pad: float = 0, **kwargs) -> None:
    """
    Set the title all the way to the top.

    Args:
        text: Text for the title.
        ax: `Axes` to use. If ``None`` is given, takes the current axes. Defaults to ``None``.
        pad: Padding from border.
        kwargs: passed on to fontdict.
    """
    if not ax:
        ax = plt.gca()

    # could not get set_title() to work properly, so one parameter at a time
    fdict = dict(fontsize=matplotlib.rcParams['font.size'],
                 fontweight=matplotlib.rcParams['font.weight'],
                 va="top", ha="center")
    fdict.update(kwargs)
    ax.set_title(text, transform=ax.figure.transFigure, fontdict=fdict)

    ax.title.set_position([.5, float(fdict['va'] == "top") + pad * (-1)**(fdict['va'] == 'top')])


def get_legend_ncols(labels, max_length: int = 78):
    """Calculate the number of columns in legend dynamically."""
    return max([max_length/max([len(label) for label in labels]), 1])


def transpose_legend_order(ncol: int, handles=None, labels=None, ax: Axes = None):
    """ Reorder handles and labels, so that the legend order is transposed,
    i.e. the entries are row-first instead of column-first."""
    def reorder(items):
        return list(itertools.chain(*[items[i::ncol] for i in range(ncol)]))

    if handles is None or labels is None:
        if ax is None:
            ax = plt.gca()

        ax_handles, ax_labels = ax.get_legend_handles_labels()
        if handles is None:
            handles = ax_handles

        if labels is None:
            labels = ax_labels

    return reorder(handles), reorder(labels)


def make_top_legend(
    ax: Axes, ncol: int, frame: bool = False, handles=None, labels=None, pad: float = 0.02, transposed=False
):
    """Create a legend on top of the plot."""
    if ncol < 1:
        return

    if transposed:
        handles, labels = transpose_legend_order(ncol, handles, labels, ax)

    leg = ax.legend(
        handles=handles,
        labels=labels,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0+pad),
        fancybox=frame,
        shadow=frame,
        frameon=frame,
        ncol=ncol,
    )

    leg.axes.figure.canvas.draw()
    legend_width = leg.get_window_extent().transformed(leg.axes.transAxes.inverted()).width
    if legend_width > 1:
        x_shift = (legend_width - 1) / 2.
        ax.legend(handles=handles, labels=labels, loc='lower right',
                  bbox_to_anchor=(1.0 + x_shift, 1.0+pad),
                  fancybox=frame, shadow=frame, frameon=frame, ncol=ncol)

    return leg


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    """
    Order of Magnitude Formatter.

    To set a fixed order of magnitude and fixed significant numbers.
    As seen on: https://stackoverflow.com/a/42658124/5609590

    See: set_sci_magnitude
    """
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)


def set_sci_magnitude(ax, axis="both", order=0, fformat="%1.1f", offset=True, math_text=True):
    """
    Uses the OMMFormatter to set the scientific limits on axes.

    Args:
        ax: Plotting axes.
        axis (str): **x**, **y** or **both**.
        order (int): Magnitude Order.
        fformat (str): Format to use.
        offset (bool): Formatter offset.
        math_text (bool): Whether to use mathText.
    """
    oomf = OOMFormatter(order=order, fformat=fformat, offset=offset, mathText=math_text)

    if axis == "x" or axis == "both":
        ax.xaxis.set_major_formatter(oomf)

    if axis == "y" or axis == "both":
        ax.yaxis.set_major_formatter(oomf)

    ax.ticklabel_format(axis=axis, style="sci", scilimits=(order, order), useMathText=math_text)

    return ax


def get_fontsize_as_float(font_size: str | float) -> float:
    try:
        scale = {
            'xx-small': 0.579,
            'x-small': 0.694,
            'small': 0.833,
            'medium': 1.0,
            'large': 1.200,
            'x-large': 1.440,
            'xx-large': 1.728,
            'larger': 1.2,
            'smaller': 0.833,
            None: 1.0}[font_size]
    except KeyError:
        return font_size
    return scale * matplotlib.rcParams['font.size']
