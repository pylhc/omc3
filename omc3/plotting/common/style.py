"""
Plot Style
----------------------------

Helper functions to make the most awesome* plots out there.

* please feel free to add more stuff
"""
import matplotlib


REMOVE_ENTRY = "REMOVE ENTRY"  # id to remove entries in manual style


_PRESENTATION_PARAMS = {
    # u'axes.autolimit_mode': u'data',
    u'backend': u'pdf',
    u'axes.edgecolor': u'k',
    u'axes.facecolor': u'w',
    u'axes.grid': True,
    u'axes.grid.axis': u'both',
    u'axes.grid.which': u'major',
    u'axes.labelcolor': u'k',
    # u'axes.labelpad': 4.0,
    u'axes.labelsize': 22,
    u'axes.labelweight': u'normal',
    u'axes.linewidth': 1.8,
    # u'axes.titlepad': 16.0,
    u'axes.titlesize': u'xx-large',
    u'axes.titleweight': u'bold',
    u'figure.edgecolor': u'w',
    u'figure.facecolor': u'w',
    u'figure.figsize': [10.24, 7.68],
    u'figure.frameon': True,
    u'figure.titlesize': u'xx-large',
    u'figure.titleweight': u'normal',
    u'font.size': 20.0,
    u'font.stretch': u'normal',
    u'font.weight': u'normal',
    u'font.family': 'sans-serif',
    u'font.serif': ['Computer Modern Roman'],
    u'grid.alpha': .6,
    u'grid.color': u'#b0b0b0',
    u'grid.linestyle': u'--',
    u'grid.linewidth': 1,
    u'legend.edgecolor': u'0.8',
    u'legend.facecolor': u'inherit',
    u'legend.fancybox': True,
    u'legend.fontsize': 20.0,
    u'legend.framealpha': 0.9,
    u'legend.frameon': False,
    u'legend.handleheight': 0.7,
    u'legend.handlelength': 2.0,
    u'legend.handletextpad': 0.8,
    u'legend.labelspacing': 0.5,
    u'legend.loc': u'best',
    u'legend.markerscale': 1.2,
    u'legend.numpoints': 1,
    u'legend.scatterpoints': 1,
    u'legend.shadow': False,
    u'lines.antialiased': True,
    # u'lines.color': u'C0',
    u'lines.linestyle': u'-',
    u'lines.linewidth': 2,
    u'lines.marker': u'o',
    u'lines.markeredgewidth': 2,
    u'lines.markersize': 14.0,
    u'lines.solid_capstyle': u'projecting',
    u'lines.solid_joinstyle': u'round',
    u'markers.fillstyle': u'none',
    u'text.antialiased': True,
    u'text.color': u'k',
    u'xtick.alignment': u'center',
    u'xtick.bottom': True,
    u'xtick.color': u'k',
    u'xtick.direction': u'out',
    u'xtick.labelsize': u'medium',
    u'xtick.major.bottom': True,
    u'xtick.major.pad': 3.5,
    u'xtick.major.size': 3.5,
    u'xtick.major.top': True,
    u'xtick.major.width': 1.2,
    u'xtick.minor.bottom': True,
    u'xtick.minor.pad': 3.4,
    u'xtick.minor.size': 2.0,
    u'xtick.minor.top': True,
    u'xtick.minor.visible': False,
    u'xtick.minor.width': 1,
    u'xtick.top': False,
    u'ytick.alignment': u'center_baseline',
    u'ytick.color': u'k',
    u'ytick.direction': u'out',
    u'ytick.labelsize': u'medium',
    u'ytick.left': True,
    u'ytick.major.left': True,
    u'ytick.major.pad': 3.5,
    u'ytick.major.right': True,
    u'ytick.major.size': 3.5,
    u'ytick.major.width': 1.2,
    u'ytick.minor.left': True,
    u'ytick.minor.pad': 3.4,
    u'ytick.minor.right': True,
    u'ytick.minor.size': 2.0,
    u'ytick.minor.visible': False,
    u'ytick.minor.width': 1,
    # u'ytick.right': False
}


_STANDARD_PARAMS = {
    u'axes.autolimit_mode': u'data',
    u'axes.edgecolor': u'k',
    u'axes.facecolor': u'w',
    u'axes.grid': True,
    u'axes.grid.axis': u'both',
    u'axes.grid.which': u'major',
    u'axes.labelcolor': u'k',
    u'axes.labelpad': 4.0,
    u'axes.labelsize': u'medium',
    u'axes.labelweight': u'normal',
    u'axes.linewidth': 1.5,
    u'axes.titlepad': 6.0,
    u'axes.titlesize': u'x-large',
    u'axes.titleweight': u'bold',
    u'figure.edgecolor': u'w',
    u'figure.facecolor': u'w',
    u'figure.figsize': [10.24, 7.68],
    u'figure.frameon': True,
    u'figure.titlesize': u'large',
    u'figure.titleweight': u'normal',
    u'font.size': 15.0,
    u'font.stretch': u'normal',
    u'font.weight': u'normal',
    u'font.family': 'sans-serif',
    u'font.serif': ['Computer Modern Roman'],
    u'font.sans-serif': ['Computer Modern Sans serif'],
    u'grid.alpha': .6,
    u'grid.color': u'#b0b0b0',
    u'grid.linestyle': u'--',
    u'grid.linewidth': 0.6,
    u'legend.edgecolor': u'0.8',
    u'legend.facecolor': u'inherit',
    u'legend.fancybox': True,
    u'legend.fontsize': 16.,
    u'legend.framealpha': 0.8,
    u'legend.frameon': False,
    u'legend.handleheight': 0.7,
    u'legend.handlelength': 2.0,
    u'legend.handletextpad': 0.8,
    u'legend.labelspacing': 0.5,
    u'legend.loc': u'best',
    u'legend.markerscale': .8,
    u'legend.numpoints': 1,
    u'legend.scatterpoints': 1,
    u'legend.shadow': False,
    u'lines.antialiased': True,
    # u'lines.color': u'C0',
    u'lines.linestyle': u'-',
    u'lines.linewidth': 1.5,
    u'lines.marker': u'o',
    u'lines.markeredgewidth': 1.0,
    u'lines.markersize': 8.0,
    u'lines.solid_capstyle': u'projecting',
    u'lines.solid_joinstyle': u'round',
    u'markers.fillstyle': u'none',
    u'text.antialiased': True,
    u'text.color': u'k',
    u'xtick.alignment': u'center',
    u'xtick.bottom': True,
    u'xtick.color': u'k',
    u'xtick.direction': u'out',
    u'xtick.labelsize': u'medium',
    u'xtick.major.bottom': True,
    u'xtick.major.pad': 3.5,
    u'xtick.major.size': 3.5,
    u'xtick.major.top': True,
    u'xtick.major.width': 0.8,
    u'xtick.minor.bottom': True,
    u'xtick.minor.pad': 3.4,
    u'xtick.minor.size': 2.0,
    u'xtick.minor.top': True,
    u'xtick.minor.visible': False,
    u'xtick.minor.width': 0.6,
    u'xtick.top': False,
    u'ytick.alignment': u'center_baseline',
    u'ytick.color': u'k',
    u'ytick.direction': u'out',
    u'ytick.labelsize': u'medium',
    u'ytick.left': True,
    u'ytick.major.left': True,
    u'ytick.major.pad': 3.5,
    u'ytick.major.right': True,
    u'ytick.major.size': 3.5,
    u'ytick.major.width': 0.8,
    u'ytick.minor.left': True,
    u'ytick.minor.pad': 3.4,
    u'ytick.minor.right': True,
    u'ytick.minor.size': 2.0,
    u'ytick.minor.visible': False,
    u'ytick.minor.width': 0.6,
    u'ytick.right': False
}

STYLES = dict(standard=_STANDARD_PARAMS, presentation=_PRESENTATION_PARAMS)

# Style ######################################################################


def set_style(style='standard', manual=None):
    """Sets the style for all following plots.

    Args:
        style: Choose Style, either 'standard' or 'presentation'
        manual: Dict of manual parameters to update. Convention: "REMOVE_ENTRY" removes entry
    """
    try:
        params = STYLES[style].copy()
    except KeyError:
        raise ValueError(f"Style '{style}' not found.")

    if manual:
        for key, value in manual.items():
            if value == REMOVE_ENTRY:
                params.pop(key)
                manual.pop(key)
        params.update(manual)

    matplotlib.rcParams.update(params)
